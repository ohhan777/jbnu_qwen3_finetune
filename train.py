import os
import transformers
from dataclasses import dataclass, field
from transformers import AutoModelForCausalLM, AutoTokenizer, DataCollatorForLanguageModeling, Trainer
import torch
from typing import Optional
from transformers import TrainingArguments
from datasets import load_dataset
import pathlib

# Set environment variable to suppress BNB_CUDA_VERSION warning
os.environ["BNB_CUDA_VERSION"] = ""


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="Qwen/Qwen3-8B")

@dataclass
class DataArguments:
    name: str = field(default="FreedomIntelligence/medical-o1-reasoning-SFT", metadata={"help": "Path to the training data, in llava's instruction.json format. Supporting multiple json files via /path/to/{a,b,c}.json"})
    language: str = field(default="en", metadata={"help": "Language of the dataset."})
    split: str = field(default="train[0:2000]", metadata={"help": "Split of the dataset to use."})
    trust_remote_code: bool = field(default=True, metadata={"help": "Whether to trust remote code."})

@dataclass
class TrainingArguments(transformers.TrainingArguments):
    output_dir: Optional[str] = field(default="output/Qwen3-8B-Medical-SFT2")
    per_device_train_batch_size: int = field(default=1)
    per_device_eval_batch_size: int = field(default=1)
    gradient_accumulation_steps: int = field(default=2)
    optim: str = field(default="paged_adamw_32bit")
    num_train_epochs: int = field(default=1)
    logging_steps: float = field(default=0.05)
    warmup_steps: int = field(default=10)
    logging_strategy: str = field(default="steps")
    learning_rate: float = field(default=2e-4)
    label_names: list[str] = field(default_factory=lambda: ["labels"])
    quant_type: str = field(default="nf4", metadata={"help": "Quantization data type to use. Should be one of `fp4` or `nf4`."})
    bits: int = field(default=4, metadata={"help": "How many bits to use."})
    lora_enable: bool = True
    lora_r: int = 64
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    lora_weight_path: str = ""
    lora_bias: str = "none"
    fp16: bool = False
    bf16: bool = False
    group_by_length: bool = True
    report_to: str = "none"


def make_supervised_data_module(tokenizer, data_args):
    train_prompt_style = """Below is an instruction that describes a task, paired with an input that provides further context. 
        Write a response that appropriately completes the request. 
        Before answering, think carefully about the question and create a step-by-step chain of thoughts to ensure a logical and accurate response.

        ### Instruction:
        You are a medical expert with advanced knowledge in clinical reasoning, diagnostics, and treatment planning. 
        Please answer the following medical question. 

        ### Question:
        {}

        ### Response:
        <think>
        {}
        </think>
        {}"""
    
    def formatting_prompts_func(examples):
        inputs = examples["Question"]
        complex_cots = examples["Complex_CoT"]
        outputs = examples["Response"]
        texts = []
        for question, cot, response in zip(inputs, complex_cots, outputs):
            # Append the EOS token to the response if it's not already there
            if not response.endswith(tokenizer.eos_token):
                response += tokenizer.eos_token
            text = train_prompt_style.format(question, cot, response)
            texts.append(text)
        
        # Tokenize the texts
        tokenized = tokenizer(texts, truncation=True, padding=True, return_tensors="pt")
        
        # Create labels (same as input_ids for causal language modeling)
        tokenized["labels"] = tokenized["input_ids"].clone()
        
        return tokenized
    
    train_dataset = load_dataset(
        data_args.name,
        data_args.language,
        split=data_args.split,
        trust_remote_code=data_args.trust_remote_code,
    )
    train_dataset = train_dataset.map(
        formatting_prompts_func,
        batched=True,
    )
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
        return_tensors="pt",
    )

    return train_dataset, data_collator




def get_model(model_args, bnb_config):
    
    model = AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        quantization_config=bnb_config,   
        device_map="auto",  
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )

    return model


def safe_save_model_for_hf_trainer(trainer, output_dir):
    state_dict = trainer.model.state_dict()
    if trainer.args.should_save:
        cpu_state_dict = {key: value.cpu() for key, value in state_dict.items()}
        del state_dict
        os.makedirs(output_dir, exist_ok=True)
        trainer.model.save_pretrained(output_dir, state_dict=cpu_state_dict, safe_serialization=True)
        # torch.save(cpu_state_dict, os.path.join(output_dir, "finetuned_model.pt"))


def get_peft_state_dict(model, bias):
    """Get LoRA state dict without zero_3 dependency"""
    if bias == "none":
        to_return = {k: t for k, t in model.named_parameters() if "lora_" in k}
    elif bias == "all":
        to_return = {k: t for k, t in model.named_parameters() if "lora_" in k or "bias" in k}
    elif bias == "lora_only":
        to_return = {}
        maybe_lora_bias = {}
        lora_bias_names = set()
        for k, t in model.named_parameters():
            if "lora_" in k:
                to_return[k] = t
                bias_name = k.split("lora_")[0] + "bias"
                lora_bias_names.add(bias_name)
            elif "bias" in k:
                maybe_lora_bias[k] = t
        for k, t in maybe_lora_bias.items():
            if k in lora_bias_names:
                to_return[k] = t
    else:
        raise NotImplementedError
    return to_return


def get_peft_state_non_lora_dict(named_params, require_grad_only=True):
    """Get non-LoRA state dict without zero_3 dependency"""
    to_return = {k: t for k, t in named_params if "lora_" not in k}
    if require_grad_only:
        to_return = {k: t for k, t in to_return.items() if t.requires_grad}
    to_return = {k: v.cpu() for k, v in to_return.items()}
    return to_return
        



def train():

    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    bnb_config = None
    if training_args.bits in [4, 8]:
        from transformers import BitsAndBytesConfig
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=False,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )

    model = get_model(model_args, bnb_config)
    model.config.use_cache = False
    model.config.pretraining_tp = 1


    tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path, use_fast=True)

    # LoRA
    lora_config = None
    if training_args.lora_enable:
        from peft import LoraConfig, get_peft_model
        lora_config = LoraConfig(
            r=training_args.lora_r,
            lora_alpha=training_args.lora_alpha,
            lora_dropout=training_args.lora_dropout,
            bias=training_args.lora_bias,
            task_type="CAUSAL_LM",
            target_modules=[
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                "gate_proj",
                "up_proj",
                "down_proj",
            ],  # Target modules for LoRA
        )
        model = get_peft_model(model, lora_config)

        training_dataset, data_collator = make_supervised_data_module(tokenizer, data_args)   
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=training_dataset,
        data_collator=data_collator,
        )
    

    if list(pathlib.Path(training_args.output_dir).glob("checkpoint-*")):
        trainer.train(resume_from_checkpoint=True)
    else:
        trainer.train()
    trainer.save_state()

    if training_args.lora_enable:
        # Get LoRA state dict without zero_3
        state_dict = get_peft_state_dict(model, training_args.lora_bias)
        non_lora_state_dict = get_peft_state_non_lora_dict(model.named_parameters())
        if training_args.local_rank == 0 or training_args.local_rank == -1:
            if hasattr(model, "config"):
                model.config.save_pretrained(training_args.output_dir)
            if hasattr(model, "generation_config"):
                model.generation_config.save_pretrained(training_args.output_dir)
            model.save_pretrained(training_args.output_dir, state_dict=state_dict)
            torch.save(non_lora_state_dict, os.path.join(training_args.output_dir, "non_lora_trainables.bin"))
    else:
        safe_save_model_for_hf_trainer(trainer=trainer, output_dir=training_args.output_dir)



if __name__ == "__main__":
    train()
