# load model and tokenizer
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, AutoConfig, TextIteratorStreamer
from threading import Thread
from peft import PeftModel
import torch
import os
import sys
# Set environment variable to suppress BNB_CUDA_VERSION warning
os.environ["BNB_CUDA_VERSION"] = ""


def load_model_and_tokenizer(model_path, use_lora=True):
    """Load model and tokenizer with LoRA support"""
    
    # Set up quantization config (same as training)
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=False,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    
    if use_lora:
        # Load tokenizer from base model
        lora_cfg_pretrained = AutoConfig.from_pretrained(model_path)
        tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-8B", use_fast=False)
        
        # Load base model
        base_model = AutoModelForCausalLM.from_pretrained(
            "Qwen/Qwen3-8B",  # Base model path
            quantization_config=bnb_config,   
            device_map="auto",  
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
        )
        
        # Load LoRA adapter
        model = PeftModel.from_pretrained(base_model, model_path)
    else:
        # Load tokenizer and full fine-tuned model
        tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            quantization_config=bnb_config,   
            device_map="auto",  
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
        )
    
    model.config.use_cache = True  # Enable cache for inference
    return model, tokenizer

# Model paths
MODEL_PATH = "./output/Qwen3-8B-Medical-SFT"  # Path to your trained model
USE_LORA = True  # Set to False if you trained without LoRA

# Load model and tokenizer
model, tokenizer = load_model_and_tokenizer(MODEL_PATH, use_lora=USE_LORA)

inference_prompt_style = """Below is an instruction that describes a task, paired with an input that provides further context. 
    Write a response that appropriately completes the request. 
    Before answering, think carefully about the question and create a step-by-step chain of thoughts to ensure a logical and accurate response.

    ### Instruction:
    You are a medical expert with advanced knowledge in clinical reasoning, diagnostics, and treatment planning. 
    Please answer the following medical question. 

    ### Question:
    {}

    ### Response:
    <think>

    """

def generate_response(question):
    """Generate streaming response for a given question"""
    
    # Format the prompt
    prompt = inference_prompt_style.format(question)
    
    # Tokenize input
    inputs = tokenizer(
        [prompt],
        return_tensors="pt",
        truncation=True,
        max_length=2048
    ).to(model.device)
    
    streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

    # Generate response in a separate thread
    generation_kwargs = {
        "input_ids": inputs.input_ids,
        "attention_mask": inputs.attention_mask,
        "max_new_tokens": 1024,
        "temperature": 0.7,
        "do_sample": True,
        "pad_token_id": tokenizer.eos_token_id,
        "eos_token_id": tokenizer.eos_token_id,
        "use_cache": True,
        "streamer": streamer,
    }
    
    thread = Thread(target=model.generate, kwargs=generation_kwargs)
    thread.start()
    
    # Yield new tokens as they are generated
    for new_token in streamer:
        yield new_token
    
    thread.join()

# Test with sample questions
if __name__ == "__main__":
    # Load dataset for testing
    from datasets import load_dataset
    dataset = load_dataset("FreedomIntelligence/medical-o1-reasoning-SFT", "en", split="train[2000:]")
    
    # Test with a sample question
    question = dataset[0]['Question']
    print("=" * 80)
    print("[Question]")
    print(question)
    print("\n" + "=" * 80)
    print("[Think Mode]")
    
    current_text = ""
    think_ended = False
    
    for new_token in generate_response(question):
        current_text += new_token
        
        # Check if we've found </think> and haven't processed it yet
        if "</think>" in current_text and not think_ended:
            # Split at </think>
            parts = current_text.split("</think>")
            think_part = parts[0]
            answer_part = parts[1] if len(parts) > 1 else ""
            
            # Print think part
            print(think_part)
            print("\n" + "=" * 40)
            print("[Answer]")
            
            # Print answer part if it exists
            if answer_part:
                print(answer_part, end="", flush=True)
            
            think_ended = True
        elif think_ended:
            # We're in answer mode, just print the new token
            print(new_token, end="", flush=True)
        else:
            # We're still in think mode, print the new token
            print(new_token, end="", flush=True)
        
        sys.stdout.flush()

    print()  # Final newline
