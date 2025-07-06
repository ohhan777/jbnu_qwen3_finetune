# load model and tokenizer
from transformers import AutoModelForCausalLM, AutoTokenizer
import os

# Set environment variable to suppress BNB_CUDA_VERSION warning
os.environ["BNB_CUDA_VERSION"] = ""

# model = AutoModelForCausalLM.from_pretrained("./output/checkpoint-1000", device_map="auto")
# tokenizer = AutoTokenizer.from_pretrained("./output/checkpoint-1000", device_map="auto")

model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-8B", device_map="auto")
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-8B", device_map="auto")

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


# load dataset
from datasets import load_dataset
dataset = load_dataset("FreedomIntelligence/medical-o1-reasoning-SFT", "en", split="train[2000:]")

question = dataset[10]['Question']
inputs = tokenizer(
    [inference_prompt_style.format(question) + tokenizer.eos_token],
    return_tensors="pt"
).to("cuda")

outputs = model.generate(
    input_ids=inputs.input_ids,
    attention_mask=inputs.attention_mask,
    max_new_tokens=32768,
    eos_token_id=tokenizer.eos_token_id,
    use_cache=True,
)
response = tokenizer.batch_decode(outputs, skip_special_tokens=True)
print(inference_prompt_style.format(question))
print(response[0].split("### Response:")[1])