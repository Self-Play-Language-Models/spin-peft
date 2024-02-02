import argparse
import os
from dataclasses import dataclass, field
from typing import Dict, Optional
import torch
from datasets import Dataset, load_dataset
from peft import LoraConfig, PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, HfArgumentParser, TrainingArguments
from transformers import BitsAndBytesConfig
from trl import DPOTrainer
from jinja2 import Template
from transformers import pipeline
from tqdm import tqdm
import pandas as pd

# given a model, generate a synthetic 
def generate_data(model, dataset, tokenizer, batch_size=8):
    #text_generator = pipeline("text-generation", model=llama_model, tokenizer=tokenizer)  # Use appropriate device index

    prompts = dataset["prompt"]
    responses = dataset["response"]

    gt_responses = []
    generated_prompts = []
    generated_responses = []

    for i in tqdm(range(0, len(prompts), batch_size)):
        batch_prompts = prompts[i:i+batch_size]
        batch_responses = prompts[i:i+batch_size]
        encoding = tokenizer(batch_prompts, padding=True, return_tensors='pt').to('cuda:0')
        
        with torch.no_grad():
            generated_ids = model.generate(**encoding)
        generated_texts = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)

        batch_generated_responses = generated_texts
        generated_responses.extend(batch_generated_responses)

    print(len(prompts))
    print(len(responses))
    print(len(generated_responses))
    generated_data = Dataset.from_pandas(pd.DataFrame({"prompt": prompts, "chosen": responses, "rejected": generated_responses}))

    return generated_data

if __name__ == "__main__":
    MODEL_NAME = "alignment-handbook/zephyr-7b-sft-full"
    DATASET_NAME = "ultrachat200k"
    LR = 5e-4
    
    LORA_R = 32
    LORA_ALPHA = 16
    LORA_DROPOUT = 0.05
    
    DPO_BETA = 1
    SPIN_ITER = 4
    
    # Create an ArgumentParser object
    parser = argparse.ArgumentParser(description='generate dataset for SPIN.')
    
    # Add arguments
    parser.add_argument('iter', type=int, help='Description of argument 2')
    
    # Parse the command-line arguments
    args = parser.parse_args()
    
    iter = args.iter
    
    # initialize the base model
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
    )
    
    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,        # "meta-llama/Llama-2-7b-hf"
        #quantization_config=bnb_config,
        device_map={"": 0},
        trust_remote_code=True,
        #use_auth_token=True,
    )
    base_model.config.use_cache = False

    if iter != 10:
        # initialize peft config
        peft_config = LoraConfig(
            r=LORA_R,
            lora_alpha=LORA_ALPHA,
            lora_dropout=LORA_DROPOUT,
            target_modules=["q_proj", "v_proj"],
            bias="none",
            task_type="CAUSAL_LM",
        )
        
        model_peft = PeftModel.from_pretrained(base_model, f"peft_checkpoint_{iter}")
    
        
        base_model = model_peft.merge_and_unload()

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    
    # preprocess the ultrachat200k dataset (or any sharegpt format ds)
    dataset = load_dataset("HuggingFaceH4/ultrachat_200k", split="train_sft")
    
    tstr = "{% for message in messages %}\n{% if message['role'] == 'user' %}\n{{ '<|user|>\n' + message['content'] + eos_token }}\n{% elif message['role'] == 'system' %}\n{{ '<|system|>\n' + message['content'] + eos_token }}\n{% elif message['role'] == 'assistant' %}\n{{ '<|assistant|>\n'  + message['content'] + eos_token }}\n{% endif %}\n{% if loop.last and add_generation_prompt %}\n{{ '<|assistant|>' }}\n{% endif %}\n{% endfor %}"
    template = Template(tstr)
    
    def preprocess(item):
        output = template.render(messages=item["messages"], add_generation_prompt=True, eos_token='\n')
        return {"prompt": item["prompt"], "response": output}
    
    dataset = dataset.map(preprocess, remove_columns=["messages", "prompt_id"])
    orig_dataset = generate_data(base_model, dataset, tokenizer)

    print(orig_dataset)

    orig_dataset.save_to_disk(f"dataset_iter_{iter}")

