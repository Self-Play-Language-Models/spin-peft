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
import re 
import random


def maybe_insert_system_message(messages, tokenizer):
    if messages[0]["role"] == "system":
        return

    # chat template can be one of two attributes, we check in order
    chat_template = tokenizer.chat_template
    if chat_template is None:
        chat_template = tokenizer.default_chat_template

    # confirm the jinja template refers to a system message before inserting
    if "system" in chat_template:
        messages.insert(0, {"role": "system", "content": ""})



def apply_chat_template(
    example,
    tokenizer,
    task,
):
    if task in ["sft", "generation"]:
        messages = example["messages"]
        # We add an empty system message if there is none
        maybe_insert_system_message(messages, tokenizer)
        example["text"] = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True if task == "generation" else False
        )
        substring = "\<\|assistant\|\>"
        random_match = random.choice([match.end() for match in re.finditer(substring, example["text"], flags=re.IGNORECASE)])
        example["prompt"] = example["text"][:random_match]
        example["response"] = example["text"][random_match:]
        
    elif task == "rm":
        if all(k in example.keys() for k in ("chosen", "rejected")):
            chosen_messages = example["chosen"]
            rejected_messages = example["rejected"]
            # We add an empty system message if there is none
            maybe_insert_system_message(chosen_messages, tokenizer)
            maybe_insert_system_message(rejected_messages, tokenizer)

            example["text_chosen"] = tokenizer.apply_chat_template(chosen_messages, tokenize=False)
            example["text_rejected"] = tokenizer.apply_chat_template(rejected_messages, tokenize=False)
        else:
            raise ValueError(
                f"Could not format example as dialogue for `rm` task! Require `[chosen, rejected]` keys but found {list(example.keys())}"
            )
    elif task == "dpo":
        if all(k in example.keys() for k in ("chosen", "rejected")):
            # For DPO, the inputs are triples of (prompt, chosen, rejected), where `chosen` and `rejected` are the final turn of a dialogue
            # We therefore need to extract the N-1 turns to form the prompt
            prompt_messages = example["chosen"][:-1]
            # Prepend a system message if the first message is not a system message
            if example["chosen"][0]["role"] != "system":
                prompt_messages.insert(0, {"role": "system", "content": ""})
            # Now we extract the final turn to define chosen/rejected responses
            chosen_messages = example["chosen"][-1:]
            rejected_messages = example["rejected"][-1:]
            example["text_chosen"] = tokenizer.apply_chat_template(chosen_messages, tokenize=False)
            example["text_rejected"] = tokenizer.apply_chat_template(rejected_messages, tokenize=False)
            example["text_prompt"] = tokenizer.apply_chat_template(prompt_messages, tokenize=False)
        else:
            raise ValueError(
                f"Could not format example as dialogue for `dpo` task! Require `[chosen, rejected]` keys but found {list(example.keys())}"
            )
    else:
        raise ValueError(
            f"Task {task} not supported, please ensure that the provided task is one of {['sft', 'generation', 'rm', 'dpo']}"
        )
    return example

    
# given a model, generate a synthetic 
def generate_data(model, dataset, tokenizer, batch_size=16):
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
            generated_ids = model.generate(**encoding, max_length=1024)
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

    if iter != 0:
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
    dataset = load_dataset("HuggingFaceH4/ultrachat_200k", split="train_sft[:5000]")
    
    dataset = dataset.map(
        apply_chat_template,
        fn_kwargs={"tokenizer": tokenizer, "task": "sft"},
        num_proc=16,
        desc="Formatting comparisons with prompt template")
    #dataset = dataset.map(apply_chat_template, remove_columns=["messages", "prompt_id"])
    orig_dataset = generate_data(base_model, dataset, tokenizer)

    print(orig_dataset)

    orig_dataset.save_to_disk(f"dataset_iter_{iter}")

