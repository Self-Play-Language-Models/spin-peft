#!/usr/bin/env python
# coding: utf-8

# In[2]:


#get_ipython().system('pip3 install datasets peft transformers accelerate trl torch bitsandbytes accelerate')


# In[1]:


import os
from dataclasses import dataclass, field
from typing import Dict, Optional

import torch
from datasets import Dataset, load_dataset
from peft import LoraConfig, PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, HfArgumentParser, TrainingArguments
from transformers import BitsAndBytesConfig

from trl import DPOTrainer


# In[2]:


#get_ipython().system('pip3 install accelerate')


# In[3]:


MODEL_NAME = "alignment-handbook/zephyr-7b-sft-full"
DATASET_NAME = "ultrachat200k"
LR = 5e-4

LORA_R = 32
LORA_ALPHA = 16
LORA_DROPOUT = 0.05

DPO_BETA = 1


# In[62]:


from jinja2 import Template

# preprocess the ultrachat200k dataset (or any sharegpt format ds)
dataset = load_dataset("HuggingFaceH4/ultrachat_200k", split="train_sft[:40]")

tstr = "{% for message in messages %}\n{% if message['role'] == 'user' %}\n{{ '<|user|>\n' + message['content'] + eos_token }}\n{% elif message['role'] == 'system' %}\n{{ '<|system|>\n' + message['content'] + eos_token }}\n{% elif message['role'] == 'assistant' %}\n{{ '<|assistant|>\n'  + message['content'] + eos_token }}\n{% endif %}\n{% if loop.last and add_generation_prompt %}\n{{ '<|assistant|>' }}\n{% endif %}\n{% endfor %}"
template = Template(tstr)

def preprocess(item):
    output = template.render(messages=item["messages"], add_generation_prompt=True, eos_token='\n')
    return {"prompt": item["prompt"], "response": output}

dataset = dataset.map(preprocess, remove_columns=["messages", "prompt_id"])


# In[54]:


bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
)

base_model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,        # "meta-llama/Llama-2-7b-hf"
    quantization_config=bnb_config,
    device_map={"": 0},
    trust_remote_code=True,
    #use_auth_token=True,
)
base_model.config.use_cache = False

# initialize peft config
peft_config = LoraConfig(
    r=LORA_R,
    lora_alpha=LORA_ALPHA,
    lora_dropout=LORA_DROPOUT,
    target_modules=["q_proj", "v_proj"],
    bias="none",
    task_type="CAUSAL_LM",
)


# In[70]:


tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, pad_token_id=tokenizer.eos_token_id, use_fast=True)


# In[73]:


from transformers import pipeline
from tqdm import tqdm
# given a model, generate a synthetic 
def generate_data(llama_model, dataset, batch_size=1):
    text_generator = pipeline("text-generation", model=llama_model, tokenizer=tokenizer)  # Use appropriate device index

    prompts = dataset["prompt"]
    responses = dataset["response"]

    generated_prompts = []
    generated_responses = []

    for i in tqdm(range(0, len(prompts), batch_size)):
        batch_prompts = prompts[i:i+batch_size]

        print(batch_prompts)

        batch_responses = text_generator(batch_prompts, max_length=512)

        #batch_generated_responses = [response[0]['generated_text'] for response in batch_responses]
        #batch_responses = prompts[i:i+batch_size]

        #generated_prompts.extend(batch_prompts)
        #generated_responses.extend(batch_generated_responses)

    generated_data = pd.DataFrame({"question": generated_prompts, "response_j": batch_responses, "response_k": generated_responses})

    return generated_data


# In[74]:


orig_dataset = generate_data(model, dataset)


# In[46]:


from peft import get_peft_model

"""
model = AutoPeftModelForCausalLM.from_pretrained(
    MODEL_NAME, # location of saved SFT model
    low_cpu_mem_usage=True,
    torch_dtype=torch.float16,
    load_in_4bit=True,
    is_trainable=True,
)
"""


model =  base_model
model_peft = get_peft_model(model, peft_config)

dataset = orig_dataset

for i in range(SPIN_ITER):

    model_peft = get_peft_model(model, peft_config)

    print("Training")
    dpo_trainer = DPOTrainer(
        model=model_peft,
        ref_model=None,
        args=training_args,
        beta=DPO_BETA,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        peft_config=peft_config,
    )
    dpo_trainer.train()
    dpo_trainer.save_pretrained(f"peft_checkpoint_{i}")

    print("Generating Data")

    dataset = generate_data(peft_model, orig_dataset)

    model = peft_model.unload()
    


# In[ ]:





# In[ ]:





# In[ ]:




