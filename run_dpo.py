import argparse
import os
from dataclasses import dataclass, field
from typing import Dict, Optional
import torch
from datasets import Dataset, load_dataset, load_from_disk
from peft import LoraConfig, PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, HfArgumentParser, TrainingArguments
from transformers import BitsAndBytesConfig
from trl import DPOTrainer
from jinja2 import Template
from transformers import pipeline
from tqdm import tqdm
import pandas as pd
from peft import get_peft_model

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
    #base_model.config.use_cache = False
    
    # initialize peft config
    peft_config = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        target_modules=["q_proj", "v_proj"],
        bias="none",
        task_type="CAUSAL_LM",
    )
    
    #model_peft = PeftModel.from_pretrained(base_model, f"peft_checkpoint_{iter}")
    #base_model = model_peft.merge_and_unload()

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    
    # preprocess the ultrachat200k dataset (or any sharegpt format ds)
    dataset = load_from_disk(f"dataset_iter_{iter-1}")

    training_args = TrainingArguments(
        per_device_train_batch_size=16,
        #per_device_eval_batch_size=8,
        max_steps=1250,
        logging_steps=1,
        save_steps=2500,
        gradient_accumulation_steps=1,
        gradient_checkpointing=True,
        learning_rate=3e-6,
        #evaluation_strategy="steps",
        #eval_steps=100,
        output_dir="spin",
        report_to="wandb",
        #lr_scheduler_type=script_args.lr_scheduler_type,
        #warmup_steps=script_args.warmup_steps,
        #optim=script_args.optimizer_type,
        bf16=True,
        remove_unused_columns=False,
        run_name=f"spin_{iter}",
    )

    model_peft = get_peft_model(
        base_model,         
        peft_config
        )

    dpo_trainer = DPOTrainer(
        model=model_peft,
        ref_model=None,
        args=training_args,
        beta=DPO_BETA,
        train_dataset=dataset,
        #eval_dataset=,
        tokenizer=tokenizer,
        peft_config=peft_config,
    )
    dpo_trainer.train()
    dpo_trainer.model.save_pretrained(f"peft_checkpoint_{iter}")
