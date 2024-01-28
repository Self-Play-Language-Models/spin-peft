#!/usr/bin/env python
# coding: utf-8

# In[2]:


#get_ipython().system('pip install transformers datasets peft')


# In[3]:


import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft.tuners.lora import LoraConfig, LoraModel
from datasets import load_dataset

from transformers import Trainer


# Load model and tokenizer

# In[4]:


model_checkpoint = "Intel/neural-chat-7b-v1-1"
device = "cuda" if torch.cuda.is_available() else "cpu"
model = AutoModelForCausalLM.from_pretrained(model_checkpoint, device_map="auto").to(device)
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)


# In[ ]:





# Define lambda regularization parameter as per paper details

# In[5]:


lambda_reg = 0.1


# Placeholder for the dataset loading function

# In[92]:


ds = load_dataset("HuggingFaceH4/ultrachat_200k", split="train_sft[:2000]")


# In[93]:


print(ds[0])


# In[94]:


from jinja2 import Template
tstr = "{% for message in messages %}\n{% if message['role'] == 'user' %}\n{{ '<|user|>\n' + message['content'] + eos_token }}\n{% elif message['role'] == 'system' %}\n{{ '<|system|>\n' + message['content'] + eos_token }}\n{% elif message['role'] == 'assistant' %}\n{{ '<|assistant|>\n'  + message['content'] + eos_token }}\n{% endif %}\n{% if loop.last and add_generation_prompt %}\n{{ '<|assistant|>' }}\n{% endif %}\n{% endfor %}"

template = Template(tstr)

def preprocess(item):
    
    #conv = get_conversation_template("vicuna")
    #roles = {"human": 'user', "gpt": 'assistant'}

    add_generation_prompt = True  # You can set this according to your needs

    output = template.render(messages=item["messages"], add_generation_prompt=add_generation_prompt, eos_token='\n')


    return {"prompt": item["prompt"], "response": output}

ds = ds.map(preprocess, remove_columns=["messages", "prompt_id"])


# In[95]:


dataset = ds

def group_batch(batch):
    return {k: [v] for k, v in batch.items()}
dataset = dataset.map(group_batch, batched=True, batch_size=1)


# Define LoRA configuration

# In[96]:


lora_config = LoraConfig(
    r=32,  # rank of LoRA
    lora_alpha=64,  # scaling factor for initialization
    lora_dropout=0.05,
    bias="none",
)


# Wrap the model with LoRA layers for parameter-efficient training

# In[97]:


peft_model = LoraModel(model, lora_config, adapter_name="iter0").to(device)


# Define the compute_spin_loss function (unchanged from previous)

# In[98]:


def compute_spin_loss(model_logits_gt, opponent_logits_gt, model_logits_syn, opponent_logits_syn, lambda_reg):
    model_probs_gt = torch.nn.functional.softmax(model_logits_gt, dim=-1)
    model_probs_syn = torch.nn.functional.softmax(model_logits_syn, dim=-1)
    opponent_probs_gt = torch.nn.functional.softmax(opponent_logits_gt, dim=-1)
    opponent_probs_syn = torch.nn.functional.softmax(opponent_logits_syn, dim=-1)

    print(model_probs_gt.shape)

    if model_probs_gt.shape[1] < model_probs_syn.shape[1]:
        model_probs_syn = model_probs_syn[:, :model_probs_gt.shape[1]]

    if model_probs_gt.shape[1] > model_probs_syn.shape[1]:
        model_probs_gt = model_probs_gt[:, :model_probs_syn.shape[1]]
    if opponent_probs_gt.shape[1] < opponent_probs_syn.shape[1]:
        model_probs_syn = model_probs_syn[:, :model_probs_gt.shape[1]]

    if opponent_probs_gt.shape[1] > opponent_probs_syn.shape[1]:
        model_probs_gt = model_probs_gt[:, :model_probs_syn.shape[1]]

    # Calculate losses
    loss_gt = -torch.log(model_probs_gt / opponent_probs_gt)
    loss_syn = -torch.log(model_probs_syn / opponent_probs_syn)

    # Apply the logistic loss to the log odds ratio
    logistic_loss_gt = torch.log(1 + torch.exp(-lambda_reg * loss_gt))
    logistic_loss_syn = torch.log(1 + torch.exp(-lambda_reg * loss_syn))

    # Combine losses for the final spin loss
    spin_loss = logistic_loss_gt.mean() + logistic_loss_syn.mean()
    return spin_loss


# Training setup

# In[99]:


optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, peft_model.parameters()), lr=5e-6)
tokenizer.pad_token = tokenizer.eos_token 


# Training loop for T iterations

# In[101]:


from tqdm import tqdm
T = 3  # Set the number of iterations
for iteration in range(T):
    total_loss = 0
    
    # Disable adapter layers for the opponent model
    peft_model.disable_adapter_layers()
    
    synthetic_data = []
    for data in tqdm(dataset):
        prompt = data['prompt']
        # Tokenize and generate synthetic data using the opponent model
        prompt_ids = tokenizer(prompt, return_tensors='pt', padding=True, truncation=True).input_ids.to(device)
        with torch.no_grad():
            peft_model.eval()  # Set model to evaluation mode
            synthetic_response_ids = peft_model.generate(prompt_ids, max_length=50)
            synthetic_data.append(synthetic_response_ids)
    
    # Enable adapter layers for training the main player model
    peft_model.enable_adapter_layers()
    
    # Train the main player model using the synthetic data and real responses
    peft_model.train()  # Set model to training moderesponse_tensors = ppo_trainer.generate(query_tensors, **generation_kwargs)

    for i, data in tqdm(enumerate(dataset)):
        ground_truth = data['response']
        ground_truth_ids = tokenizer(ground_truth, return_tensors='pt', padding=True, truncation=True).input_ids.to(device)
        synthetic_response_ids = synthetic_data[i]

        # Calculate logits for ground truth and synthetic responses
        main_player_logits_gt = peft_model(ground_truth_ids).logits
        main_player_logits_syn = peft_model(synthetic_response_ids).logits

        # Get opponent's logits for synthetic responses (as they were generated before enabling LoRA)
        with torch.no_grad():
            opponent_logits_syn = peft_model(synthetic_response_ids).logits
        
        # Compute the loss (assuming the function is defined above)
        loss = compute_spin_loss(
            main_player_logits_gt, opponent_logits_syn, 
            main_player_logits_syn, opponent_logits_syn, 
            lambda_reg
        )
        total_loss += loss.item()

        # Backpropagation and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print(loss)
    
    # Print average loss
    average_loss = total_loss / len(dataset)
    print(f"Iteration {iteration + 1}/{T}, Average Loss: {average_loss}")


# Save the final model parameters

# In[ ]:


#final_model_params = torch.save(peft_model.state_dict(), "saved_weights.pt")

peft_model.save_pretrained("deez")
print("Training complete.")

