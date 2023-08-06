import torch
import wandb
import time
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
from tqdm import tqdm
import numpy as np
import pandas as pd
tqdm.pandas()

from datasets import load_dataset

import torch.nn.functional as F
import torch

from transformers import AutoTokenizer, pipeline

from trl.gpt2 import GPT2HeadWithValueModel
from trl.ppo import PPOTrainer, ppo_initialize
from trl.core import build_bert_batch_from_txt, listify_batch

import random

config = {
    "model_name": "gpt2-xl",
    "steps": 40000,
    "batch_size": 32,
    "forward_batch_size": 32,
    "ppo_epochs": 1,   
    "lr": 2.82e-6,
    "init_kl_coef":0.2,
    "target": 6,
    "horizon":10000,
    "gamma":0.99,
    "lam":0.95,
    "cliprange": .2,
    "cliprange_value":.2,
    "vf_coef":.1, 
    "topic": 'topic/topic.txt',
    "prompt": 'topic/prompt.txt',
}

device0 = torch.device("cuda:0")
device1 = torch.device("cuda:1")
print(device0)
print(device1)

sent_kwargs = {
    "return_all_scores": True,
    "function_to_apply": "none",
    "batch_size": config["forward_batch_size"]
}

gpt2_model = GPT2HeadWithValueModel.from_pretrained(config['model_name'])
gpt2_model_ref = GPT2HeadWithValueModel.from_pretrained(config['model_name'])

gpt2_tokenizer = AutoTokenizer.from_pretrained(config['model_name'])
gpt2_tokenizer.pad_token = gpt2_tokenizer.eos_token

gpt2_model.to(device0)
gpt2_model_ref.to(device0)
topic_pipe = pipeline("zero-shot-classification","facebook/bart-large-mnli", tokenizer='facebook/bart-large-mnli', device=0) # reward

############# for policy freezing ############
for module in [gpt2_model.transformer, gpt2_model.lm_head]:
    for param in module.parameters():
        param.requires_grad = False

gen_kwargs = {
    "top_k": 0.0,
    "top_p": 1.0,
    "do_sample": True,
    "pad_token_id": gpt2_tokenizer.eos_token_id,
    "max_new_tokens": 80,
    "temperature": 2.8
}

ppo_trainer = PPOTrainer(gpt2_model, gpt2_model_ref, gpt2_tokenizer, **config)
prompt = ppo_initialize(config['topic'], config['prompt'])
batch_size = config['batch_size']

for epoch in tqdm(range(500)):

    print(epoch)

    torch.cuda.empty_cache()
    logs = dict()
    game_data = dict()
    timing = dict()
    t0 = time.time()

    #### Get response from gpt2
    t = time.time()
    query_tensors = []
    response_tensors = []
    response_list = []
    prompt_list = random.sample(prompt, batch_size)
    for i in range(batch_size):
        query = gpt2_tokenizer.encode(prompt_list[i][1], return_tensors="pt").to(device0)
        response = gpt2_model.generate(query, **gen_kwargs)
        query_tensors.append(query.squeeze())
        response_tensors.append(response.squeeze())
        response_list.append([prompt_list[i][0], ':'.join(gpt2_tokenizer.decode(response.squeeze(), skip_special_tokens=True, cleaned_up_tokenization_spaces=False).split(':')[1:])[1:]])
    timing['time/get_response'] = time.time()-t

    # MNLI Score
    rewards = []
    for i in range(batch_size):
        prob = topic_pipe(response_list[i][1], response_list[i][0], multi_label = False)["scores"][0]
        logit = -np.log(1/prob -1)+4
        rewards.append(logit)
    rewards = torch.tensor(rewards).to(device0)
    timing['time/get_sentiment_preds'] = time.time()-t
    print('Total Rewards: ', torch.mean(rewards))

    # Run PPO training 
    t = time.time()
    rewards = rewards.cpu()
    rewards = rewards.to(device0)
    stats = ppo_trainer.step(query_tensors, response_tensors, rewards)
    timing['time/optimization'] = time.time()-t

    # Log everything
    timing['time/epoch'] = time.time()-t0
    logs.update(timing)
    logs.update(stats)
    logs['env/reward_mean'] = torch.mean(rewards).cpu().numpy()
    logs['env/reward_std'] = torch.std(rewards).cpu().numpy()
    logs['env/reward_dist'] = rewards.cpu().numpy()
    wandb.log(logs)

    num = num+1

os.makedirs(experiment_name)
gpt2_model.save_pretrained(experiment_name)
gpt2_tokenizer.save_pretrained(experiment_name)
