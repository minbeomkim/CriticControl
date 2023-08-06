import torch
import wandb
import time
import os
from tqdm import tqdm
import numpy as np
import pandas as pd
from collections import Counter
tqdm.pandas()

from datasets import load_dataset
import datasets

from transformers import GPT2Tokenizer, T5Tokenizer, BertTokenizer, BartForConditionalGeneration, PreTrainedTokenizerFast, BartTokenizer
from transformers import AutoModelForSequenceClassification, AutoTokenizer, T5ForConditionalGeneration, BertForSequenceClassification, GPT2LMHeadModel
from transformers import AutoTokenizer, pipeline, top_k_top_p_filtering
import torch.nn.functional as F
import torch

from trl.gpt2 import GPT2HeadWithValueModel, topic_generation
from trl.ppo import PPOTrainer, ppo_initialize
from trl.core import build_bert_batch_from_txt, listify_batch

from evaluate import load
from rouge_score import rouge_scorer, scoring
from nltk.tokenize import sent_tokenize
from distinct import distinct

config = {
    "model_name": "model/gpt2-xl-critic",
    "batch_size": 1,
    "forward_batch_size": 1,
    "ppo_epochs": 4,
    "lr": 1.41e-5,
    "init_kl_coef":0.2,
    "target": 6,
    "horizon":10000,
    "gamma":1,
    "lam":0.95,
    "cliprange": .2,
    "cliprange_value":.2,
    "vf_coef":.1, 
    "topic": 'topic/topic.txt',
    "prompt": 'topic/prompt.txt',
}

device0 = torch.device("cuda:0")
device1 = torch.device("cuda:1")

sent_kwargs = {
    "return_all_scores": True,
    "function_to_apply": "none",
    "batch_size": config["forward_batch_size"]
}

gpt2_model = GPT2HeadWithValueModel.from_pretrained(config['model_name'])
gpt2_tokenizer = AutoTokenizer.from_pretrained(config['model_name'])
gpt2_tokenizer.pad_token = gpt2_tokenizer.eos_token
prompt = ppo_initialize(config['topic'], config['prompt'])

gpt2_model.to(device0)


###################### Experiment Setting ###################

result_data = dict()

#### get response from gpt2 and gpt2_ref
with torch.no_grad():

    #### Get response from gpt2
    response_list = []
    for i in range(len(prompt)):
        query = gpt2_tokenizer.encode(prompt[i][1], return_tensors="pt").to(device0)
        response = topic_generation(gpt2_model, query)
        response_result = ':'.join(gpt2_tokenizer.decode(response.squeeze(), skip_special_tokens=True).split(':')[1:])[1:]
        response_list.append([prompt[i][0], response_result.replace("\n", " ")])

    sentences = []
    topics = []
    for i in range(len(prompt)):
        sentences.append(response_list[i][1])
        topics.append(response_list[i][0])

result_data['sentences'] = sentences
result_data['topics'] = topics
        
df_results = pd.DataFrame(result_data)
save = df_results.to_json('json/topic.json', orient='table')
