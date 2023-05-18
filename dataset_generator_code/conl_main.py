import os
import json
import argparse
import pickle as pc
from collections import defaultdict
import torch
import random
import numpy as np
import itertools
from copy import deepcopy
from tqdm import tqdm

from transformers import AutoTokenizer, AutoModelForSequenceClassification
from constant import *

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--max-tokens', default=128, type=int)
    parser.add_argument('--freq-penalty', default=0.4, type=float)
    parser.add_argument('--pres-penalty', default=0.4, type=float)
    parser.add_argument('--top-p', default=1, type=float)
    parser.add_argument('--temp', default=0.8, type=float)
    parser.add_argument('--num-trial', default=5, type=int)
    parser.add_argument('--datatype', default='train', type=str)
    parser.add_argument('--profile-save-dir', default=None, type=str)

    return parser.parse_args()

def define_models():
    consistent_clf_path = 'ynie/roberta-large_conv_contradiction_detector_v0'
    consistent_tokenizer = AutoTokenizer.from_pretrained(consistent_clf_path)
    consistent_clf = AutoModelForSequenceClassification.from_pretrained(consistent_clf_path)

    return consistent_tokenizer, consistent_clf

tokenizer, model = define_models()
device = 'cuda'
model.to(device)
model.eval()

def get_consistency(premise, hyp):
    tokenized_input_seq_pair = tokenizer.encode_plus(premise, hyp, max_length=128, return_token_type_ids=True, truncation=True)
    input_ids = torch.Tensor(tokenized_input_seq_pair['input_ids']).long().unsqueeze(0).to(device)
    # remember bart doesn't have 'token_type_ids', remove the line below if you are using bart.
    token_type_ids = torch.Tensor(tokenized_input_seq_pair['token_type_ids']).long().unsqueeze(0).to(device)
    attention_mask = torch.Tensor(tokenized_input_seq_pair['attention_mask']).long().unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(input_ids,
                        attention_mask=attention_mask,
                        token_type_ids=token_type_ids,
                        labels=None)
        
        predicted_prob = torch.softmax(outputs[0], dim=1)[0].tolist() # batch_size only one

    non_contradiction = predicted_prob[0]
    contradiction = predicted_prob[1]
    if contradiction == None or contradiction == 'None':
        print(premise, hyp)
        assert 1 == 0
    return contradiction

def sampling_persona_set(sentences):
    # 1. sample a sentence from category pool        
    temp_sampled_sents = defaultdict(str)
    total_size = len(sentences)
    categorymap = {}
    for t, (k, v) in enumerate(sentences.items()):
        categorymap[t] = k
        sampled_sent = random.sample(v, 1)
        temp_sampled_sents[k] = sampled_sent[0]['utter']
        sentences[k].remove(sampled_sent[0])
        
    group_categorymap = {'demographic': [], 'psychographic': [], 'wellness': []}
    for k, v in categorymap.items():
        if v in TARGET_DEMOGRAPHIC_ATTRMAP.keys():
            group_categorymap['demographic'].append(k)
        elif v in TARGET_PSYCHOGRAPHICS_ATTRMAP.keys():
            group_categorymap['psychographic'].append(k)
        else:
            group_categorymap['wellness'].append(k)
        print(k, v)
    print(group_categorymap)

    # 2-1. init contradict matrix
    contradict_matrix = []
    sent_matrix = []
    sent_map = {}
    col_size = len(temp_sampled_sents.keys())
    for i in range(col_size):
        tmp = [0.0 for j in range(col_size)]
        tmp_sent = ['Nope' for j in range(col_size)]
        contradict_matrix.append(tmp)
        sent_matrix.append(tmp_sent)
        
    for i, (k, sent) in enumerate(temp_sampled_sents.items()):
        sent_map[i] = sent
    
    for i, (k, sent) in enumerate(tqdm(temp_sampled_sents.items())):
        for j, (k_j, sent_j) in enumerate(tqdm(temp_sampled_sents.items())):
            
            if i == j or j < i:
                continue
            else:
                def _check_contradiction(sent1, sent2):
                    
                    contradiction = get_consistency(sent1, sent2)
                    if contradiction > 0.9:

                        if len(sentences[k_j]) != 0:
                            sampled_sent = random.sample(sentences[k_j], 1)
                            
                            sentences[k_j].remove(sampled_sent[0])

                            for idx, _sent in sent_map.items():
                                retval = _check_contradiction(_sent, sampled_sent[0]['utter'])
                                if retval != False:
                                    contradict_matrix[idx][j] = retval
                                    sent_map[j] = sampled_sent[0]['utter']

                        else:
                            for idx in range(i+1):
                                contradict_matrix[idx][j] = -1.0
                           
                        return False
                    else:
                        contradict_matrix[i][j] = contradiction
                        
                        return contradiction
                    
                # 2-2. calculate contradiction scores
                flag = _check_contradiction(sent, sent_j)
                
                #if not flag:
                #    print(contradict_matrix)
                #    assert 1 == 0

    # 3. sampling sentences
    contradict_matrix = np.array(contradict_matrix)
    sum_matrix = np.sum(contradict_matrix, axis=0)
    indices = np.where(sum_matrix < 0.0)
    indices = indices[0].tolist()
    
    total_indices = [idx for idx in range(total_size)]
    for idx in indices:
        total_indices.remove(idx)
    
    select_pool = {'demographic': [], 'psychographic': [], 'wellness': []}
    for idx in total_indices:
        if idx in group_categorymap['demographic']:
            select_pool['demographic'].append(idx)
        elif idx in group_categorymap['psychographic']:
            select_pool['psychographic'].append(idx)
        else:
            select_pool['wellness'].append(idx)
    
    ret_set = []
    for _ in range(200):
        selected_demo_indices = random.sample(select_pool['demographic'], 2)
        selected_psy_indices = random.sample(select_pool['psychographic'], 2)
        selected_wellness_indices = random.sample(select_pool['wellness'], 1)

        selected_demo_sents = [sent_map[idx] for idx in selected_demo_indices]
        selected_psy_sents = [sent_map[idx] for idx in selected_psy_indices]
        selected_wellness_sents = [sent_map[idx] for idx in selected_wellness_indices]

        selected_set = selected_demo_sents + selected_psy_sents + selected_wellness_sents
        ret_set.append(selected_set)

    assert len(ret_set) == 200
    return ret_set


if __name__ == '__main__':
    args = parse_args()
    
    profile_save_dir = args.profile_save_dir

    # profile data loaded
    total_profile_results = defaultdict(list)
    for ele in os.listdir(profile_save_dir):
        if '.pkl' not in ele:
            continue
        
        filename = os.path.join(profile_save_dir, ele)
        with open(filename, 'rb') as f:
            profile_data = pc.load(f)

        key = ele.split('_results.pkl')[0]
        total_profile_results[key] = profile_data
        
    # sampling persona set
    temp_sampled_sents = defaultdict(str)
    
    categorymap = {}
    for t, (k, v) in enumerate(total_profile_results.items()):
        categorymap[t] = k
        sampled_sent = random.sample(v, 1)
        temp_sampled_sents[k] = sampled_sent[0]['utter']

    group_categorymap = {'demographic': [], 'psychographic': [], 'wellness': []}
    for k, v in categorymap.items():
        if v in TARGET_DEMOGRAPHIC_ATTRMAP.keys():
            group_categorymap['demographic'].append(v)
        elif v in TARGET_PSYCHOGRAPHICS_ATTRMAP.keys():
            group_categorymap['psychographic'].append(v)
        else:
            group_categorymap['wellness'].append(v)

    persona_set_save_dir = args.persona_set_dir
    os.makedirs(persona_set_save_dir, exist_ok=True)
    target_nums = 5
    total_persona_set = []
    for i in tqdm(range(target_nums)):
        sampled_set = sampling_persona_set(deepcopy(total_profile_results))
        
        with open(os.path.join(persona_set_save_dir, f'{i+1}_sampled_set.pkl'), 'wb') as f:
            pc.dump(sampled_set, f)