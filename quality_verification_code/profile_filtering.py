import os
import argparse
import pickle as pc
from collections import defaultdict
from tqdm import tqdm

import stanza
from transformers import pipeline

from prompt_filter import ProfilePipeline
from constant import *

os.environ['CUDA_VISIBLE_DEVICES'] = '1'


def define_models():
    lemma = stanza.Pipeline(lang='en', processors='tokenize,mwt,pos,lemma')
    classifier = pipeline('zero-shot-classification')

    return {
        'lemma': lemma,
        'clf': classifier,
    }

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("--profile-save-dir", type=str, default='./result/profile')
    parser.add_argument("--filtered-profile-save-dir", type=str, default="./result/filtered_profile")

    args = parser.parse_args()

    models = define_models()

    profile_save_dir = args.profile_save_dir

    # aggregating generation results
    total_generations = defaultdict(list)
    for ele in os.listdir(profile_save_dir):
        if '++' not in ele:
            continue
        
        filename = os.path.join(profile_save_dir, ele)
        with open(filename, 'rb') as f:
            profile_results = pc.load(f)
        
        persona_attr = ele.split('++')[0]
        entity_key = TARGET_ALL_ATTRMAP[persona_attr][1]

        generations = []
        for k, v in profile_results.items():
            resp = [ele['response'] for ele in v][:5]
            generations.extend(resp)
        
        key = persona_attr
        total_generations[key] = generations
    
    th = 0.9
    
    total_survival_data = defaultdict(list)
    filter_save_dir = args.filtered_profile_save_dir
    os.makedirs(filter_save_dir, exist_ok=True)
    
    # profile filtering
    for attr, generations in tqdm(total_generations.items()):
        candidate_label = TARGET_ALL_ATTRMAP[attr][1]

        pipeline = ProfilePipeline(models, th, candidate_label)
        
        filter_results = pipeline.do_filtering(generations)

        with open(os.path.join(filter_save_dir, f'{attr}_results.pkl'), 'wb') as f:
            pc.dump(filter_results, f)

        total_survival_data[attr] = pipeline.survival_data

    ## total survival ratio 
    total_survival_size = defaultdict(int)
    for category, survival_data in total_survival_data.items():
        print(f'{category}: {survival_data}')

        for k, v in survival_data.items():
            total_survival_size[k] += v[0]

    total_org_size = total_survival_size['origin_data_size']
    for k, v in total_survival_size.items():
        survival_rate = v * 100 / total_org_size
        print(f'{k} | # of Exs: {v} | Survival Rate: {round(survival_rate, 2)}')
        
        
        

