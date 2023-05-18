import os
import json
import argparse
import pickle as pc
from tqdm import tqdm

from prompt_generator import ChatPromptGenerator


def organize_personachat(conv_data):
    conv_last_idx = conv_data[-1]['conv_idx']

    conversations = {}
    for idx in tqdm(range(conv_last_idx+1)):
        tmp_conv = []
        for example in conv_data:
            if example['conv_idx'] == idx:
                tmp_conv.append(example)
        
        conversations[idx] = tmp_conv
    return conversations

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--persona_set_dir', type=str, default=None)
    parser.add_argument('--personachat_dir', type=str, default=None)
    parser.add_argument('--generation_save_dir', type=str, default=None)

    return parser.parse_args()

if __name__ == '__main__':
    
    args = parse_args()

    total_persona_set = []
    personaset_save_dir = args.persona_set_dir #'/home/yjlee/workspace/KT/emnlp2022/code/dataset_generator_code/result/persona_set/new_wellness_v2'
    for ele in os.listdir(personaset_save_dir):
        if '.pkl' not in ele:
            continue
        
        filedir = os.path.join(personaset_save_dir, ele)
        with open(filedir, 'rb') as f:
            personaset = pc.load(f)
        
        total_persona_set += ['\n'.join(x) for x in personaset]
    
    # remove duplication of persona set
    print(f'before # of persona set: {len(total_persona_set)}')
    total_persona_set = list(set(total_persona_set))
    print(f'after # of persona set: {len(total_persona_set)}')

    # load persona chat dataset
    # sampling persona chat dataset
    personachat_data_dir = args.personachat_dir #f'/home/yjlee/workspace/KT/data/annotated_data/personachat:both_original/train.jsonl'
    with open(personachat_data_dir, 'r') as f:
        personachat_data = [json.loads(line.strip()) for line in f.readlines()]
    
    personachat = organize_personachat(personachat_data)
    origin_data_fullsize = len(personachat.keys())

    generations_save_dir = args.generation_save_dir #f'./result/chat/new_wellness'
    os.makedirs(generations_save_dir, exist_ok=True)

    chat_generator = ChatPromptGenerator(total_persona_set, personachat)
    num_trials = 1000 # how many time do you want to generate the dialogue using GPT-3

    total_generations = []
    for i in tqdm(range(num_trials)):
        generations = chat_generator.generate()
        total_generations.append(generations)
        
        with open(os.path.join(generations_save_dir, f'{i}_results.pkl'), 'wb') as f:
            pc.dump(generations, f)
    
