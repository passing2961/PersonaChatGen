import os
import argparse
from tqdm import tqdm
from collections import defaultdict
import itertools

import torch
from parlai.core.metrics import F1Metric
from detoxify import Detoxify
from transformers import AutoTokenizer, AutoModelForSequenceClassification


class ChatPipeline(object):
    """
    Filtering pipeline for persona chat
    """
    def __init__(self, conversations, pair_type, consistent_th, toxic_clf, consistent_tokenizer, consistent_clf, device='cuda'):
        self.org_size = 0
        self.survival_data = defaultdict(list)
        self.consistent_th = consistent_th

        self.consistent_tokenizer = consistent_tokenizer
        self.consistent_clf = consistent_clf
        self.toxic_clf = toxic_clf

        self.consistent_clf.to(device)
        self.consistent_clf.eval()

        self.device = device

        self.conversations = conversations
        self.org_size = len(conversations)
        self.pair_type = pair_type
        self.pair_conversations = []
        #self.make_pair_conversations()

        self.API_KEY = 'AIzaSyC5tstKNGQIKF8XnmtFiAdZIjVg7xufiSo'

    def make_pair_conversations(self, conversations):
        
        for conv_idx, conversation in enumerate(conversations):
            my_persona, partner_persona, my_conv, partner_conv = conversation

            if self.pair_type == 'pu':
                my_pu = list(itertools.product(my_persona, my_conv))
                partner_pu = list(itertools.product(partner_persona, partner_conv))
                self.pair_conversations.append((my_pu, partner_pu, conversation))
        
    def calculate_survival_rate(self, after_size):
        survival_rate = after_size * 100 / self.org_size
        return round(survival_rate, 1)

    def get_consistency(self, premise, hyp):
        tokenized_input_seq_pair = self.consistent_tokenizer.encode_plus(premise, hyp, max_length=128, return_token_type_ids=True, truncation=True)
        input_ids = torch.Tensor(tokenized_input_seq_pair['input_ids']).long().unsqueeze(0).to(self.device)
        # remember bart doesn't have 'token_type_ids', remove the line below if you are using bart.
        token_type_ids = torch.Tensor(tokenized_input_seq_pair['token_type_ids']).long().unsqueeze(0).to(self.device)
        attention_mask = torch.Tensor(tokenized_input_seq_pair['attention_mask']).long().unsqueeze(0).to(self.device)

        with torch.no_grad():
            outputs = self.consistent_clf(input_ids,
                                          attention_mask=attention_mask,
                                          token_type_ids=token_type_ids,
                                          labels=None)
            
            predicted_prob = torch.softmax(outputs[0], dim=1)[0].tolist() # batch_size only one

        return {'non_contradiction': predicted_prob[0], 'contradiction': predicted_prob[1]}
    
    def consistency_filtering(self):
        results = []
        for conv_idx, conversation in enumerate(tqdm(self.pair_conversations)):
            
            my_cnt, partner_cnt = 0, 0
            my_pu, partner_pu, conv = conversation
            for ele in my_pu:
                consistent_result = self.get_consistency(ele[0], ele[1])
                if consistent_result['contradiction'] > self.consistent_th:
                    my_cnt += 1
            
            for ele in partner_pu:
                consistent_result = self.get_consistency(ele[0], ele[1])
                if consistent_result['contradiction'] > self.consistent_th:
                    partner_cnt += 1
            
            total_cnt = my_cnt + partner_cnt
            if total_cnt == 0:
                results.append(conv)
        
        #self.survival_data['origin_data_size'] = [self.org_size, 100]
        after_size = len(results)
        survival_rate = self.calculate_survival_rate(after_size)    
        self.survival_data['consistency'] = [after_size, survival_rate]
        return results
        
    def toxic_filtering(self, conversations):
        results = []
        for conv_idx, conversation in enumerate(tqdm(conversations)):
            
            _, _, my_conv, partner_conv = conversation

            my_cnt, partner_cnt = 0, 0
            for utter in my_conv:
                output = self.toxic_clf.predict(utter)
                toxic_score = output['toxicity']
                if toxic_score > 0.7:
                    my_cnt += 1
            
            for utter in partner_conv:
                output = self.toxic_clf.predict(utter)
                toxic_score = output['toxicity']
                if toxic_score > 0.7:
                    partner_cnt += 1
            
            total_cnt = my_cnt + partner_cnt
            if total_cnt == 0:
                results.append(conversation)
        
        after_size = len(results)
        survival_rate = self.calculate_survival_rate(after_size)    
        self.survival_data['toxicity'] = [after_size, survival_rate]
        return results

    def _get_copy_ratio(self, personas, conv):
        copy_cnt = 0
        total_cnt = 0
        for persona in personas:
            tmp_f1_scores = []
            for utter in conv:
                f1_score = F1Metric.compute(guess=utter, answers=[persona])
                tmp_f1_scores.append(f1_score)
            
            if max(tmp_f1_scores) >= 0.8:
                copy_cnt += 1
            total_cnt += 1
            
        assert copy_cnt <= 5
        return copy_cnt

    def copy_paste_filtering(self):
        results = []
        for conv_idx, conversation in enumerate(tqdm(self.conversations)):
            
            my_persona, partner_persona, my_conv, partner_conv = conversation

            my_copy_ratio = self._get_copy_ratio(my_persona, my_conv)
            partner_copy_ratio = self._get_copy_ratio(partner_persona, partner_conv)

            if my_copy_ratio > 1 or partner_copy_ratio > 1:
                continue

            results.append(conversation)

        self.survival_data['origin_data_size'] = [self.org_size, 100]
        after_size = len(results)
        survival_rate = self.calculate_survival_rate(after_size)    
        self.survival_data['copy-paste'] = [after_size, survival_rate]
        return results
            
    def do_filtering(self):
        copy_results = self.copy_paste_filtering()

        for k, v in self.survival_data.items():
            print(k, v)
        self.make_pair_conversations(copy_results)
        consistent_results = self.consistency_filtering()
        
        toxic_results = self.toxic_filtering(consistent_results)
        
        for k, v in self.survival_data.items():
            print(k, v)
        
        return toxic_results

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--personachatgen_dir', type=str, default=None)
    parser.add_argument('--file_save_dir', type=str, default=None)

    return parser.parse_args()

if __name__ == '__main__':

    args = parse_args()

    # model define
    consistent_clf_path = 'ynie/roberta-large_conv_contradiction_detector_v0'
    consistent_tokenizer = AutoTokenizer.from_pretrained(consistent_clf_path)
    consistent_clf = AutoModelForSequenceClassification.from_pretrained(consistent_clf_path)

    toxic_clf = Detoxify('original')
    
    root_dir = args.personachatgen_dir # '/home/yjlee/workspace/KT/emnlp2022/code/dataset_generator_code/result/parlai_chat/all_final'
    file_save_dir = args.file_save_dir
    os.makedirs(file_save_dir, exist_ok=True)

    for persona_type in ['self', 'both']:
        for datatype in ['train', 'valid']:
            chat_dir = f'{root_dir}/{datatype}_{persona_type}_original_no_cands.txt'

            conversations = []
            my_persona, partner_persona = [], []
            my_conv, partner_conv = [], []
            with open(chat_dir, 'r') as f:
                for line in f.readlines():
                    line = line.strip()
                    split_idx = line.find(' ')

                    conv_idx = int(line[:split_idx])
                    if conv_idx == 1:
                        conversations.append((my_persona, partner_persona, my_conv, partner_conv))
                        my_persona, partner_persona = [], []
                        my_conv, partner_conv = [], []

                    utter = line[split_idx+1:]

                    if utter.startswith("your persona:"):
                        my_persona.append(utter.split('your persona: ')[-1])
                    elif utter.startswith("partner's persona:"):
                        partner_persona.append(utter.split("partner's persona: ")[-1])
                    else:
                        partner_utter, my_utter = utter.split('\t')
                        my_conv.append(my_utter)
                        partner_conv.append(partner_utter)
            
            if len(my_conv) > 1:
                conversations.append((my_persona, partner_persona, my_conv, partner_conv))
            
            conversations = conversations[1:]
            
            pipeline = ChatPipeline(conversations, 'pu', 0.9, toxic_clf, consistent_tokenizer, consistent_clf, device='cuda')
            final_conversations = pipeline.do_filtering()
            
            print(pipeline.survival_data)

            f = open(os.path.join(file_save_dir, f'{datatype}_{persona_type}_original_no_cands.txt'), 'w')

            for conv in final_conversations:
                my_persona, partner_persona, my_conv, partner_conv = conv
                line_idx = 1
                for ele in my_persona:
                    f.write(f'{line_idx} your persona: {ele}\n')
                    line_idx += 1
                
                if persona_type == 'both':
                    for ele in partner_persona:
                        f.write(f"{line_idx} partner's persona: {ele}\n")
                        line_idx += 1
                
                for p_utter, m_utter in zip(partner_conv, my_conv):
                    f.write(f'{line_idx} {p_utter}\t{m_utter}\n')
                    line_idx += 1
            
            f.close()
        