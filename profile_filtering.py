import os
import re
import argparse
import pickle as pc
from collections import defaultdict
from tqdm import tqdm

import stanza
from transformers import pipeline

from constant import *

os.environ['CUDA_VISIBLE_DEVICES'] = '1'


class ProfilePipeline(object):
    """
    Filtering pipeline for profile sentences
    """
    def __init__(self, models, th, candidate_label):
        self.org_size = 0
        self.survival_data = defaultdict(list)
        self.th = th
        self.candidate_label = candidate_label

        self.lemma = models['lemma']
        self.clf = models['clf']

    def calculate_survival_rate(self, after_size):
        survival_rate = after_size * 100 / self.org_size
        return round(survival_rate, 1)

    def regex_based_filtering(self, sentences):
        """
        If the profile sentence don't match with the regex pattern,
        then we regard the sentence as an inappropriate generated sentence.
        So, we filter it out.
        """
        result = []
        
        for sentence in sentences:
            sentence = sentence[1:] # remove a whitespace prefix

            delims = [f'\n{i+1}. ' for i in range(1, 5)]
            splitted_sent = re.split('|'.join(delims), sentence)
            
            pattern = '(?P<utter>.*) [\(|\[](?P<attr>.*): (?P<value>.*)[\)|\]]' # [] case should be possible
            compiled_regex = re.compile(pattern)

            self.org_size += len(splitted_sent)
            for example in splitted_sent:
                matched = compiled_regex.match(example)
                
                if matched:
                    result.append(matched.groupdict())

        self.survival_data['origin_data_size'] = [self.org_size, 100]
        after_size = len(result)
        survival_rate = self.calculate_survival_rate(after_size)    
        self.survival_data['filtering:regex'] = [after_size, survival_rate]
        #self.f.write(f'[Regex] Cumulative Survival Rate (%): {survival_rate}\n')
        return result

    def explicit_filtering(self, sentences):
        results = []

        for sentence in sentences:
            utter = sentence['utter']
            attr = sentence['attr']
            value = sentence['value']

            if value == '':
                continue
            
            if value in utter: #and attr == self.candidate_label:
                results.append(sentence)
            

        after_size = len(results)
        survival_rate = self.calculate_survival_rate(after_size)
        self.survival_data['filtering:explicit'] = [after_size, survival_rate]
        #self.f.write(f'[Explicit] Cumulative Survival Rate (%): {survival_rate}\n')
        return results

    def persona_category_filtering(self, sentences):
        results = []
        
        for sentence in sentences:
            utter = sentence['utter']
            value = sentence['value']

            output = self.clf(utter, self.candidate_label)
            if output['scores'][0] < self.th:
                continue
            
            sentence['scores'] = output['scores'][0]
            results.append(sentence)
        
        after_size = len(results)
        survival_rate = self.calculate_survival_rate(after_size)
        self.survival_data['filtering:persona_category'] = [after_size, survival_rate]
        #self.f.write(f'[Preserving] Cumulative Survival Rate (%): {survival_rate}\n')
        return results
    
    
    def duplication_filtering(self, sentences):
        results = []
        dup_results = {}
        for sentence in sentences:
            utter, attr, value, scores = sentence.values()

            if utter in dup_results.keys():
                if dup_results[utter][1] != value:
                    print(f'Warning: entity value mismatch occurs! Previous value = {dup_results[utter][1]}, Current value = {value} of utterance = {utter}.')
                    continue

            dup_results[utter] = [attr, value, scores]
        
        for k, v in dup_results.items():
            results.append({'utter': k, 'attr': v[0], 'value': v[1], 'scores': v[2]})

        after_size = len(results)
        survival_rate = self.calculate_survival_rate(after_size)
        self.survival_data['filtering:duplication'] = [after_size, survival_rate]
        return results

    def do_filtering(self, sentences):

        regex_filter_results = self.regex_based_filtering(sentences)
        explicit_filter_results = self.explicit_filtering(regex_filter_results)
        persona_category_filter_results = self.persona_category_filtering(explicit_filter_results)
        
        duplication_filter_results = self.duplication_filtering(persona_category_filter_results)

        return duplication_filter_results
    
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
        
        
        

