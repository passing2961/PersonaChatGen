from typing import List
import json
import random
import openai
from collections import defaultdict
from tqdm import tqdm

from constant import *


class PromptGenerator(object):

    def __init__(self, personachat_data_dir: str, datatype: str, num_trial: int):

        self.num_trial = num_trial

        self.personachat_data = self.load_data(personachat_data_dir)

        self.main_categories = ['demographic', 'psychographic']
        self.all_attr = FEWSHOT_ALL_ATTR_ENT_PAIRS

        self.fewshot_category_key = dict(FEWSHOT_DEMOGRAPHIC_ATTR, **FEWSHOT_PSYCHOGRAPHICS_ATTR)
        self.fewshot_category_template = dict(FEWSHOT_DEMOGRAPHIC_ATTRMAP, **FEWSHOT_PSYCHOGRAPHICS_ATTRMAP)
        
        self.PROMPT = "### User's persona: {USER_PERSONA}\n\nGenerate five profile sentences related to the given user's persona and the \"{PERSONA_VALUE}\" in each sentence:\n"
        
    def load_data(self, data_dir):
        with open(data_dir, 'r') as f:
            return [json.loads(line.strip()) for line in f.readlines()]

    def get_response(self, prompt_input: str, temp: float, max_tokens: int, top_p: float, freq_penalty: float, pres_penalty: float, stop_seq: List):
        
        openai.api_key = "<API_KEY>"
        openai.organization = "<ORG_ID>"

        response = openai.Completion.create(
            engine="text-davinci-002",
            prompt=prompt_input,
            temperature=temp,
            max_tokens=max_tokens,
            top_p=top_p,
            frequency_penalty=freq_penalty,
            presence_penalty=pres_penalty,
            stop=stop_seq,
            #n=1,
            #logprobs=5,
        )
        
        return response['choices'][0]['text']


class ProfilePromptGenerator(PromptGenerator):

    def __init__(self, datatype: str, num_trial: int):
        super().__init__(datatype=datatype, num_trial=num_trial)

        self.profile_sent_by_category = self.group_by_persona_category()

    def group_by_persona_category(self):
        profile_sent_by_category = defaultdict(list)

        for main_category in self.main_categories:
            for (category, entity_key) in self.all_attr[main_category]:
                entity_values = ENT_KEY2VAL[entity_key]
                for i, example in enumerate(self.personachat_data):
                    
                    try:
                        persona_category = example['persona_attribute']
                        
                        if persona_category != category:
                            continue
                        
                        if example['persona_value'] not in entity_values:
                            continue

                        key = persona_category + '|' + entity_key
                        profile_sent_by_category[key].append({
                            'profile_sent': example['original_persona_sentence'],
                            'persona_category': example['persona_attribute'],
                            'persona_entity_value': example['persona_value']
                        })

                    except KeyError:
                        continue

        return profile_sent_by_category

    def construct_prompt(self, persona_category, entity_value):
        return self.PROMPT.format(USER_PERSONA=persona_category, PERSONA_VALUE=entity_value)

    def remove_duplicate_sents(self, examples):
        ret_val = {}
        ret_sent = []
        for example in examples:
            profile_sent = example['profile_sent']
            entity_val = example['persona_entity_value']

            if profile_sent in ret_val.keys():
                continue
            
            ret_val[profile_sent] = entity_val
            ret_sent.append(profile_sent)

        return ret_val, ret_sent

    def generate(
        self,
        target_persona,
        temp=0.7,
        max_tokens=128,
        top_p=1.0,
        freq_penalty=0.4,
        pres_penalty=0.4,
    ):
        results = defaultdict(list)

        target_persona_template = TARGET_ALL_ATTRMAP[target_persona]
        target_prompt = self.construct_prompt(*target_persona_template)

        for k, v in tqdm(self.profile_sent_by_category.items()):
            try:
                fwt_category_key = self.fewshot_category_key[k]
            except KeyError:
                print('KeyError: ', k)
                continue

            duplicated_ents, duplicated_sents = self.remove_duplicate_sents(v)
            if len(duplicated_sents) < 5:
                continue
            
            for _ in range(self.num_trial):
                selected_incontext_examples = random.sample(duplicated_sents, 5)

                try:
                    category_template = self.fewshot_category_template[fwt_category_key]
                except KeyError:
                    print('KeyError: ', k)
                    continue
                
                fewshot_prompt = self.construct_prompt(*category_template)

                for i, profile_sent in enumerate(selected_incontext_examples):
                    #profile_sent = example['profile_sent']
                    #entity_value = example['persona_entity_value']
                    entity_value = duplicated_ents[profile_sent]
                
                    fewshot_prompt += f'{i+1}. {profile_sent} ({category_template[1]}: {entity_value})\n'

                input_prompt = fewshot_prompt + '\n' + target_prompt + '1.'
                
                resp = self.get_response(
                    input_prompt,
                    temp,
                    max_tokens,
                    top_p,
                    freq_penalty,
                    pres_penalty,
                    stop_seq=['###']
                )
                
                results[fwt_category_key + '++' + target_persona].append({
                    'input_prompt': input_prompt,
                    'response': resp,
                })

        return results

class ChatPromptGenerator(PromptGenerator):

    def __init__(self, persona_set=None, persona_chat=None):

        self.persona_set = persona_set
        self.persona_chat = persona_chat
        self.persona_chat_idx = list(persona_chat.keys())

        with open('personachat_prompt.txt', 'r') as f:
            self.CHAT_PROMPT = f.read()

    def random_sampling(self):
        sampled_persona_chat_idx = random.sample(self.persona_chat_idx, 1)
        sampled_persona_chat = self.persona_chat[sampled_persona_chat_idx[0]]

        sampled_persona_set = random.sample(self.persona_set, 2)
        return sampled_persona_chat, sampled_persona_set

    def execute_personachat(self, personachat):
        ret_val = {}
        dialog = []
        for example in personachat:
            if example['utter_idx'] == 0:
                spk_A = example['persona_sent']
            elif example['utter_idx'] == 1:
                spk_B = example['persona_sent']
            
            dialog.append(example['utterance'])
        
        ret_val = {
            'spk_A': spk_A,
            'spk_B': spk_B,
            'dialog': dialog
        }

        return ret_val

    def construct_dialog_for_prompt(self, conv):
        dialog_prompt_A = ''
        dialog_prompt_B = ''
        for i, utter in enumerate(conv):
            if i % 2 == 0:
                dialog_prompt_A += f'You: {utter}\n'
                dialog_prompt_B += f'Friend: {utter}\n'
            else:
                dialog_prompt_A += f'Friend: {utter}\n'
                dialog_prompt_B += f'You: {utter}\n'

        return dialog_prompt_A, dialog_prompt_B

    def generate(
        self,
        temp=0.8,
        max_tokens=128,
        top_p=1.0,
        freq_penalty=0.4,
        pres_penalty=0.4,
    ):

        sampled_persona_chat, sampled_persona_set = self.random_sampling()
        
        _persona_chat = self.execute_personachat(sampled_persona_chat)
        
        target_A_persona_set = sampled_persona_set[0]
        target_B_persona_set = sampled_persona_set[1]

        fewshot_A_persona_set = _persona_chat['spk_A']
        fewshot_B_persona_set = _persona_chat['spk_B']
        fewshot_conv = _persona_chat['dialog']

        fewshot_conv_A, fewshot_conv_B = self.construct_dialog_for_prompt(fewshot_conv)

        prompt_A = self.CHAT_PROMPT.format(
            FEWSHOT_PERSONA='\n'.join(fewshot_A_persona_set),
            FEWSHOT_CONV=fewshot_conv_A.strip(),
            TARGET_PERSONA=target_A_persona_set
        )
        
        prompt_B = self.CHAT_PROMPT.format(
            FEWSHOT_PERSONA='\n'.join(fewshot_B_persona_set),
            FEWSHOT_CONV=fewshot_conv_B.strip(),
            TARGET_PERSONA=target_B_persona_set
        )

        result = {
            'prompt_A': prompt_A, 'prompt_B': prompt_B,
            'prompt_A_text': target_A_persona_set, 
            'prompt_B_text': target_B_persona_set,
        }
        
        generations = self.generate_dataset(
            result, 
            temp,
            max_tokens,
            top_p,
            freq_penalty,
            pres_penalty,
            16
        )
        return generations
    
    def generate_dataset(
        self, 
        prompt, 
        temp,
        max_tokens,
        top_p,
        freq_penalty,
        pres_penalty,
        num_turns=12
    ):

        prompt_A = prompt['prompt_A']
        prompt_B = prompt['prompt_B']

        prompt_A_text = prompt['prompt_A_text']
        prompt_B_text = prompt['prompt_B_text']
        
        generated_dialog = []
        for i in range(num_turns):
            if i % 2 == 0:
                prompt_A += 'You:'
                prompt_B += 'Friend:'
                resp = self.get_response(
                    prompt_A, 
                    temp,
                    max_tokens,
                    top_p,
                    freq_penalty,
                    pres_penalty,
                    stop_seq=['Friend:', 'You:', '\n']
                )
                
            else:
                prompt_A += 'Friend:'
                prompt_B += 'You:'
                resp = self.get_response(
                    prompt_B, 
                    temp,
                    max_tokens,
                    top_p,
                    freq_penalty,
                    pres_penalty,
                    stop_seq=['Friend:', 'You:', '\n']
                )
            resp = resp[1:].strip()

            prompt_A += f' {resp}\n'
            prompt_B += f' {resp}\n'

            generated_dialog.append({
                'utter_idx': i, 'utter': resp,
                'prompt_A': prompt_A_text,
                'prompt_B': prompt_B_text,
                'input_prompt_A': prompt_A,
                'input_prompt_B': prompt_B
            })
        
        return generated_dialog

