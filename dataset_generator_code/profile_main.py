import os
import pickle as pc
import argparse
from tqdm import tqdm

from prompt_generator import ProfilePromptGenerator
from constant import TARGET_ALL_ATTRMAP


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--max-tokens', default=128, type=int)
    parser.add_argument('--freq-penalty', default=0.4, type=float)
    parser.add_argument('--pres-penalty', default=0.4, type=float)
    parser.add_argument('--top-p', default=1, type=float)
    parser.add_argument('--temp', default=0.7, type=float)
    parser.add_argument('--num-trial', default=5, type=int)
    parser.add_argument('--datatype', default='train', type=str)
    parser.add_argument('--save-dir', type=str, default='./result/profile')

    return parser.parse_args()

if __name__ == '__main__':

    args = parse_args()
    
    profile_generator = ProfilePromptGenerator(
        datatype=args.datatype,
        num_trial=args.num_trial
    )
    
    save_dir = args.save_dir
    os.makedirs(save_dir, exist_ok=True)

    for i, target_persona in tqdm(enumerate(TARGET_ALL_ATTRMAP)):
        results = profile_generator.generate(target_persona)
  
        with open(os.path.join(save_dir, f'{target_persona}++generations.pkl'), 'wb') as f:
            pc.dump(results, f)
    

    
    