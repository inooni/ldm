# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

from typing import List, Optional
import argparse
import fire
import json
import re
from tqdm import tqdm
from llama import Llama, Dialog


def get_prompt():
    with open('prompt/example.txt') as f:
        prompt = f.read()
    return prompt

def get_datasets():
    anno_dir = '/hub_data2/miso/ldm/datasets/visdial/dialogs/train_imgid_dialogs.json'
    with open(anno_dir) as f:
        annotations = json.load(f)
    return annotations

def save_datasets(results):
    save_dir = 'results/example_imgid.json'
    with open(save_dir, 'w') as f:
        json.dump(results, f, indent=4)

def extract_generated_sentence(result):
    # match = re.search(r'\n', result)
    # if match:
    #     return match.group(1)
    # else:
    #     return result
    try:
        return result.split('\n')[-1].replace('\"', '')
    except:
        return result

def main(
    ckpt_dir: str,
    tokenizer_path: str,
    # save_dir: str = None,
    temperature: float = 0.6,
    top_p: float = 0.9,
    max_seq_len: int = 512,
    max_batch_size: int = 8,
    max_gen_len: Optional[int] = None,
):
    
    generator = Llama.build(
        ckpt_dir=ckpt_dir,
        tokenizer_path=tokenizer_path,
        max_seq_len=max_seq_len,
        max_batch_size=max_batch_size,
    )
    prompt = get_prompt()
    annotations = get_datasets()
    
    dialogs = list()
    image_ids = list()
    generations = dict()
    data_cnt = 0

    for image_id, contents in tqdm(annotations.items()):
        qa_pairs = ''
        for dialog in contents['dialogs']:
            qa_pairs += 'A: ' + dialog['question'] + ' B: ' + dialog['answer'] + '\n'
        input_user = "Original Dialogs: " + qa_pairs
        input_dialog = [
            {"role": "system", "content": prompt},
            {"role": "user", "content": input_user}
        ]
        dialogs.append(input_dialog)
        image_ids.append(image_id)
        data_cnt += 1

        if len(dialogs) == max_batch_size or data_cnt == len(annotations):
            try:
                results = generator.chat_completion(
                            dialogs,  # type: ignore
                            max_gen_len=max_gen_len,
                            temperature=temperature,
                            top_p=top_p,
                        )
                
                for img_id, dia, res in zip(image_ids, dialogs, results):
                    res = extract_generated_sentence(res['generation']['content'])
                    dia = dia[1]['content'].split('Original Dialogs: ')[-1]
                    generations[img_id] = {"dialog": dia, "instruction": res}
                    # generations.append({"dialog": dia, "instruction": res})
                save_datasets(generations)
                dialogs = list()
                image_ids = list()
            except:
                dialogs = list()
                image_ids = list()
                continue
        else:
            continue
    save_datasets(generations)
if __name__ == "__main__":
    fire.Fire(main)