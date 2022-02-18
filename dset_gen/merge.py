from nbformat import write
from utils.file_process_tools import find_all_ext
import jsonlines
from collections import defaultdict
import random
import pickle as pkl

import argparse
parser = argparse.ArgumentParser()
parser.add_argument(
    "--split",
    choices=["train", "val_seen", "val_unseen"],
    required=True
)
args = parser.parse_args()

root = f'dset_gen/generated/merged/{args.split}_tmp'
from tqdm import tqdm 

out_path = f'dset_gen/generated/merged/{args.split}/merged.pkl'
counter = defaultdict(int)

file_list = find_all_ext(root, 'jsonl')
cnt = 0

out_list = []
for file in tqdm(file_list):
    with jsonlines.open(file) as reader:
        for obj in reader:
            # if obj['target'] == 0 and random.random() < 0.98:
            #     continue
            # if len(obj['room_list']) > 10:
            #     print(f"SKIP {obj['ep_id']}")
            #     continue
            
            cnt += 1
            counter[obj['target']] +=1
            # with jsonlines.open(out_path, 'a') as writer:
            #     writer.write(obj)
            out_list.append(obj)
print(counter)
with open(out_path, 'wb') as handle:
    pkl.dump(out_list, handle, protocol=pkl.HIGHEST_PROTOCOL)

print(f"{cnt} data processed")

