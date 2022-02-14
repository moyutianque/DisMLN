from nbformat import write
from utils.file_process_tools import find_all_ext
import jsonlines
from collections import defaultdict
import random
root = 'dset_gen/generated/raw'
from tqdm import tqdm 

out_path = 'dset_gen/generated/cache_full_merge/merged.jsonl'
counter = defaultdict(int)

file_list = find_all_ext(root, 'jsonl')
for file in tqdm(file_list):
    with jsonlines.open(file) as reader:
        for obj in reader:
            if obj['score'][0] == 0 and random.random() < 0.98:
                continue
            counter[obj['score'][0]] +=1
            with jsonlines.open(out_path, 'a') as writer:
                writer.write(obj)

print(counter)
