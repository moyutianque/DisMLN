from random import random
from utils.file_process_tools import find_all_ext
import h5py
import jsonlines
import argparse
from collections import defaultdict
from tqdm import tqdm 

parser = argparse.ArgumentParser()
parser.add_argument(
    "--split",
    choices=["train", "val_seen", "val_unseen"],
    required=True
)
args = parser.parse_args()

root = f'dset_gen/generated/merged/{args.split}_tmp'
out_path = f'dset_gen/generated/merged/{args.split}/{args.split}.h5'

counter = defaultdict(int)
cnt = 0
f = h5py.File(out_path, "w")
file_list = find_all_ext(root, 'jsonl')
for file in tqdm(file_list):
    with jsonlines.open(file) as reader:
        for b in reader:
            
            counter[b['target']] +=1

            if args.split == "train" and b['target'] == 0:
                if random() < 0.8:
                    continue

            grp = f.create_group(b['ep_id'])
            grp.attrs['instruction'] = b['instruction']
            grp.attrs['scene_name'] = b['scene_name']
            grp.attrs['target'] = b['target']
            grp.attrs['ndtw'] = b['ndtw']
            grp.attrs['dtw'] = b['dtw']
            grp.attrs['start'] = b['raw']['start']
            grp.attrs['start_rot'] = b['raw']['start_rot']


            grp.create_dataset("annt/normalized_traj", data=b['normalized_traj'])
            grp.create_dataset("annt/room_list", data=b['room_list'])
            grp.create_dataset("annt/points_list", data=b['points_list'])
            # dt = h5py.special_dtype(vlen=str)
            # grp.create_dataset("annt/instruction", data=b['instruction'], dtype=dt)
            # grp.create_dataset("annt/target", data=b['target'])
            # grp.create_dataset("annt/ndtw", data=b['ndtw'])
            # grp.create_dataset("annt/dtw", data=b['dtw'])
            # grp.create_dataset("annt/scene_name", data=b['scene_name'], dtype=dt)

            # grp.create_dataset("raw/start", data=b['raw']['start'])
            # grp.create_dataset("raw/start_rot", data=b['raw']['start_rot'])
            b['raw']['path'] = [[value[0][0], value[0][1], value[1]] for value in b['raw']['path']]
            grp.create_dataset("raw/path", data=b['raw']['path'])
            b['raw']['gt_path'] = [[value[0][0], value[0][1], value[1]] for value in b['raw']['gt_path']]
            grp.create_dataset("raw/gt_path", data=b['raw']['gt_path'])
            cnt += 1


print(counter)
print(f"{cnt} data processed")