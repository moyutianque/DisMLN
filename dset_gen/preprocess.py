from utils.file_process_tools import find_all_ext
import jsonlines
from utils.instance_process import get_surrounding_objs, get_room_compass, get_surrounding_objs_old, get_room_compass_old
from utils.layeridx2wordidx import room_layeridx2wordidx, obj_layeridx2wordidx
import numpy as np
import random
import h5py
import os
import os.path as osp
from tqdm import tqdm
from multiprocessing import Pool
import signal
import pickle as pkl

from utils.map_tools import execution_time
from utils.instance_process import get_key_points
from collections import defaultdict
from utils.math_tools import scaler
import math

class TimeoutException(Exception):   # Custom exception class
    pass

def timeout_handler(signum, frame):   # Custom signal handler
    raise TimeoutException
# Change the behavior of SIGALRM
signal.signal(signal.SIGALRM, timeout_handler)

import argparse
parser = argparse.ArgumentParser()
parser.add_argument(
    "--split",
    choices=["train", "val_seen", "val_unseen"],
    required=True
)
args = parser.parse_args()

num_process =100
# Load maps
annt_root = f'/data/leuven/335/vsc33595/ProjectsVSC/DisMLN/dset_gen/generated/raw_{args.split}'
out_dir = f'./dset_gen/generated/merged/{args.split}_tmp'

print(f"running preprocessing..., output to {out_dir}")


map_path = 'data/maps/gmap_floor1_mpp_0.05_channel_last_with_bounds'
max_num_points = 50
gmaps = dict()
files = os.listdir(map_path)
scene_ids = [file.split('_')[0] for file in files]
map_key_points = dict()

flag = False
for scene_id in tqdm(scene_ids):
    file_name = f"./tmp/points_instances/{scene_id}.npy"
    if not os.path.exists(file_name):
        flag = True
        continue
    objs, rooms = np.load(file_name, allow_pickle=True)
    map_key_points[scene_id] = {"objs": objs, "rooms": rooms}
    print(f"scene {scene_id} has objs instance {objs.shape}, room instance {rooms.shape}")
print(f"using old generation code: {flag}")

if flag:
    for scene_id in tqdm(scene_ids): 
        gmap_path = osp.join(map_path, f"{scene_id}_gmap.h5")
        with h5py.File(gmap_path, "r") as f:
            nav_map  = f['nav_map'][()]
            room_map = f['room_map'][()] 
            obj_maps = f['obj_maps'][()] 
            obj_maps[:,:,1] = ((obj_maps[:,:,1]>0) ^ (obj_maps[:,:,15]>0)) * obj_maps[:,:,15] + obj_maps[:,:,1] # merge stairs to floor
            bounds = f['bounds'][()]
        grid_dimensions = (nav_map.shape[0], nav_map.shape[1])
        gmaps[scene_id] = {
            "nav_map": nav_map,
            "room_map": room_map,
            "obj_maps": obj_maps,
            "bounds": bounds,
            "grid_dimensions": grid_dimensions
        }

    map_key_points = dict()
    for k,v in gmaps.items():
        file_name = f"./tmp/points_instances/{k}.npy"
        objs, rooms = get_key_points(v['obj_maps'], v['room_map'])
        np.save(f"./tmp/points_instances/{k}.npy", [objs, rooms])
        print(f"scene {k} has objs instance {objs.shape}, room instance {rooms.shape}")
        map_key_points[scene_id] = {"objs": objs, "rooms": rooms}

# Load path annotations
data = []
paths = find_all_ext(annt_root, 'jsonl')
for file_path in paths:
    with jsonlines.open(file_path) as reader:
        for obj in reader:
            data.append(obj)


# room campass chunks
room_compass_chunks = 12

def process_one_ep(datum, ep_idx):
    instruction = datum['instruction']['instruction_text']
    d_id = datum['id']
    target = datum['score']
    start = datum['start']
    goal = datum['goal']
    ndtw = datum['ndtw']
    
    path = datum['path']
    points_list = []
    room_list = []
    try:
        trajectory = [] 
        for agent_stat in path:
            # normalize traj path
            scaled_r = scaler((0,datum['scene_shape'][0]), (-2, 2), agent_stat[0][0])
            scaled_c = scaler((0,datum['scene_shape'][1]), (-2, 2), agent_stat[0][1])
            direction_r = 2 * math.cos(agent_stat[1]) # [-2,2]
            direction_c = 2 * math.sin(agent_stat[1]) # [-2,2]
            trajectory.append((scaled_r, scaled_c, 0, direction_r, direction_c, 0))

            room_satus = get_room_compass(
                map_key_points[datum['scene_name']]['rooms'], (*agent_stat[0], agent_stat[1]),
                radius=10, num_chunks=room_compass_chunks, is_radian=True
            )
            _, relative_dict = get_surrounding_objs(
                map_key_points[datum['scene_name']]['objs'], (*agent_stat[0], agent_stat[1]), 
                5, is_radian=True
            )
            
            points = [] # object points
            for k, v in relative_dict.items():
                for point in v:
                    points.append((point[0], point[1], 0, obj_layeridx2wordidx[k])) # point: [rel_r, rel_c, rel_ang, ins_id]
            
            if len(points) == 1: # corner case, no object arround
                print(f"No object for ep {d_id} scene {datum['scene_name']} at {agent_stat}")

            if len(points) > 50:
                points = random.sample(points, 50)
            else:
                choices = np.random.choice(len(points), 50-len(points))
                points.extend(np.array(points)[choices].tolist())
            
            # room compass tensor
            rooms = []
            for i in range(room_compass_chunks):
                m = room_satus.get(i, None)
                if m is not None:
                    rooms.append(room_layeridx2wordidx[m[0]])
                else:
                    rooms.append(0)

            room_list.append(rooms)
            points_list.append(points)

        with jsonlines.open(osp.join(out_dir, f'{ep_idx}.jsonl'), 'w') as writer:
            writer.write({
                'scene_name': datum['scene_name'],
                "ep_id": d_id,
                'instruction': instruction,
                'normalized_traj': trajectory,
                'room_list': room_list,
                'points_list': points_list,
                'target': target[0],
                "ndtw": ndtw,
                "dtw": datum['dtw'],
                # 'raw': {
                #     "start": start,
                #     "start_rot": datum['start_rot'],
                #     "path": datum['path'],
                #     "gt_path": datum['gt_path'],
                # }
            })
    except Exception as e:
        print(f"Failed for ep-{d_id}, msg: {e}")

@execution_time   
def gen():
    pool = Pool(num_process)
    print(f"Creating {len(data)} data, remember to check.")
    for i, episode in enumerate(data):
        if flag:
            raise ValueError("Please generate first")
        else:
            pool.apply_async(process_one_ep, args=(episode, i, ))
            # process_one_ep(episode, i)
        # process_one_ep(episode, i)

    pool.close()
    pool.join()
    print("success")

if __name__ == "__main__":
    gen()