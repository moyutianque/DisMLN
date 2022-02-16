from utils.file_process_tools import find_all_ext
import jsonlines
from utils.instance_process import get_surrounding_objs, get_room_compass
from utils.layeridx2wordidx import room_layeridx2wordidx, obj_layeridx2wordidx
import numpy as np
import random
import h5py
import os
import os.path as osp
from tqdm import tqdm
from multiprocessing import Pool
import signal

class TimeoutException(Exception):   # Custom exception class
    pass

def timeout_handler(signum, frame):   # Custom signal handler
    raise TimeoutException
# Change the behavior of SIGALRM
signal.signal(signal.SIGALRM, timeout_handler)

num_process =100
# Load maps
annt_root = '/data/leuven/335/vsc33595/ProjectsVSC/DisMLN/dset_gen/generated/cache_full_merge/down4x_SP/train_raw'
map_path = 'data/maps/gmap_floor1_mpp_0.05_channel_last_with_bounds'
out_dir = './dset_gen/generated/cache_full_merge/down4x_SP/test'

max_num_points = 50
gmaps = dict()
files = os.listdir(map_path)
scene_ids = [file.split('_')[0] for file in files]
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

# Load path annotations
data = []
paths = find_all_ext(annt_root, 'jsonl')
for file_path in paths:
    with jsonlines.open(file_path) as reader:
        for obj in reader:
            if obj['scene_name'] not in ['r1Q1Z4BcV1o', 'JeFG25nYj2']:
                continue
            data.append(obj)

# room campass chunks
room_compass_chunks = 12

def process_one_ep(datum, ep_idx):
    maps = gmaps[datum['scene_name']]

    instruction = datum['instruction']['instruction_text']
    # instruction = self.embedding_layer(torch.tensor(datum['instruction']['instruction_tokens']).long())
    d_id = datum['id']
    target = datum['score']
    start = datum['start']
    goal = datum['goal']
    ndtw = datum['ndtw']

    path = datum['path']

    signal.alarm(60*5)
    try:

        points_list = []
        room_list = []
        for agent_stat in path:
            room_satus = get_room_compass(
                maps['room_map'], maps['nav_map'], (*agent_stat[0], agent_stat[1]),
                radius=10, num_chunks=room_compass_chunks, is_radian=True
            )
            _, relative_dict = get_surrounding_objs(
                maps['obj_maps'], maps['nav_map'], (*agent_stat[0], agent_stat[1]), 
                3, floor_idx=1, is_radian=True, skip_not_valid=True
            )
            
            points = []
            for k, v in relative_dict.items():
                for point in v:
                    points.append((point[0], point[1], 0, obj_layeridx2wordidx[k])) # point: [rel_r, rel_c, rel_ang, ins_id]
            
            if len(points) == 0: # corner case, no object arround
                continue

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
                'room_list': room_list,
                'points_list': points_list,
                'target': target[0],
                "ndtw": ndtw,
                'raw': datum
            })
    except TimeoutException:
        print(f"Scene: {datum['scene_name']} cannot end in {60*5} seconds")
        exit()
    else:
        # Reset the alarm
        signal.alarm(0)
        
def gen():
    pool = Pool(num_process)
    for i, episode in enumerate(data):
        pool.apply_async(process_one_ep, args=(episode, i, ))
        # process_one_ep(episode, i)

    pool.close()
    pool.join()
    print("success")

if __name__ == "__main__":
    gen()