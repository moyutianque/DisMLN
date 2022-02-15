import jsonlines
import h5py
import os
import os.path as osp
from tqdm import tqdm
import numpy as np
from PIL import Image
from utils.file_process_tools import find_all_ext
from utils.map_tools import colorize_nav_map, draw_path, draw_point, draw_agent
from utils.map_tools2 import gen_valid_map, colorize_valid_map
num_process =100
# Load maps
annt_root = '/data/leuven/335/vsc33595/ProjectsVSC/DisMLN/dset_gen/generated/raw'
map_path = 'data/maps/gmap_floor1_mpp_0.05_channel_last_with_bounds'
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

data = []
paths = find_all_ext(annt_root, 'jsonl')
for file_path in paths:
    with jsonlines.open(file_path) as reader:
        for obj in reader:
            maps = gmaps[obj['scene_name']]
            
            # nav_map = colorize_nav_map(maps['nav_map'])
            valid_map, valid_map_fixed=gen_valid_map(maps['nav_map'], maps['obj_maps'])
            nav_map = colorize_valid_map(valid_map_fixed)

            # draw current path
            path = [agent_stat[0] for agent_stat in obj['path']]
            rots = [agent_stat[1] for agent_stat in obj['path']]
            
            draw_path(nav_map, path, None, None, None, is_grid=True)
            draw_agent(obj['start'], obj['start_rot'], nav_map) 
            
            for i, (point, rot) in enumerate(zip(path, rots)):
                if i==0:
                    continue
                draw_agent(point, rot, nav_map, is_radian=True, color=(255, 0, 0, 255))
            
            # draw gt_path
            path = [agent_stat[0] for agent_stat in obj['gt_path']]
            rots = [agent_stat[1] for agent_stat in obj['gt_path']]
            draw_path(nav_map, path, None, None, None, is_grid=True, color=(255,128,128,255))
            
            for i, (point, rot) in enumerate(zip(path, rots)):
                if i==0:
                    continue
                draw_agent(point, rot, nav_map, is_radian=True, color=(255, 128, 0, 255))
            print("DTW distance:", obj['dtw'])
            print("NDTW distance:", obj['ndtw'])
            pil_img = Image.fromarray(np.copy(nav_map))
            pil_img.save('./tmp/sample.png')

            a = input("Enter to continue")