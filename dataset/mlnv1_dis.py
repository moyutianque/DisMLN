from dis import Instruction
import json
import torch
from torch.utils.data import Dataset
import os
import os.path as osp
import h5py
import jsonlines
from utils.file_process_tools import find_all_ext
from tqdm import tqdm
import random
import numpy as np

from torch.utils.data._utils.collate import default_collate
from utils.instance_process import get_surrounding_objs, get_room_compass
from utils.parse_wordmap import load_embeddings
from utils.layeridx2wordidx import room_layeridx2wordidx, obj_layeridx2wordidx

class MLNv1_Dis_Dataset(Dataset):
    def __init__(self, config, split):
        # Load maps
        annt_root = config.annt_root.format(split=split)
        map_path = config.map_path
        self.max_num_points = config.max_num_points
        self.maps = dict()
        files = os.listdir(map_path)
        scene_ids = [file.split('_')[0] for file in files]
        print(f"Start Loading {len(scene_ids)} maps from {map_path}")
        for scene_id in tqdm(scene_ids):
            gmap_path = osp.join(map_path, f"{scene_id}_gmap.h5")
            with h5py.File(gmap_path, "r") as f:
                nav_map  = f['nav_map'][()]
                room_map = f['room_map'][()] 
                obj_maps = f['obj_maps'][()] 
                obj_maps[:,:,1] = ((obj_maps[:,:,1]>0) ^ (obj_maps[:,:,15]>0)) * obj_maps[:,:,15] + obj_maps[:,:,1] # merge stairs to floor
                bounds = f['bounds'][()]
            grid_dimensions = (nav_map.shape[0], nav_map.shape[1])
            self.maps[scene_id] = {
                "nav_map": nav_map,
                "room_map": room_map,
                "obj_maps": obj_maps,
                "bounds": bounds,
                "grid_dimensions": grid_dimensions
            }
        
        # Load path annotations
        self.data = []
        paths = find_all_ext(annt_root, 'jsonl')
        for file_path in paths:
            # num_lines = sum(1 for _ in open(file_path))
            # if num_lines < 2:
            #     print(f"start point might wrong in file {file_path}")
            #     continue

            with jsonlines.open(file_path) as reader:
                for obj in reader:
                    self.data.append(obj)
        
        # embeddings
        self.embedding_layer = load_embeddings()

        # room campass chunks
        self.room_compass_chunks = config.room_compass_chunks

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        datum = self.data[idx]
        maps = self.maps[datum['scene_name']]

        instruction = datum['instruction']['instruction_text']
        # instruction = self.embedding_layer(torch.tensor(datum['instruction']['instruction_tokens']).long())
        d_id = datum['id']
        target = datum['score']
        start = datum['start']
        goal = datum['goal']

        path = datum['path']

        key_point_objs = []
        key_point_room = []
        for agent_stat in path:
            room_satus = get_room_compass(
                maps['room_map'], maps['nav_map'], (*agent_stat[0], agent_stat[1]),
                radius=10, num_chunks=self.room_compass_chunks, is_radian=True
            )
            _, relative_dict = get_surrounding_objs(
                maps['obj_maps'], maps['nav_map'], (*agent_stat[0], agent_stat[1]), 
                3, floor_idx=1, is_radian=True
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
            
            # object compass tensor 
            obj_pt_tensor = torch.tensor(points)

            obj_pt_tensor = torch.cat([obj_pt_tensor[:, :3], self.embedding_layer(obj_pt_tensor[:, 3].long())], dim=1)
            # print(obj_pt_tensor.size())
            key_point_objs.append(obj_pt_tensor)

            # room compass tensor
            rooms = []
            for i in range(self.room_compass_chunks):
                m = room_satus.get(i, None)
                if m is not None:
                    rooms.append(room_layeridx2wordidx[m[0]])
                else:
                    rooms.append(0)
            rooms_emb = self.embedding_layer(torch.tensor(rooms).long())
            # print(rooms_emb.size())
            key_point_room.append(rooms_emb)
            
        return instruction, key_point_objs, key_point_room, target[0]
    
    def collate_fc(self, batch):
        instructions = default_collate([b[0] for b in batch])
        seq_len = []
        for b in batch:
            assert len(b[1]) == len(b[2])
            seq_len.append(len(b[1]))
        key_point_objs = torch.cat([torch.stack(b[1]) for b in batch], dim=0)
        key_point_room = torch.cat([torch.stack(b[2]) for b in batch], dim=0)
        targets = default_collate([b[3] for b in batch])
        return instructions, key_point_objs, key_point_room, seq_len, targets


if __name__ == "__main__":
    annt_path='/data/leuven/335/vsc33595/ProjectsVSC/DisMLN/dset_gen/samples'
    dset = MLNv1_Dis_Dataset(annt_path)
    from torch.utils.data import DataLoader
    dloader = DataLoader(dset, batch_size=4, shuffle=True, collate_fn=dset.collate_fc)
    batch = next(iter(dloader))
    print(batch[3])
    print(batch[1].size())
    import ipdb;ipdb.set_trace() # breakpoint 54

    print()