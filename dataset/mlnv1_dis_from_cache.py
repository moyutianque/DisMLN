import torch
from torch.utils.data import Dataset
import jsonlines
from utils.file_process_tools import find_all_ext

from torch.utils.data._utils.collate import default_collate
from utils.parse_wordmap import load_embeddings
from utils.pos_emb import get_embedder
import pickle as pkl

class MLNv1_Dis_Dataset_Cached(Dataset):
    def __init__(self, config, split):
        # Load maps
        self.target_type = config.target_type
        self.ndtw_weight = config.ndtw_weight
        annt_root = config.annt_root.format(split=split)
        # Load path annotations
        self.data = []
        paths = find_all_ext(annt_root, 'pkl')
        for file_path in paths:
            with open(file_path, 'rb') as handle:
                b = pkl.load(handle)
                self.data.extend(b)
        print(f"Successfully load {len(self.data)} data for split {split}")
        
        # embeddings
        self.embedding_layer = load_embeddings()
        self.agent_loc_emb, _ = get_embedder(10)
        self.agent_rot_emb, _ = get_embedder(4)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        datum = self.data[idx]
        instruction = datum['instruction']
        room_list = datum['room_list']
        points_list = datum['points_list']
        if len(room_list) == 0:
            print(datum)
        assert len(room_list) > 0
        assert len(points_list) > 0

        if self.target_type == 'dis_score':
            target = datum['target']
            target = target/10.0
        elif self.target_type == 'ndtw':
            target = datum['ndtw']
        elif self.target_type == 'both':
            target = datum['ndtw'] * self.ndtw_weight + datum['target']/10.0 * (1-self.ndtw_weight) # weighted sum of target
        else:
            raise NotImplementedError()
        
        normalized_traj = torch.tensor(datum['normalized_traj'])
        loc_emb = self.agent_loc_emb(normalized_traj[:, :3])
        rot_emb = self.agent_rot_emb(normalized_traj[:, 3:])
        agent_pos_emb = torch.cat([loc_emb, rot_emb], dim=1)
        info={
            "scene_name": datum['scene_name'],
            "ep_id": datum['ep_id'],
            "ndtw": datum['ndtw'],
            "dis_score": datum["target"],
        }

        key_point_objs = []
        key_point_room = []
        for points, rooms in zip(points_list, room_list):
            obj_pt_tensor = torch.tensor(points)
            obj_pt_tensor = torch.cat([obj_pt_tensor[:, :3], self.embedding_layer(obj_pt_tensor[:, 3].long())], dim=1)
            key_point_objs.append(obj_pt_tensor)
            rooms_emb = self.embedding_layer(torch.tensor(rooms).long())
            key_point_room.append(rooms_emb)
            
        return instruction, key_point_objs, key_point_room, agent_pos_emb, target, info
    
    def collate_fc(self, batch):
        instructions = default_collate([b[0] for b in batch])
        seq_len = []
        for b in batch:
            assert len(b[1]) == len(b[2])
            seq_len.append(len(b[1]))
        key_point_objs = torch.cat([torch.stack(b[1]) for b in batch], dim=0)
        key_point_room = torch.cat([torch.stack(b[2]) for b in batch], dim=0)
        agent_pos_embs = [b[3] for b in batch]

        targets = default_collate([b[4] for b in batch])
        infos = [ b[5] for b in batch]
        return instructions, key_point_objs, key_point_room, agent_pos_embs, seq_len, targets, infos

