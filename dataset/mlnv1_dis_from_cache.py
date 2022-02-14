import torch
from torch.utils.data import Dataset
import jsonlines
from utils.file_process_tools import find_all_ext

from torch.utils.data._utils.collate import default_collate
from utils.parse_wordmap import load_embeddings

class MLNv1_Dis_Dataset_Cached(Dataset):
    def __init__(self, config, split):
        # Load maps
        annt_root = config.annt_root.format(split=split)
        # Load path annotations
        self.data = []
        paths = find_all_ext(annt_root, 'jsonl')
        for file_path in paths:
            with jsonlines.open(file_path) as reader:
                for obj in reader:
                    self.data.append(obj)
        
        # embeddings
        self.embedding_layer = load_embeddings()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        datum = self.data[idx]
        instruction = datum['instruction']
        room_list = datum['room_list']
        points_list = datum['points_list']
        target = datum['target']

        key_point_objs = []
        key_point_room = []
        for points, rooms in zip(points_list, room_list):
            obj_pt_tensor = torch.tensor(points)
            obj_pt_tensor = torch.cat([obj_pt_tensor[:, :3], self.embedding_layer(obj_pt_tensor[:, 3].long())], dim=1)
            key_point_objs.append(obj_pt_tensor)
            rooms_emb = self.embedding_layer(torch.tensor(rooms).long())
            key_point_room.append(rooms_emb)
            
        return instruction, key_point_objs, key_point_room, target
    
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

