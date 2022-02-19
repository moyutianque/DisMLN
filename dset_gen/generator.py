import os
import os.path as osp
import gzip
import json
from tkinter import Y
import numpy as np
import math
import multiprocessing
from multiprocessing import Pool
from utils.map_tools import  simloc2maploc, get_maps, get_agent_orientation
from utils.map_tools2 import gen_valid_map, shortest_path2, create_candidates, euclidean_distance
import random
random.seed(0)
from scipy.spatial import distance
import copy
import jsonlines
from PIL import Image
num_process = 100
from fastdtw import fastdtw

import signal

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

out_root = './dset_gen/generated/raw_{split}'

class dset_generator(object):
    def __init__(self, nav_radius=21, split='train', sp_radius=1, sp_step=1, discretized_in_meter=2, meters_per_pixel=0.05, sample_gap=1.2) -> None:
        """
        Args:
            nav_radius: in meters
            sp_radius: in meters
            sp_step: in pixel
            discretized_in_meter: in meters
            meters_per_pixel: in meters
            sample_gap: in meters
        """
        print(f"generating data for {split} set")
        self.out_dir = out_root.format(split=split)
        os.makedirs(self.out_dir, exist_ok=True)
        self.split = split
        data_path = f"data/annt/{split}/{split}"
        with gzip.open(data_path+".json.gz", "rt") as f:
            annt_json = json.load(f)
        
        with gzip.open(data_path+"_gt.json.gz", "rt") as f:
            self.gt_json = json.load(f)
        
        episodes = annt_json['episodes']
        if os.environ.get('DEBUG', False): 
            self.eps = random.sample(episodes, 20)
        else:
            self.eps = episodes

        # Map config
        self.meters_per_pixel = meters_per_pixel
        self.map_root = f"data/maps/gmap_floor1_mpp_0.05_channel_last_with_bounds"
        self.sample_gap = sample_gap

        # Shortest path config
        self.nav_radius = nav_radius
        self.sp_radius=sp_radius
        self.sp_step = sp_step
        
        # Path Discretizing Config
        self.scale_percentage=100 # shortest path scale factor
        self.dis_steps = (discretized_in_meter / (meters_per_pixel * sp_step)) * (self.scale_percentage/100.)
        self.dis_radius = discretized_in_meter / (meters_per_pixel * sp_step)
        if self.dis_steps <= 0:
            self.dis_steps=1

    def process_one_ep(self, episode):
        ep_id = episode["episode_id"]
        gt_annt = self.gt_json[str(ep_id)]

        scene_name = episode['scene_id'].split('/')[1]
        self.scene_name = scene_name
        instruction = episode['instruction']

        start_position = episode['start_position']
        end_position = episode['goals'][0]['position']
        
        nav_map, room_map, obj_maps, grid_dimensions, bounds\
            = get_maps(scene_name, self.map_root, merged=False)
        
        upper_bound, lower_bound = bounds[0], bounds[1]

        # Agent positions
        start_grid_pos = simloc2maploc(
            start_position, grid_dimensions, upper_bound, lower_bound
        )

        # Processing data
        local_bound = self.create_bound(start_grid_pos, nav_map)
        _, valid_map_medianblured = gen_valid_map(nav_map, obj_maps, bound = None) # NOTE: we crop valid map in latter code
        r,c = start_grid_pos
        if not valid_map_medianblured[r,c]:
            for x in range(-1, 2):
                for y in range(-1,2):
                    if valid_map_medianblured[r+x,c+y]:
                        r = r+x
                        c = c+y
                        break
            if not valid_map_medianblured[r,c]:
                print("SKIP: ", scene_name, ep_id)
                return
        # pil_img = Image.fromarray((valid_map * 255).astype(np.uint8))
        # pil_img.save('./tmp/sample.png')
        # pil_img = Image.fromarray((valid_map_medianblured * 255).astype(np.uint8))
        # pil_img.save('./tmp/sample_blured.png')
        # print("=========")
        # exit()

        signal.alarm(60*5)
        try:
            start_grid_pos = (r,c)

            end_grid_pos = simloc2maploc(
                end_position, grid_dimensions, upper_bound, lower_bound
            )
            end_radius = episode['goals'][0]['radius']
            start_rot = episode['start_rotation']

            # Add gt path
            gt_locations = self.gt_json[str(ep_id)]['locations']
            gt_path = []
            for p_idx, point in enumerate(gt_locations):
                if p_idx == 0:
                    gt_path.append(start_grid_pos) # for skipping invalid start point
                else:
                    gt_path.append(
                        simloc2maploc(
                            point, grid_dimensions, upper_bound, lower_bound
                        )
                    )
            candidate_pathes, scores, original_candidate_path = self.get_candidate_paths(
                valid_map_medianblured[local_bound[0]:local_bound[2], local_bound[1]:local_bound[3]], 
                start_grid_pos, get_agent_orientation(start_rot), end_grid_pos, gt_path, 
                offset=(local_bound[0], local_bound[1])
            )
        except TimeoutException:
            print(f"Scene {scene_name} {ep_id} timeout")
        else:
            signal.alarm(0)
            gt_idx = -2
            # gt_locs = original_candidate_path[gt_idx]
            gt_locs = original_candidate_path[gt_idx]
            for i in range(len(candidate_pathes)):
                if (i >= len(candidate_pathes) -2) and self.split in ['val_seen', 'val_unseen']:
                    continue
                path = original_candidate_path[i]

                # Calculate DTW and NDTW
                dtw_distance = fastdtw(
                    path, gt_locs, dist=euclidean_distance
                )[0]
                nDTW = np.exp(
                    -dtw_distance
                    / (len(gt_locs) * 3.0 / self.meters_per_pixel)
                )
                
                out_dict= {
                    "id": f"{ep_id}-{i}",
                    "scene_name": scene_name,
                    "scene_shape": valid_map_medianblured.shape,
                    "instruction": instruction,
                    "path": candidate_pathes[i],
                    "score": scores[i],
                    "start": start_grid_pos,
                    "start_rot": start_rot,
                    "dtw": dtw_distance,
                    "gt_path": candidate_pathes[-1],
                    "ndtw": nDTW,
                    "goal": {
                        "radius": end_radius,
                        "end_point": end_grid_pos
                    }
                }
                with jsonlines.open(osp.join(self.out_dir, f'{ep_id}.jsonl'), mode='a') as writer:
                    writer.write(out_dict)

                
    def gen(self):
        # out_dict = {}
        pool = Pool(num_process)
        for episode in self.eps:
            pool.apply_async(self.process_one_ep, args=(episode, ))
            #self.process_one_ep(episode)

        pool.close()
        pool.join()
        print("success")

    # utils methods
    def create_bound(self, start_pos, nav_map):
        bound = int(round(self.nav_radius/self.meters_per_pixel))
        r,c = start_pos
        h,w = nav_map.shape[:2]
        lr = max(0, r - bound) 
        lc = max(0, c - bound) 
        hr = min(h, r + bound)
        hc = min(w, c + bound)
        return (lr, lc, hr, hc)
    
    def get_rot(self, point1, point2):
        return math.atan2(point2[1]-point1[1], point2[0]-point1[0])

    def discretize_path(self, path, start_rot, rot_smooth_range=2):
        """ discretize path and get agent orientation 
            Cases need to handle:
                1. path length shorter than dis_steps
                2. rotation of start and end point
                3. smooth index exceed range
        """
        dis_path = [(path[0], start_rot)] # start status
        for i in range(0,len(path), int(self.dis_steps)):
            if i==0:
                continue
            if i == len(path)-1:
                break

            p1 = path[max(i-rot_smooth_range, 0)]
            p2 = path[i]
            p3 = path[min(i+rot_smooth_range, len(path)-1)]
            dis_path.append((path[i], self.get_rot(p1, p3)))
        
        dis_path.append((path[-1], self.get_rot(path[-2], path[-1]))) # end status
        return dis_path

    def discretize_gtpath(self, path, start_rot, rot_smooth_range=2):
        dis_path = [(path[0], start_rot)] # start status
        for i in range(0,len(path)):
            if i==0:
                continue
            if i == len(path)-1:
                break

            if euclidean_distance(path[i], dis_path[-1][0]) < self.dis_radius:
                continue

            p1 = path[max(i-rot_smooth_range, 0)]
            p2 = path[i]
            p3 = path[min(i+rot_smooth_range, len(path)-1)]
            dis_path.append((path[i], self.get_rot(p1, p3)))
        
        dis_path.append((path[-1], self.get_rot(path[-2], path[-1]))) # end status
        return dis_path

    def get_target_score(self, point, goal_point):
        """ assign target label (dist, score_cls) """
        dist = distance.euclidean(point, goal_point) * self.meters_per_pixel
        if dist >= 5: # scoring region radius 5m
            score = 0
        else:
            score = 10 - int(dist / 0.5) # 10 level score 10 is the highest, 1 is the lowest
        return (score, dist)

    def get_candidate_paths(self, valid_map, start_point_raw, start_rot, goal_point, gt_path, offset=(0,0)):
        """
        Args:
            obj_maps: raw object maps, index 1 indicate floor
        """
        # most time consuming part
        start_point = (start_point_raw[0]-offset[0], start_point_raw[1]-offset[1]) 
        solver = shortest_path2(valid_map, start_point, self.sp_radius, step=self.sp_step, scale_percent=self.scale_percentage)

        candidate_targets = create_candidates(
            valid_map, sample_gap=self.sample_gap, floor_idx = 1, meters_per_pixel=self.meters_per_pixel
        )
        candidate_targets.append((goal_point[0]-offset[0], goal_point[1]-offset[1]))

        candidate_pathes = []
        for target in candidate_targets:
            path = solver.find_path_by_target(target)
            if len(path) < 3:
                continue
            path[:,0] += offset[0]
            path[:,1] += offset[1]
            candidate_pathes.append(path.tolist())
        candidate_pathes.append(gt_path)
        original_candidate_path = copy.deepcopy(candidate_pathes) 

        # gen score
        scores = []
        for i in range(len(candidate_pathes)-1):
            candidate_pathes[i] = self.discretize_path(candidate_pathes[i], start_rot)
            scores.append(self.get_target_score(candidate_pathes[i][-1][0], goal_point))
        
        candidate_pathes[-1] = self.discretize_gtpath(candidate_pathes[-1], start_rot)
        scores.append(self.get_target_score(candidate_pathes[-1][-1][0], goal_point))
        
        return candidate_pathes, scores, original_candidate_path

    
    
if __name__ == '__main__':
    # generator=dset_generator(split='val_seen')
    # generator=dset_generator(split='val_unseen')
    generator=dset_generator(split=args.split)
    generator.gen()


    #sssss