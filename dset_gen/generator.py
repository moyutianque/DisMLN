import os
import os.path as osp
import gzip
import json
import numpy as np
import math
import multiprocessing
from multiprocessing import Pool
from utils.map_tools import shortest_path, simloc2maploc, get_maps, create_candidates, get_agent_orientation
import random
random.seed(0)
from scipy.spatial import distance
import copy
import jsonlines
num_process = 50

class dset_generator(object):
    def __init__(self, nav_radius=42, split='train', sp_radius=1, sp_step=1, discretized_in_meter=2, meters_per_pixel=0.05, sample_gap=1.2) -> None:
        """
        Args:
            nav_radius: in meters
            sp_radius: in meters
            sp_step: in pixel
            discretized_in_meter: in meters
            meters_per_pixel: in meters
            sample_gap: in meters
        """
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
        self.dis_steps = int(discretized_in_meter / (meters_per_pixel * sp_step))
        assert self.dis_steps > 0

    def process_one_ep(self, episode):
        # rank_id = multiprocessing.current_process()._identity[0]
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
        # TODO: shift start pos if it not on valid position (sometimes happened)
        xs = [0,0,-1,-1,-1,1,1,1]
        ys = [1,-1,0,1,-1,0,1,-1]
        r,c = start_grid_pos
        if nav_map[r,c] <=0 and obj_maps[r,c,1] <= 0:
            for x,y in zip(xs, ys):
                if nav_map[r+x, c+y] > 0:
                    r = r+x
                    c = c+y
                    break
        
        assert nav_map[r,c] > 0 or obj_maps[r,c,1] > 0, f"\n{nav_map[r-2:r+3,c-2:c+3]}\n {obj_maps[r-2:r+3,c-2:c+3,1]}"
        start_grid_pos = (r,c)

        end_grid_pos = simloc2maploc(
            end_position, grid_dimensions, upper_bound, lower_bound
        )
        end_radius = episode['goals'][0]['radius']
        start_rot = episode['start_rotation']

        # Processing data
        bound = self.create_bound(start_grid_pos, nav_map)

        # Add gt path
        gt_locations = self.gt_json[str(ep_id)]['locations']
        gt_path = []
        for point in gt_locations:
            gt_path.append(
                simloc2maploc(
                    point, grid_dimensions, upper_bound, lower_bound
                )
            )
        candidate_pathes, scores = self.get_candidate_paths(
            nav_map, obj_maps, start_grid_pos, get_agent_orientation(start_rot), end_grid_pos, gt_path, bound
        )

        for i in range(len(candidate_pathes)):
            out_dict= {
                "id": f"{ep_id}-{i}",
                "scene_name": scene_name,
                "instruction": instruction,
                "path": candidate_pathes[i],
                "score": scores[i],
                "start": start_grid_pos,
                "start_rot": start_rot,
                "goal": {
                    "radius": end_radius,
                    "end_point": end_grid_pos
                }
            }
            with jsonlines.open(f'./dset_gen/generated/raw/{self.split}_{ep_id}.jsonl', mode='a') as writer:
                writer.write(out_dict)
                
    def gen(self):
        # out_dict = {}
        pool = Pool(num_process)
        for episode in self.eps:
            pool.apply_async(self.process_one_ep, args=(episode, ))

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
        for i in range(0,len(path), self.dis_steps):
            if i==0:
                continue
            if i == len(path)-1:
                break

            p1 = path[max(i-rot_smooth_range, 0)]
            p2 = path[i]
            p3 = path[min(i+rot_smooth_range, len(path)-1)]
            dis_path.append((path[i], self.get_rot(p3, p1)))
        
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

    def get_candidate_paths(self, nav_map, obj_maps, start_point, start_rot, goal_point, gt_path, bound):
        """
        Args:
            obj_maps: raw object maps, index 1 indicate floor
        """
        solver = shortest_path(nav_map, obj_maps, start_point, self.sp_radius, step=self.sp_step, bound=bound)
        candidate_targets = create_candidates(
            nav_map, obj_maps, sample_gap=self.sample_gap, floor_idx = 1, meters_per_pixel=self.meters_per_pixel, bound = bound
        )

        candidate_pathes = []
        for target in candidate_targets:
            path = solver.find_path_by_target(target).tolist()
            if len(path) < 3:
                continue
            candidate_pathes.append(path)
        candidate_pathes.append(gt_path)

        # gen score
        scores = []
        for i in range(len(candidate_pathes)):
            candidate_pathes[i] = self.discretize_path(candidate_pathes[i], start_rot)
            scores.append(self.get_target_score(candidate_pathes[i][-1][0], goal_point))
        return candidate_pathes, scores 

    
    
if __name__ == '__main__':
    generator=dset_generator()
    generator.gen()
    