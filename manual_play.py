import segmentation
import gym
import numpy as np
import math
from config import *

def main():
    env = gym.make("SegmentationEnv-v0", 
        objs_dir=args.objs_dir, 
        max_scenes=args.max_scenes,
        sample_size=args.sample_size,
        diff_punishment=args.diff_punishment,
        max_steps_per_scene=args.max_steps_per_scene,
        scene_mode=args.scene_mode,
        point_mode=args.point_mode,
        voxel_size=args.voxel_size,
        voxel_mode=args.voxel_mode,
        single_scenes=args.single_scenes,
        early_diff=args.early_diff)

    total_reward = 0
    obs = env.reset()
    while(True):
        #env.render()
        env.render_state()
        try:
            quit = input("Quit [y/n]: ")
            if quit == "y":
                break
            seed_point_x = float(input("Seed Point X: "))
            seed_point_x /= env.max_params[0]
            seed_point_y = float(input("Seed Point Y: "))
            seed_point_y /= env.max_params[1]
            seed_point_z = float(input("Seed Point Z: "))
            seed_point_z /= env.max_params[2]
            K = float(input("K: "))
            K /= env.max_params[3]
            K *= 2
            K -= 1
            #K *= env.action_space.high[0,0]
            angle_threshold = float(input("Angle Threshold: "))
            angle_threshold /= 180
            angle_threshold *= 2
            angle_threshold -= 1
            #angle_threshold *= env.action_space.high[0,0]
            curvature_threshold = float(input("Curvature Threshold: "))
            curvature_threshold /= env.max_params[-1]
            curvature_threshold *= 2
            curvature_threshold -= 1
        except:
            continue
        action = np.array([seed_point_x, seed_point_y, seed_point_z, K, angle_threshold, curvature_threshold])
        obs, reward, done, info = env.step(action)
        total_reward += reward
        print(obs.shape, reward, total_reward, done, info)
        if done:
            env.render()
            total_reward = 0
            obs = env.reset()
            
if __name__ == "__main__":
    main()
