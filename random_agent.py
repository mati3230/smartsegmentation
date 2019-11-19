import numpy as np
import segmentation
import gym
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
        single_scenes=args.single_scenes)

    obs = env.reset()
    total_reward = 0
    episode_length = 0
    global_step = 0
    while(True):
        global_step += 1
        #env.render()
        #env.render_state()
        action = (np.random.rand(env.action_space.shape[0]) * 2) - 1
        obs, reward, done, info = env.step(action)
        #print(obs.shape, reward, done, info)
        total_reward += reward
        episode_length += 1
        if done:
            #env.render()
            obs = env.reset()
            total_reward = 0
            episode_length = 0
        
if __name__ == "__main__":
    main()