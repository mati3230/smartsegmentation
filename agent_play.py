import segmentation
import gym
from config import *
import os
import importlib

from stable_baselines.common.vec_env import DummyVecEnv, VecCheckNan
from stable_baselines import PPO2
from stable_baselines.results_plotter import load_results, ts2xy
from stable_baselines.bench import Monitor

import tensorflow as tf
import numpy as np

def main():
    save_path = args.checkpoint_dir + args.policy + "/" + args.policy
    env = gym.make("SegmentationEnv-v0", 
        objs_dir=args.objs_dir, 
        max_scenes=args.max_scenes,
        sample_size=args.sample_size,
        diff_punishment=args.diff_punishment,
        max_steps_per_scene=args.max_steps_per_scene,
        scene_mode=args.scene_mode,
        training=False,
        point_mode=args.point_mode,
        voxel_size=args.voxel_size,
        voxel_mode=args.voxel_mode,
		single_scenes=args.single_scenes,
        early_diff=args.early_diff)
        
    env = DummyVecEnv([lambda: env])  # The algorithms require a vectorized environment to run
    env = VecCheckNan(env, raise_exception=True)
    
    model = PPO2.load(save_path, env=env)
    
    n_episodes = 10
    for i in range(n_episodes):
        total_reward = 0
        obs = env.reset()
        while True:
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
            total_reward += reward
            if done: 
                print("Total Reward: ", total_reward)
                break
    
    env.close()

if __name__ == "__main__":
    main()
