import segmentation
import gym
import numpy as np
import math
from config import *
import os

def mk_dir(path):
    if not os.path.exists(path): 
        os.makedirs(path)

def clean(trajectories, steps):
    trajectories["actions"] = np.delete(trajectories["actions"], obj=np.s_[-steps:], axis=0)
    trajectories["episode_starts"] = np.delete(trajectories["episode_starts"], obj=np.s_[-steps:], axis=0)
    trajectories["rewards"] = np.delete(trajectories["rewards"], obj=np.s_[-steps:], axis=0)
    return trajectories

def main():
    data_dir = "./pretrain_data"
    mk_dir(data_dir)
    
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
    trajectories = {}
    trajectories["actions"] = np.zeros((0,env.action_space.shape[0]))
    trajectories["episode_starts"] = np.zeros((0,1), dtype=bool)
    
    obs_dirs=[]
    
    trajectories["rewards"] = np.zeros((0,1))
    trajectories["episode_returns"] = np.zeros((0,1))
    episode_start = True
    n_episodes = 1000
    episode = 0
    random_agent=True
    quality_threshold = 0.9
    steps=0
    episodes_to_save = 5
    saved=False
    traj_step = 0
    observations = []
    while(episode < n_episodes):
        if random_agent:
            action = (np.random.rand(env.action_space.shape[0]) * 2) - 1
        else:
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

        trajectories["actions"] = np.vstack((trajectories["actions"], action))
        ep = np.array([[episode_start]])
        trajectories["episode_starts"] = np.vstack((trajectories["episode_starts"], np.array([[episode_start]], dtype=bool)))
        observations.append(obs)
        
        obs, reward, done, info = env.step(action)
        
        episode_start = done
        trajectories["rewards"] = np.vstack((trajectories["rewards"], np.array([reward])))
        total_reward += reward
        steps += 1
        if done:
            if not random_agent:
                env.render()
            print(total_reward)
            
            if(total_reward < quality_threshold):
                trajectories = clean(trajectories, steps)
            else:
                trajectories["episode_returns"] = np.vstack((trajectories["episode_returns"], np.array([[total_reward]])))
                
                for i in range(steps):
                    obs_file = data_dir + "/obs_" + str(traj_step + i) + ".npz"
                    np.savez(obs_file, **{"obs": observations[i]})
                    obs_dirs.append(obs_file)
                    
                traj_step += steps
                episode += 1
                saved = False
                print("episode: ", episode, "reward: ", total_reward)
                #env.render()
            
            total_reward = 0
            steps=0
            obs = env.reset()
            if trajectories["episode_returns"].shape[0] % episodes_to_save == 0 and not saved:
                trajectories["obs"] = np.array(obs_dirs)
                np.savez("expert_trajectories.npz", **trajectories)
                saved=True
                print("save")
            observations.clear()
    if not saved:
        trajectories["obs"] = np.array(obs_dirs)
        np.savez("expert_trajectories.npz", **trajectories)
    print("finish")
            
if __name__ == "__main__":
    main()
