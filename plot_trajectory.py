import segmentation
import gym
import numpy as np
from config import *

def main():
    traj = np.load("expert_trajectories.npz")
    #start_idx = np.where(traj["episode_starts"][:]==True)[0]
    #actions = traj["actions"][start_idx[0]:start_idx[1]]
    actions = traj["actions"]
    
    res = np.all(traj["rewards"] >= 0.1)
    assert res
    
    env = gym.make("SegmentationEnv-v0", 
        objs_dir=args.objs_dir, 
        max_scenes=1,
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

    obs = env.reset()
    done=False
    total_reward = 0
    for i in range(actions.shape[0]):
        action = actions[i]
        obs, reward, done, info = env.step(action)
        print(reward)
        total_reward += reward
        if done:
            break 
    print("Total Reward: ", total_reward, ", Done: ", done)
    env.render()
    env.close()

if __name__ == "__main__":
    main()
