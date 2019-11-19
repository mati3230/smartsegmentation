import segmentation
import gym
from config import *
import os
import importlib

from stable_baselines.common.vec_env import DummyVecEnv, VecCheckNan
from stable_baselines import PPO2
from stable_baselines.results_plotter import load_results, ts2xy
from stable_baselines.gail import ExpertDataset
from stable_baselines import GAIL
from stable_baselines.bench import Monitor

import tensorflow as tf
import numpy as np

def mk_dir(path):
    if not os.path.exists(path): 
        os.makedirs(path)

best_mean_reward, n_steps = -np.inf, 0
save_path = log_dir = ""
model=None

def callback(_locals, _globals):
    """
    Callback called at each step (for DQN an others) or after n steps (see ACER or PPO2)
    :param _locals: (dict)
    :param _globals: (dict)
    """
    global n_steps, best_mean_reward, save_path, log_dir
    # Print stats every 10 calls
    if (n_steps + 1) % args.save_interval == 0:
        # Evaluate policy training performance
        x, y = ts2xy(load_results(log_dir), "timesteps")
        if len(x) > 0:
            mean_reward = np.mean(y[-100:])
            print(x[-1], 'timesteps')
            print("Best mean reward: {:.2f} - Last mean reward per episode: {:.2f}".format(best_mean_reward, mean_reward))

            # New best model, you could save the agent here
            if mean_reward > best_mean_reward:
                best_mean_reward = mean_reward
                # Example for saving best model
                print("Saving new best model")
                model.save(save_path)
    n_steps += 1
    return True

def main():
    global save_path, log_dir, model, best_mean_reward
    mk_dir(args.checkpoint_dir + args.policy)
    save_path = args.checkpoint_dir + args.policy + "/" + args.policy
    log_dir = args.summary_dir + args.policy
    mk_dir(log_dir)
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
        early_diff=args.early_diff,
        wall_weight=args.wall_weight)
    env = Monitor(env, log_dir, allow_early_resets=True)
        
    env = DummyVecEnv([lambda: env])  # The algorithms require a vectorized environment to run
    env = VecCheckNan(env, raise_exception=True)
    
    net_module = importlib.import_module(args.policy)
    model = PPO2(net_module.Policy, env, verbose=args.verbose, tensorboard_log=log_dir, learning_rate=args.learning_rate, ent_coef=args.ent_coef,
                cliprange=args.cliprange, cliprange_vf=args.cliprange_vf, lam=args.lam, gamma=args.gamma, seed=args.seed, n_cpu_tf_sess=args.n_cpu_tf_sess,
                noptepochs=args.noptepochs, nminibatches=args.nminibatches, n_steps=args.n_steps, max_grad_norm=args.max_grad_norm)
    
    if os.path.isfile("expert_trajectories.npz") and args.pretrain==1: 
        print("------------start pretrain------------")
        #dataset = ExpertDataset(expert_path="expert_trajectories.npz", special_shape=True, traj_limitation=100, batch_size=16)
        dataset = ExpertDataset(expert_path="expert_trajectories.npz", special_shape=True, train_fraction=args.train_fraction, batch_size=args.pretrain_batch_size)
        #model.pretrain(dataset, learning_rate=0.001, n_epochs=1000)
        model = model.pretrain(dataset, val_interval=1, learning_rate=args.pretrain_learning_rate, n_epochs=args.pretrain_n_epochs)
        print("pretrain finished -- save model")
        model.save(save_path)
        returns = []
        
        print("Calculate mean reward")
        n_episodes = 10
        for i in range(n_episodes):
            total_reward = 0
            obs = env.reset()
            while True:
                action, _states = model.predict(obs, deterministic=True)
                obs, reward, done, info = env.step(action)
                total_reward += reward
                if done: 
                    returns.append(total_reward)
                    break
        returns = np.array(returns)
        best_mean_reward = np.mean(returns)
        print("Best mean reward: {:.2f}".format(best_mean_reward))
    
    model.learn(total_timesteps=args.total_timesteps, callback=callback)
    env.close()

if __name__ == "__main__":
    main()
