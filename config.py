import argparse

parser = argparse.ArgumentParser()

parser.add_argument("--policy", type=str, default="pointnet")
parser.add_argument("--checkpoint_dir", type=str, default="./save_model/")
parser.add_argument("--summary_dir", type=str, default="./summary_log/")
parser.add_argument("--total_timesteps", type=int, default=5000000)
parser.add_argument("--debugging", type=bool, default=False)
parser.add_argument("--save_interval", type=int, default=2)


parser.add_argument("--pretrain", type=int, default=0)
parser.add_argument("--pretrain_learning_rate", type=float, default=0.00025)
parser.add_argument("--train_fraction", type=float, default=0.9)
parser.add_argument("--pretrain_n_epochs", type=int, default=1000)
parser.add_argument("--pretrain_batch_size", type=int, default=4)


parser.add_argument("--objs_dir", type=str, default="./")
parser.add_argument("--sample_size", type=int, default=1024)
parser.add_argument("--diff_punishment", type=float, default=0.0125)
parser.add_argument("--max_steps_per_scene", type=int, default=15)
parser.add_argument("--max_scenes", type=int, default=4)
parser.add_argument("--scene_mode", type=str, default="Random")
parser.add_argument("--point_mode", type=str, default="None")
parser.add_argument("--voxel_size", type=int, default=60)
parser.add_argument("--voxel_mode", type=str, default="Custom")
parser.add_argument("--single_scenes", type=bool, default=True)
parser.add_argument("--early_diff", type=bool, default=True)
parser.add_argument("--wall_weight", type=float, default=0.5)


parser.add_argument("--verbose", type=int, default=1)
parser.add_argument("--learning_rate", type=float, default=0.00025)
parser.add_argument("--ent_coef", type=float, default=0.0001)
parser.add_argument("--cliprange", type=float, default=0.2)
parser.add_argument("--cliprange_vf", type=float, default=-1.0)
parser.add_argument("--lam", type=float, default=0.95)
parser.add_argument("--gamma", type=float, default=0.99)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--n_cpu_tf_sess", type=int, default=1)
parser.add_argument("--noptepochs", type=int, default=4)
parser.add_argument("--nminibatches", type=int, default=16)
parser.add_argument("--n_steps", type=int, default=1024)
parser.add_argument("--max_grad_norm", type=float, default=0.5)


args = parser.parse_args()
print(args)
