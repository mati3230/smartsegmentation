import argparse

parser = argparse.ArgumentParser()

# policy arguments
parser.add_argument("--policy", type=str, default="pointnet", help="The policy that will be used for training. The available policies are 'pointnet', 'ldgcnn' and 'vox_custom'. Default: pointnet")
parser.add_argument("--checkpoint_dir", type=str, default="./save_model/", help="The relative directory where the trained model will be saved and loaded., Default: ./save_model/")
parser.add_argument("--summary_dir", type=str, default="./summary_log/", help="The relative directory where the tensorboard log files will be saved. Default: ./summary_log/")
parser.add_argument("--total_timesteps", type=int, default=5000000, help="maximum number of time steps that will be conducted to train the agent. Default: 5000000")
parser.add_argument("--debugging", type=bool, default=False, help="Flag that can be used to print informations in the policies. Default: False")
parser.add_argument("--save_interval", type=int, default=2, help="An eventually better model will saved after n_steps. Default: 2")

# pretrain arguments
parser.add_argument("--pretrain", type=int, default=0, help="Flag to start the pretraining. Set this variable to 1 to execute the pretraining. An expert_trajectories.npz and pretrain_data folder with trajectories should be available. Default: 0")
parser.add_argument("--pretrain_learning_rate", type=float, default=0.00025, help="Learning rate for the pretraining. Default: 0.00025")
parser.add_argument("--train_fraction", type=float, default=0.9, help="How much of the pretrain data should be used for the training. The remaining data will be used for the evaluation. Default: 0.9")
parser.add_argument("--pretrain_n_epochs", type=int, default=1000, help="Number of epochs of the pretraining. Default: 1000")
parser.add_argument("--pretrain_batch_size", type=int, default=4, help="Batch size of the pretraining. Default: 4")

# training environment parameters
parser.add_argument("--objs_dir", type=str, default="./", help="Location of the 'PointcloudScenes' directory. Default: ./")
parser.add_argument("--sample_size", type=int, default=1024, help="How many points should be sampled from the point cloud as observation for the agent. Default: 1024")
parser.add_argument("--diff_punishment", type=float, default=0.0125, help="Segment difference factor that will be applied in the reward calculation. The agent will be punished for estimating to many or less objects. Default: 0.0125")
parser.add_argument("--max_steps_per_scene", type=int, default=15, help="How many segmentation steps can be done to segment the point cloud. Default: 15")
parser.add_argument("--max_scenes", type=int, default=4, help="How many scenes will be considered for the segmentation. Default: 4")
parser.add_argument("--scene_mode", type=str, default="Random", help="The order in which the scenes are presented. Possible values are 'Random' and 'Linear'. Default: Random")
parser.add_argument("--point_mode", type=str, default="None", help="Determines the representation of the point cloud. If the value 'None' is selected, the point cloud will be represented as sampled point cloud. The query points will be included in the state representation in the 'Query' mode. The mode 'Voxel' will represent the point cloud as voxel grid. Default: None")
parser.add_argument("--voxel_size", type=int, default=60, help="The size of the voxel grid (e.g. a size of 60 produces a voxel grid of dimension 60x60x60). Default: 60")
parser.add_argument("--voxel_mode", type=str, default="Custom", help="Determines the features of a voxel. If 'None' is selected, a voxel has only a occupied feature. The features occupied, mean normal and mean curvature are available in the 'Custom' mode. Default: Custom")
parser.add_argument("--single_scenes", type=bool, default=True, help="If 'True', the segmentation of one scene counts as an episode. If 'False', the segmentation of all scenes account as one episode. Default: True")
parser.add_argument("--early_diff", type=bool, default=True, help="If 'True', the segment difference will be applied after more segmentation steps are applied than objects in the scene. If 'False', the segment difference will be applied at the end of hte episode. Default: True")
parser.add_argument("--wall_weight", type=float, default=0.5, help="The weight of the wall objects in the reward calculation. Default: 0.5")

# PPO parameters
parser.add_argument("--verbose", type=int, default=1, help="If '1', some informations about the training progress are printed in the console. Default: 1")
parser.add_argument("--learning_rate", type=float, default=0.00025, help="learning rate of the optimisation algorithm. Default: 0.00025")
parser.add_argument("--ent_coef", type=float, default=0.0001, help="Entropy coefficient of the PPO algorithm. Default: 0.0001")
parser.add_argument("--cliprange", type=float, default=0.2, help="Clipping value of the PPO algorithm. Default: 0.2")
parser.add_argument("--cliprange_vf", type=float, default=-1.0, help="Optional clipping range the value function. If '-1' is selected, no clipping range for the value function is applied. Default: -1.0")
parser.add_argument("--lam", type=float, default=0.95, help="The lambda factor for the generalised advantage estimation of the PPO algorihtm. Default: 0.95")
parser.add_argument("--gamma", type=float, default=0.99, help="The discount factor. Default: 0.99")
parser.add_argument("--seed", type=int, default=42, help="Random seed value for the PPO algorithm. Default: 42")
parser.add_argument("--n_cpu_tf_sess", type=int, default=1, help="Number of CPUs that used during the training. Default: 1")
parser.add_argument("--noptepochs", type=int, default=4, help="Number of epochs for the training with one batch. Default: 4")
parser.add_argument("--nminibatches", type=int, default=16, help="How many batches should be created from n_steps. Default: 16")
parser.add_argument("--n_steps", type=int, default=1024, help="Determines the buffer size to optimise the agent. Default: 1024")
parser.add_argument("--max_grad_norm", type=float, default=0.5, help="Clip parameter for the global norm of the gradient update. Default: 0.5")


args = parser.parse_args()
print(args)
