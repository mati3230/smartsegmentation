import gym
import tensorflow as tf
import tf_util as tu

from stable_baselines.common.policies import ActorCriticPolicy, register_policy, nature_cnn
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines import A2C

# Custom MLP policy of three layers of size 128 each for the actor and 2 layers of 32 for the critic,
# with a nature_cnn feature extractor
class Policy(ActorCriticPolicy):
    def __init__(self, sess, ob_space, ac_space, n_env, n_steps, n_batch, reuse=False, **kwargs):
        super(Policy, self).__init__(sess, ob_space, ac_space, n_env, n_steps, n_batch, reuse=reuse, scale=True)

        print(self.obs_ph.get_shape())
        net = tf.cast(self.obs_ph, tf.float32)
        #net = tf.Print(net, [net], "input: ", summarize=1000)
        print(net.get_shape())
        with tf.variable_scope("model", reuse=reuse):
            net = tu.conv3d(inputs=net,
               num_output_channels=16,
               kernel_size=[6,6,6],
               scope="conv1",
               stride=[1, 1, 1],
               padding="VALID")
            print(net.get_shape())
            net = tu.max_pool3d(inputs=net,
               kernel_size=[3,3,3],
               scope="pool1",
               stride=[2, 2, 2],
               padding="VALID")
            print(net.get_shape())
               
            net = tu.conv3d(inputs=net,
               num_output_channels=32,
               kernel_size=[5,5,5],
               scope="conv2",
               stride=[1, 1, 1],
               padding="VALID")
            print(net.get_shape())
            net = tu.max_pool3d(inputs=net,
               kernel_size=[3,3,3],
               scope="pool2",
               stride=[2, 2, 2],
               padding="VALID")
            print(net.get_shape())
               
            net = tu.conv3d(inputs=net,
               num_output_channels=64,
               kernel_size=[3,3,3],
               scope="conv3",
               stride=[1, 1, 1],
               padding="VALID")
            print(net.get_shape())
            net = tu.max_pool3d(inputs=net,
               kernel_size=[3,3,3],
               scope="pool3",
               stride=[2, 2, 2],
               padding="VALID")
            print(net.get_shape())
               
            net = tu.conv3d(inputs=net,
               num_output_channels=64,
               kernel_size=[2,2,2],
               scope="conv4",
               stride=[1, 1, 1],
               padding="VALID")
            print(net.get_shape())
            net = tu.max_pool3d(inputs=net,
               kernel_size=[3,3,3],
               scope="pool4",
               stride=[1, 1, 1],
               padding="VALID")
            print(net.get_shape())
            
            net = tf.layers.flatten(inputs=net)
            print(net.get_shape())
            
            with tf.name_scope("pi_h_fc1"):
                pi_h = tf.layers.dense(net, 8, activation=tf.nn.relu, kernel_initializer=tf.truncated_normal_initializer(stddev=1e-3))
                print(pi_h.get_shape())
            pi_latent = pi_h
            
            with tf.name_scope("vf_h_fc1"):
                vf_h = tf.layers.dense(net, 8, activation=tf.nn.relu, kernel_initializer=tf.truncated_normal_initializer(stddev=1e-3))
                print(vf_h.get_shape())
            value_fn = tf.layers.dense(vf_h, 1, name="vf")
            vf_latent = vf_h

            self._proba_distribution, self._policy, self.q_value = \
                self.pdtype.proba_distribution_from_latent(pi_latent, vf_latent, init_scale=0.01)

        self._value_fn = value_fn
        self._setup_init()

    def step(self, obs, state=None, mask=None, deterministic=False):
        if deterministic:
            action, value, neglogp = self.sess.run([self.deterministic_action, self.value_flat, self.neglogp],
                                                   {self.obs_ph: obs})
        else:
            action, value, neglogp = self.sess.run([self.action, self.value_flat, self.neglogp],
                                                   {self.obs_ph: obs})
        return action, value, self.initial_state, neglogp

    def proba_step(self, obs, state=None, mask=None):
        return self.sess.run(self.policy_proba, {self.obs_ph: obs})

    def value(self, obs, state=None, mask=None):
        return self.sess.run(self.value_flat, {self.obs_ph: obs})
