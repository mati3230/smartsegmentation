import tensorflow as tf
import numpy as np
import tf_util as tu

from stable_baselines.common.policies import ActorCriticPolicy

def feature_transform_net(inputs, is_training, bn_decay=None, K=64, bn=False):
    """ Feature Transform Net, input is BxNx1xK
        Return:
            Transformation matrix of size KxK """
    batch_size = tf.shape(inputs)[0]
    
    num_point = inputs.get_shape()[1].value

    f=1
    if inputs.get_shape()[2].value != 1:
        f=K
    net = tu.conv2d(inputs, 64, [1,f],
                         padding='VALID', stride=[1,1],
                         bn=bn, is_training=is_training,
                         scope='tconv1', bn_decay=bn_decay)
    net = tu.conv2d(net, 128, [1,1],
                         padding='VALID', stride=[1,1],
                         bn=bn, is_training=is_training,
                         scope='tconv2', bn_decay=bn_decay)
    
    net = tu.max_pool2d(net, [num_point,1],
                             padding='VALID', scope='tmaxpool')

    #net = tf.reshape(net, [batch_size, -1])
    net = tf.reshape(net, [batch_size, net.get_shape()[-1].value])
    
    net = tu.fully_connected(net, 256, bn=bn, is_training=is_training,
                                  scope='tfc2', bn_decay=bn_decay)

    with tf.variable_scope('transform_feat') as sc:
        weights = tf.get_variable('weights', [256, K*K],
                                  initializer=tf.constant_initializer(0.0),
                                  dtype=tf.float32)
        biases = tf.get_variable('biases', [K*K],
                                 initializer=tf.constant_initializer(0.0),
                                 dtype=tf.float32)
        biases += tf.constant(np.eye(K).flatten(), dtype=tf.float32)
        transform = tf.matmul(net, weights)
        transform = tf.nn.bias_add(transform, biases)

    transform = tf.reshape(transform, [batch_size, K, K])
    return transform

def make_pointnet(inpt, is_training, bn_decay=None, bn=False):
    print("Input: ", inpt.get_shape())
    """ Classification PointNet, input is BxNx3, output BxNx50 """
    num_point = inpt.get_shape()[1].value
    num_features = inpt.get_shape()[2].value

    input_image = tf.expand_dims(inpt, -1)
    with tf.variable_scope("transform_net1") as sc:
        transform = feature_transform_net(input_image, is_training=is_training, bn_decay=None, K=num_features, bn=bn)
    point_cloud_transformed = tf.matmul(inpt, transform)
    print("point_cloud_transformed: ", point_cloud_transformed)
    # B x N x 3 x 1
    input_image = tf.expand_dims(point_cloud_transformed, -1)
    print("Input Image: ", input_image)

    net = tu.conv2d(input_image, 64, [1,num_features],
                         padding="VALID", stride=[1,1],
                         bn=bn, is_training=is_training,
                         scope="conv1", bn_decay=bn_decay)
    print("Conv 1: ", net)
    # B x N x 1 x 64
    with tf.variable_scope("transform_net2") as sc:
        transform = feature_transform_net(net, is_training=is_training, bn_decay=None, K=64, bn=bn)
    #end_points['transform'] = transform
    # B x N x 64
    #net_transformed = tf.matmul(tf.squeeze(net, axis=[2]), transform)
    net_transformed = tf.matmul(tu.squeeze_tensor(net), transform)
    print("net_transformed: ", net_transformed)
    # B x N x 1 x 64
    point_feat = tf.expand_dims(net_transformed, [2])
    print("point_feat: ", point_feat)
    net = tu.conv2d(point_feat, 64, [1,1],
                         padding="VALID", stride=[1,1],
                         bn=bn, is_training=is_training,
                         scope="conv2", bn_decay=bn_decay)
    print("Conv 2: ", net)
    net = tu.conv2d(net, 128, [1,1],
                         padding="VALID", stride=[1,1],
                         bn=bn, is_training=is_training,
                         scope="conv3", bn_decay=bn_decay)
    print("Conv 3: ", net)
    
    global_feat = tu.max_pool2d(net, [num_point,1],
                                     padding="VALID", scope="maxpool")

    print("global_feat: ", global_feat)
    global_feat_expand = tf.tile(global_feat, [1, num_point, 1, 1])
    print("global_feat_expand: ", global_feat_expand)
    
    concat_feat = tf.concat(values=[point_feat, global_feat_expand], axis=3)
    print(concat_feat)
    
    net = tu.conv2d(concat_feat, 64, [1,1],
                         padding="VALID", stride=[1,1],
                         bn=bn, is_training=is_training,
                         scope="conv4", bn_decay=bn_decay)
    print("Conv 4: ", net)
    
    net = tu.conv2d(net, 16, [1,1],
                         padding="VALID", stride=[1,1],
                         scope="conv5")
    print("Conv 5: ", net)
    net = tu.conv2d(net, 1, [1,1],
                         padding="VALID", stride=[1,1],
                         scope="conv6")
    print("Conv 6: ", net.get_shape())
    
    with tf.name_scope("fc2"):
        net = tf.layers.flatten(inputs=net, name="flatten1")
        net = tf.layers.dense(net, 128, activation=tf.nn.relu)
        print("FC 2: ", net)

    return net

class Policy(ActorCriticPolicy):
    def __init__(self, sess, ob_space, ac_space, n_env, n_steps, n_batch, reuse=False, **kwargs):
        super(Policy, self).__init__(sess, ob_space, ac_space, n_env, n_steps, n_batch, reuse=reuse, scale=True)
        
        with tf.variable_scope("model", reuse=reuse):
            is_training = tf.constant(True)
            bn = False
            net = tf.cast(self.obs_ph, tf.float32)
            net = make_pointnet(net, is_training, bn = bn)
            with tf.name_scope("pi_h_fc1"):
                #pi_h = tu.fully_connected(net, 8, bn=bn, is_training=is_training, scope="pi_h_fc1", bn_decay=None)
                pi_h = tf.layers.dense(net, 8, name="pi_h_fc1")
                pi_h = tf.clip_by_value(
                    t=pi_h,
                    clip_value_min=-1,
                    clip_value_max=1,
                    name="action_clipping"
                )
            pi_latent = pi_h
            
            
            with tf.name_scope("vf_h_fc1"):
                vf_h = tu.fully_connected(net, 16, bn=bn, is_training=is_training,
                                  scope="vf_h_fc1", bn_decay=None)
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
