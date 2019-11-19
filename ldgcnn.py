import tensorflow as tf
import numpy as np
import tf_util

from stable_baselines.common.policies import ActorCriticPolicy

# Add input placeholder
def placeholder_inputs(batch_size, num_point):
    pointclouds_pl = tf.placeholder(tf.float32, shape=(batch_size, num_point, 3))
    labels_pl = tf.placeholder(tf.int32, shape=(batch_size))
    return pointclouds_pl, labels_pl 

# Input point cloud and output the global feature
def calc_ldgcnn_feature(point_cloud, is_training, bn_decay = None, bn = True):
    # B: batch size; N: number of points, C: channels; k: number of nearest neighbors
    # point_cloud: B*N*3
    k = 20
    
    # adj_matrix: B*N*N
    adj_matrix = tf_util.pairwise_distance(point_cloud)
    # Find the indices of knearest neighbors.
    nn_idx = tf_util.knn(adj_matrix, k=k)
    
    point_cloud = tf.expand_dims(point_cloud, axis = -2)
    # Edge_feature: B*N*k*6
    # The vector in the last dimension represents: (Xc,Yc,Zc, Xck - Xc, Yck-Yc, Yck-zc)
    # (Xc,Yc,Zc) is the central point. (Xck - Xc, Yck-Yc, Yck-zc) is the edge vector.
    edge_feature = tf_util.get_edge_feature(point_cloud, nn_idx=nn_idx, k=k)
    
    # net: B*N*k*64
    # The kernel size of CNN is 1*1, and thus this is a MLP with sharing parameters.
    net = tf_util.conv2d(edge_feature, 64, [1,1],
                         padding="VALID", stride=[1,1],
                         bn=bn, is_training=is_training,
                         scope="dgcnn1", bn_decay=bn_decay)
    
    # net: B*N*1*64
    # Extract the biggest feature from k convolutional edge features.     
    net = tf.reduce_max(net, axis=-2, keep_dims=True)
    net1 = net
    
    # adj_matrix: B*N*N
    adj_matrix = tf_util.pairwise_distance(net)
    nn_idx = tf_util.knn(adj_matrix, k=k)
    
    # net: B*N*1*67 
    # Link the Hierarchical features.
    net = tf.concat([point_cloud, net1], axis=-1)
    
    # edge_feature: B*N*k*134
    edge_feature = tf_util.get_edge_feature(net, nn_idx=nn_idx, k=k)
    
    # net: B*N*k*64
    net = tf_util.conv2d(edge_feature, 64, [1,1],
                         padding="VALID", stride=[1,1],
                         bn=bn, is_training=is_training,
                         scope="dgcnn2", bn_decay=bn_decay)
    # net: B*N*1*64
    net = tf.reduce_max(net, axis=-2, keep_dims=True)
    net2 = net
    
    adj_matrix = tf_util.pairwise_distance(net)
    nn_idx = tf_util.knn(adj_matrix, k=k)
    
    # net: B*N*1*131
    net = tf.concat([point_cloud, net1, net2], axis=-1)
    
    # edge_feature: B*N*k*262
    edge_feature = tf_util.get_edge_feature(net, nn_idx=nn_idx, k=k)  
    
    # net: B*N*k*64
    net = tf_util.conv2d(edge_feature, 64, [1,1],
                         padding="VALID", stride=[1,1],
                         bn=bn, is_training=is_training,
                         scope="dgcnn3", bn_decay=bn_decay)
    # net: B*N*1*64
    net = tf.reduce_max(net, axis=-2, keep_dims=True)
    net3 = net
    
    adj_matrix = tf_util.pairwise_distance(net)
    nn_idx = tf_util.knn(adj_matrix, k=k)
    
    # net: B*N*1*195
    net = tf.concat([point_cloud, net1, net2, net3], axis=-1)
    # edge_feature: B*N*k*390
    edge_feature = tf_util.get_edge_feature(net, nn_idx=nn_idx, k=k)  
    
    # net: B*N*k*128
    net = tf_util.conv2d(edge_feature, 128, [1,1],
                         padding="VALID", stride=[1,1],
                         bn=bn, is_training=is_training,
                         scope="dgcnn4", bn_decay=bn_decay)
    # net: B*N*1*128
    net = tf.reduce_max(net, axis=-2, keep_dims=True)
    net4 = net
    
    # input: B*N*1*323
    # net: B*N*1*1024
    net = tf_util.conv2d(tf.concat([point_cloud, net1, net2, net3, 
                                    net4], axis=-1), 1024, [1, 1], 
                         padding="VALID", stride=[1,1],
                         bn=bn, is_training=is_training,
                         scope="agg", bn_decay=bn_decay)
    # net: B*1*1*1024
    net = tf.reduce_max(net, axis=1, keep_dims=True)
    # net: B*1024
    net = tf_util.squeeze_tensor(net)
    return net

def get_model(point_cloud, is_training, bn_decay=None, bn = True):
    """ Classification PointNet, input is BxNx3, output Bx40 """
    batch_size = tf.shape(point_cloud)[0]
    layers = {}
    
    # Extract global feature
    net = calc_ldgcnn_feature(point_cloud, is_training, bn_decay, bn = bn)
    # MLP on global point cloud vector
    #net = tf.reshape(net, [batch_size, -1])
    layers["global_feature"] = net
    
    # Fully connected layers: classifier
    # net: B*512
    net = tf_util.fully_connected(net, 512, bn=bn, is_training=is_training,
                                  scope="fc1", bn_decay=bn_decay)
    layers["fc1"] = net
    # Each element is kept or dropped independently, and the drop rate is 0.5.
    net = tf_util.dropout(net, keep_prob=0.5, is_training=is_training,
                           scope="dp1")
    
    # net: B*256
    net = tf_util.fully_connected(net, 256, bn=bn, is_training=is_training,
                                  scope="fc2", bn_decay=bn_decay)
    layers["fc2"] = net
    net = tf_util.dropout(net, keep_prob=0.5, is_training=is_training,
                          scope="dp2")
    
    # net: B*40
    net = tf_util.fully_connected(net, 40, activation_fn=None, scope="fc3")
    layers["fc3"] = net
    return net, layers

class Policy(ActorCriticPolicy):
    def __init__(self, sess, ob_space, ac_space, n_env, n_steps, n_batch, reuse=False, **kwargs):
        super(Policy, self).__init__(sess, ob_space, ac_space, n_env, n_steps, n_batch, reuse=reuse, scale=True)
        
        with tf.variable_scope("model", reuse=reuse):
            # self.processed_obs [B:N:C]
            inpt = self.processed_obs[:,:,:3]
            labels = self.processed_obs[:,:,7]
            mask = tf.math.greater_equal(labels, tf.zeros_like(labels))
            mask = tf.math.logical_not(mask)
            mask = tf.expand_dims(mask, axis=-1)
            mask = tf.concat([mask,mask,mask], axis=-1)
            inpt = tf.multiply(inpt, tf.cast(mask, inpt.dtype))
            #inpt = tf.Print(inpt, [inpt], summarize=300)
            
            is_training = tf.constant(True)
            bn = False
            net, _ = get_model(inpt, is_training, bn = bn)
            with tf.name_scope("pi_h_fc1"):
                pi_h = tf_util.fully_connected(net, 8, bn=bn, is_training=is_training,
                                  scope="pi_h_fc1", bn_decay=None)
            pi_latent = pi_h
            
            
            with tf.name_scope("vf_h_fc1"):
                vf_h = tf_util.fully_connected(net, 16, bn=bn, is_training=is_training,
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


if __name__=="__main__":
    # Test the network architecture
    batch_size = 2
    num_pt = 1024
    pos_dim = 3
    
    input_feed = np.random.rand(batch_size, num_pt, pos_dim)
    label_feed = np.random.rand(batch_size)
    label_feed[label_feed>=0.5] = 1
    label_feed[label_feed<0.5] = 0
    label_feed = label_feed.astype(np.int32)    
    with tf.Graph().as_default():
      input_pl, label_pl = placeholder_inputs(batch_size, num_pt)
      pos, ftr = get_model(input_pl, tf.constant(True))
      with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        feed_dict = {input_pl: input_feed, label_pl: label_feed}
        res1, res2 = sess.run([pos, ftr], feed_dict=feed_dict)
