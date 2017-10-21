from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import datetime
import os

import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

# random data init
np.random.seed(0)
tf.set_random_seed(0)


def parzen_density(test_data, gen_data, sigma, test_data_size, gen_data_size, data_dim_size):
    x = test_data.reshape((test_data_size, 1, data_dim_size))
    mu = gen_data.reshape((1, gen_data_size, data_dim_size))
    a = (x - mu) / sigma  # ( test_data_size, gen_data_size, data_size)
    t = -0.5 * (a ** 2).sum(2)  # ( test_size, gen_data_size)

    m = np.amax(t, axis=1, keepdims=True)
    E = m + np.log(np.mean(np.exp(t - m), axis=1, keepdims=True))

    Z = data_dim_size * np.log(sigma * np.sqrt(np.pi * 2))

    return E - Z


def computeScaleMatrix(size1, size2):
    s1 = tf.constant(1.0 / size1, shape=[size1, 1])
    s2 = -tf.constant(1.0 / size2, shape=[size2, 1])

    return tf.concat([s1, s2], 0)


# general MMD calculation
def computeMMDLossSqrt(x, y, xCount, yCount, sigma=[2, 5, 10, 20, 40, 80]):
    M = tf.concat([x, y], 0)
    MM = tf.matmul(M, tf.transpose(M))
    M2 = tf.reduce_sum(M * M, 1, keep_dims=True)
    exponent = MM - 0.5 * M2 - 0.5 * tf.transpose(M2)
    scale = computeScaleMatrix(xCount, yCount)
    S = tf.matmul(scale, tf.transpose(scale))

    loss = 0
    for i in range(len(sigma)):
        kernel_val = tf.exp(1.0 / sigma[i] * exponent)
        loss += tf.reduce_sum(S * kernel_val)

    return tf.sqrt(loss)


# download MNIST
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
n_samples = mnist.train.num_examples
print("total train samples = ", n_samples, "\n")

# network meta data
MNIST_DIM = 28
net_metadata = {'input_size': MNIST_DIM * MNIST_DIM,

                'infer_layer_1_size': 500,
                'infer_layer_2_size': 500,
                'latent_z_size': 100,
                'gen_layer_1_size': 500,
                'gen_layer_2_size': 500,

                'action_function': tf.nn.relu,
                'output_function': tf.nn.sigmoid,  # final output layer, has to be bounced
                'eps': 1e-10,

                'batch_size': 100,
                'learning_rate': 0.0001,
                'training_epoch': 500,
                'save_step': 20,

                'beta_mmd': 10.0,

                'model_path': "./models.mmd/",
                }

# global data place holder for TF nodes
weights = dict()
infer_layers = dict()
gen_layers = dict()

# Graph definition: input
x = tf.placeholder(tf.float32, [None, net_metadata['input_size']], "x_input")
z = tf.placeholder(tf.float32, [None, net_metadata['latent_z_size']], "z_input")

# Graph definition: inference layer 1 & 2
weights['infer_h1'] = tf.get_variable("infer_h1",
                                      [net_metadata['input_size'],
                                       net_metadata['infer_layer_1_size']],
                                      tf.float32,
                                      tf.contrib.layers.xavier_initializer())
weights['infer_h2'] = tf.get_variable("infer_h2",
                                      [net_metadata['infer_layer_1_size'],
                                       net_metadata['infer_layer_2_size']],
                                      tf.float32,
                                      tf.contrib.layers.xavier_initializer())
weights['infer_b1'] = tf.get_variable("infer_b1",
                                      [net_metadata['infer_layer_1_size']],
                                      tf.float32,
                                      tf.zeros_initializer())
weights['infer_b2'] = tf.get_variable("infer_b2",
                                      [net_metadata['infer_layer_2_size']],
                                      tf.float32,
                                      tf.zeros_initializer())

# x --> gen_L1 --> gen_L2 --> x_out --> ( z_u, z_log_s2 )
activate = net_metadata['action_function']
infer_L1 = activate(tf.add(tf.matmul(x,
                                     weights['infer_h1']),
                           weights['infer_b1']))
infer_L2 = activate(tf.add(tf.matmul(infer_L1,
                                     weights['infer_h2']),
                           weights['infer_b2']))

# define ( VAE core ) latent layer
weights['latent_h'] = tf.get_variable("latent_h",
                                      [net_metadata['infer_layer_2_size'],
                                       net_metadata['latent_z_size']],
                                      tf.float32,
                                      tf.contrib.layers.xavier_initializer())
weights['latent_b'] = tf.get_variable("latent_b",
                                      [net_metadata['latent_z_size']],
                                      tf.float32,
                                      tf.zeros_initializer())

latent = tf.add(tf.matmul(infer_L2,
                          weights['latent_h']),
                weights['latent_b'])

# need to make sure latent and x have same number of size
mmd_loss = computeMMDLossSqrt(latent, z, net_metadata['batch_size'], net_metadata['batch_size'])

# Graph definition: generative layer 1 & 2 and Output
weights['gen_h1'] = tf.get_variable("gen_h1",
                                    [net_metadata['latent_z_size'],
                                     net_metadata['gen_layer_1_size']],
                                    tf.float32,
                                    tf.contrib.layers.xavier_initializer())
weights['gen_h2'] = tf.get_variable("gen_h2",
                                    [net_metadata['gen_layer_1_size'],
                                     net_metadata['gen_layer_2_size']],
                                    tf.float32,
                                    tf.contrib.layers.xavier_initializer())
weights['gen_h_out'] = tf.get_variable("gen_h_out",
                                       [net_metadata['gen_layer_2_size'],
                                        net_metadata['input_size']],
                                       tf.float32,
                                       tf.contrib.layers.xavier_initializer())

weights['gen_b1'] = tf.get_variable("gen_b1",
                                    [net_metadata['gen_layer_1_size']],
                                    tf.float32,
                                    tf.zeros_initializer())
weights['gen_b2'] = tf.get_variable("gen_b2",
                                    [net_metadata['gen_layer_2_size']],
                                    tf.float32,
                                    tf.zeros_initializer())
weights['gen_b_out'] = tf.get_variable("gen_b_out",
                                       [net_metadata['input_size']],
                                       tf.float32,
                                       tf.zeros_initializer())

# z --> gen_L1 --> gen_L2 --> x_out
gen_L1 = activate(tf.add(tf.matmul(latent,
                                   weights['gen_h1']),
                         weights['gen_b1']))
gen_L2 = activate(tf.add(tf.matmul(gen_L1,
                                   weights['gen_h2']),
                         weights['gen_b2']))

output = net_metadata['output_function']
x_out = output(tf.add(tf.matmul(gen_L2,
                                weights['gen_h_out']),
                      weights['gen_b_out']))

# define loss
tiny = net_metadata['eps']
ce = -tf.reduce_sum(x * tf.log(tiny + x_out) +
                    (1 - x) * tf.log(tiny + 1 - x_out),
                    1)
recon_loss = tf.reduce_mean(ce)

# MMD loss
latent_loss = net_metadata['beta_mmd'] * mmd_loss
J = recon_loss + latent_loss

# define optimizer
optimizer = tf.train.AdamOptimizer(net_metadata['learning_rate']).minimize(J)
global_step = tf.contrib.framework.get_or_create_global_step()


# Train
def run(sess, saver, net_metadata, n_samples, mnist, optimizer, J):
    bs = net_metadata['batch_size']
    for epoch in range(net_metadata['training_epoch']):
        avg_cost = 0.0
        avg_L = 0.0
        avg_R = 0.0
        total_batch = int(n_samples / bs)
        for i in range(total_batch):
            batch_x, _ = mnist.train.next_batch(bs)
            batch_z = np.random.uniform(low=-1.0, high=1.0,
                                        size=(bs, net_metadata['latent_z_size']))

            _, cost, L_, R_ = sess.run((optimizer, J, latent_loss, recon_loss), feed_dict={x: batch_x, z: batch_z})

            avg_cost += cost / n_samples * bs
            avg_L += L_ / n_samples * bs
            avg_R += R_ / n_samples * bs

        # Display logs per epoch step
        if epoch % net_metadata['save_step'] == 0:
            saver.save(sess,
                       os.path.join(model_dir, 'checkpoint'),
                       global_step=global_step)
            print("Epoch:",
                  '%03d' % (epoch + 1),
                  "avg_cost=", "{:.1f}".format(avg_cost),
                  "avg_L=", "{:.1f}".format(avg_L),
                  "avg_R=", "{:.1f}".format(avg_R),
                  "time=", datetime.datetime.now())


# Main body
model_dir = net_metadata['model_path']
if not os.path.exists(model_dir):
    os.makedirs(model_dir)

saver = tf.train.Saver(max_to_keep=1)
model_file = tf.train.latest_checkpoint(model_dir)
print("model file = %s" % model_file)

mode = "train"
print("run as %s .. " % mode)

with tf.Session() as sess:
    if mode == "train":
        if model_file:
            saver.restore(sess, model_file)
            print("restored from %s" % model_file)
        else:
            init = tf.global_variables_initializer()
            sess.run(init)
            print("start init ...\n")

        run(sess, saver, net_metadata, n_samples, mnist, optimizer, J)

    elif mode == "eval":
        saver.restore(sess, model_file)
        print("restored from %s" % model_file)

        batch_x, _label = mnist.test.next_batch(net_metadata['batch_size'])
        cost = sess.run(J, feed_dict={x: batch_x})
        print("test cost = %f" % cost)

    sess.close()
