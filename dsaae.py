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
                'save_step': 10,

                'model_path': "./models/",
                }

# global data place holder for TF nodes
weights = dict()
infer_layers = dict()
gen_layers = dict()

# Graph definition: input
x = tf.placeholder(tf.float32, [None, net_metadata['input_size']], "x_input")

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
weights['latent_h_u'] = tf.get_variable("latent_h_u",
                                        [net_metadata['infer_layer_2_size'],
                                         net_metadata['latent_z_size']],
                                        tf.float32,
                                        tf.contrib.layers.xavier_initializer())
weights['latent_h_s'] = tf.get_variable("latent_h_s",
                                        [net_metadata['infer_layer_2_size'],
                                         net_metadata['latent_z_size']],
                                        tf.float32,
                                        tf.contrib.layers.xavier_initializer())
weights['latent_b_u'] = tf.get_variable("latent_b_u",
                                        [net_metadata['latent_z_size']],
                                        tf.float32,
                                        tf.zeros_initializer())
weights['latent_b_s'] = tf.get_variable("latent_b_s",
                                        [net_metadata['latent_z_size']],
                                        tf.float32,
                                        tf.zeros_initializer())

# re-parameterization trick: ( z_u, z_log_s2 ) --> z
# Q1: do we need to add activation func here ?
# Q2: do we want to train log_s2, or log_s or s ?
z_u = tf.add(tf.matmul(infer_L2,
                       weights['latent_h_u']),
             weights['latent_b_u'])
z_log_s2 = tf.add(tf.matmul(infer_L2,
                            weights['latent_h_s']),
                  weights['latent_b_s'])
z0 = tf.random_normal((net_metadata['batch_size'],
                       net_metadata['latent_z_size']),
                      0,
                      1,
                      tf.float32)
z = tf.add(z_u, tf.multiply(tf.sqrt(tf.exp(z_log_s2)), z0))

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
gen_L1 = activate(tf.add(tf.matmul(z,
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
recon_loss = -tf.reduce_sum(x * tf.log(tiny + x_out) +
                            (1 - x) * tf.log(tiny + 1 - x_out),
                            1)

# closed form for KL
latent_loss = -0.5 * tf.reduce_sum(1 + z_log_s2 -
                                   tf.square(z_u) -
                                   tf.exp(z_log_s2), 1)

J = tf.reduce_mean(tf.add(recon_loss, latent_loss))

# define optimizer
optimizer = tf.train.AdamOptimizer(net_metadata['learning_rate']).minimize(J)

global_step = tf.contrib.framework.get_or_create_global_step()


# Train
def run(sess, saver, net_metadata, n_samples, mnist, optimizer, J):
    bs = net_metadata['batch_size']
    for epoch in range(net_metadata['training_epoch']):
        avg_cost = 0.0
        total_batch = int(n_samples / bs)
        for i in range(total_batch):
            batch_x, _ = mnist.train.next_batch(bs)
            _, cost = sess.run((optimizer, J), feed_dict={x: batch_x})
            avg_cost += cost / n_samples * bs

        # Display logs per epoch step
        if epoch % net_metadata['save_step'] == 0:
            saver.save(sess,
                       os.path.join(model_dir, 'checkpoint'),
                       global_step=global_step)
            print("Epoch:", '%03d' % (epoch + 1),
                  "avg_cost=", "{:.4f}".format(avg_cost), "time=", datetime.datetime.now())


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
