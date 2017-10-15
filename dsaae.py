import matplotlib.pyplot as plt
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
                'output_function': tf.nn.sigmoid,
                'batch_size': 100,
                'learning_rate': 0.0001,
                'training_epoch': 300,
                'display_step': 10,
                'model_path': "./models",
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
                                        tf.zeros_initializer())
weights['latent_h_s'] = tf.get_variable("latent_h_s",
                                        [net_metadata['infer_layer_2_size'],
                                         net_metadata['latent_z_size']],
                                        tf.float32,
                                        tf.zeros_initializer())
weights['latent_b_u'] = tf.get_variable("latent_b_u",
                                        [net_metadata['latent_z_size']],
                                        tf.float32,
                                        tf.zeros_initializer())
weights['latent_b_s'] = tf.get_variable("latent_b_s",
                                        [net_metadata['latent_z_size']],
                                        tf.float32,
                                        tf.zeros_initializer())

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

gen_L1 = activate(tf.add(tf.matmul(z,
                                   weights['gen_h1']),
                         weights['gen_b1']))
gen_L2 = activate(tf.add(tf.matmul(gen_L1,
                                   weights['gen_h2']),
                         weights['gen_b2']))
x_out = activate(tf.add(tf.matmul(gen_L2,
                                  weights['gen_h_out']),
                        weights['gen_b_out']))

# define loss
recon_loss = -tf.reduce_sum(x * tf.log(1e-10 + x_out) +
                            (1 - x) * tf.log(1e-10 + 1 - x_out),
                            1)

latent_loss = -0.5 * tf.reduce_sum(1 + z_log_s2 -
                                   tf.square(z_u) -
                                   tf.exp(z_log_s2), 1)

J = tf.reduce_mean(tf.add(recon_loss, latent_loss))

# define optimizer
optimizer = tf.train.AdamOptimizer(net_metadata['learning_rate']).minimize(J)


# Train
def run(sess, saver, net_metadata, n_samples, mnist, optimizer, J):
    bs = net_metadata['batch_size']
    for epoch in range(net_metadata['training_epoch']):
        avg_cost = 0.0
        total_batch = int(n_samples / bs)
        for i in range(total_batch):
            batch_x, _label = mnist.train.next_batch(bs)
            _opt, cost = sess.run((optimizer, J), feed_dict={x: batch_x})
            avg_cost += cost / n_samples * bs

        # Display logs per epoch step
        if epoch % net_metadata['display_step'] == 0:
            saver.save(sess, net_metadata['model_path'])
            print("Epoch:", '%04d' % (epoch + 1),
                  "cost=", "{:.9f}".format(avg_cost))

    sess.close()


# Main body
init = tf.global_variables_initializer()
sess = tf.InteractiveSession()
saver = tf.train.Saver()

mode = "restart"

if mode == "restart":
    print("start init ...\n")
    sess.run(init)
    run(sess, saver, net_metadata, n_samples, mnist, optimizer, J)

if mode == "continue":
    print("continue with saved model .. \n")
    model_file = tf.train.latest_checkpoint(net_metadata['model_path'])
    saver.restore(sess, model_file)
    run(sess, saver, net_metadata, n_samples, mnist, optimizer, J)

if mode == "testing":
    print("testing saved model .. \n")
    model_file = tf.train.latest_checkpoint(net_metadata['model_path'])
    saver.restore(sess, model_file)
    batch_x, _label = mnist.train.next_batch(net_metadata['batch_size'])
    x_recon = sess.run(x_out, feed_dict={x: batch_x})

    plt.figure(figsize=(8, 12))
    for i in range(5):
        plt.subplot(5, 2, 2 * i + 1)
        plt.imshow(batch_x[i].reshape(28, 28), vmin=0, vmax=1, cmap="gray")
        plt.title("Test input")
        plt.colorbar()
        plt.subplot(5, 2, 2 * i + 2)
        plt.imshow(x_recon[i].reshape(28, 28), vmin=0, vmax=1, cmap="gray")
        plt.title("Reconstruction")
        plt.colorbar()
    plt.tight_layout()
    plt.show()
