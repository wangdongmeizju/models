from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json
import os
import argparse
from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf

FLAGS = None

def weight_variable(shape):
    init = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(init)


def bias_variable(shape):
    init = tf.constant(0.1, shape=shape)
    return tf.Variable(init)


def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding="SAME")


def max_pool(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")


def main(_):

    tf_config = json.loads(os.environ.get('TF_CONFIG') or '{}')
    task_config = tf_config.get('task', {})
    task_type = task_config.get('type')
    task_index = task_config.get('index')

    FLAGS.job_name = task_type
    print('job_name:%s' %(task_type))
    FLAGS.task_index = task_index

    #ps_hosts = FLAGS.ps_hosts.split(",")
    #worker_hosts = FLAGS.worker_hosts.split(",")

    cluster_config = tf_config.get('cluster', {})
    ps_hosts = cluster_config.get('ps')
    worker_hosts = cluster_config.get('worker')
    
    ps_hosts_str = ','.join(ps_hosts)
    worker_hosts_str = ','.join(worker_hosts)
    
    FLAGS.ps_hosts = ps_hosts_str
    FLAGS.worker_hosts = worker_hosts_str
    
    # Construct the cluster and start the server
    ps_spec = FLAGS.ps_hosts.split(",")
    worker_spec = FLAGS.worker_hosts.split(",")
    
    # Get the number of workers.
    num_workers = len(worker_spec)


    # Create a cluster from the parameter server and worker hosts.
    #cluster = tf.train.ClusterSpec({"ps": ps_hosts, "worker": worker_hosts})
    cluster = tf.train.ClusterSpec({"ps": ps_spec, "worker": worker_spec})

    # Create and start a server for the local task.
    server = tf.train.Server(cluster,
                             job_name=FLAGS.job_name,
                             task_index=FLAGS.task_index,
                             start=True)

    if FLAGS.job_name == "ps":
        server.join()
    elif FLAGS.job_name == "worker":
        # Import data
        mnist = input_data.read_data_sets(FLAGS.data_dir, one_hot=True)
        
        FLAGS.batch_size = 550
        step = mnist.train.num_examples / 500 - 1
        print("train examples: %d, step: %d" % (mnist.train.num_examples, step) )
        
        

        with tf.device(tf.train.replica_device_setter(
                #worker_device="/job:worker/task:%d" % FLAGS.task_index,
                worker_device="/job:worker/task:%d/gpu:%d" % (FLAGS.task_index, 0),
                cluster=cluster)):

            x = tf.placeholder(tf.float32, [None, 784])
            y_actual = tf.placeholder(tf.float32, [None, 10])
            keep_prob = tf.placeholder(tf.float32)


            x_image = tf.reshape(x, [-1, 28, 28, 1])

            W_conv1 = weight_variable([5, 5, 1, 32])
            b_conv1 = bias_variable([32])
            h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)  
            h_pool1 = max_pool(h_conv1) 

            W_conv2 = weight_variable([5, 5, 32, 64])
            b_conv2 = bias_variable([64])
            h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)  
            h_pool2 = max_pool(h_conv2)  

            W_fc1 = weight_variable([7 * 7 * 64, 1024])
            b_fc1 = bias_variable([1024])
            h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * 64])  # reshape
            h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)  

            # dropout
            h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)  # dropout

            W_fc2 = weight_variable([1024, 10])
            b_fc2 = bias_variable([10])
            y_predict = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)  # softmax, [-1, 10]

            cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_actual * tf.log(y_predict), 1))  
            global_step = tf.train.get_or_create_global_step()
            #global_step = tf.train.get_global_step()
            optimizer = tf.train.GradientDescentOptimizer(FLAGS.learning_rate)
            train_op = optimizer.minimize(cross_entropy, global_step=global_step)

            cross_prediction = tf.equal(tf.argmax(y_predict, 1), tf.argmax(y_actual, 1))
            accuracy = tf.reduce_mean(tf.cast(cross_prediction, tf.float32))  

            # tensorboard
            tf.summary.scalar('cost', cross_entropy)
            tf.summary.scalar("accuracy", accuracy)
            summary_op = tf.summary.merge_all()

            # The StopAtStepHook handles stopping after running given steps.
            #hooks = [tf.train.StopAtStepHook(last_step=400)]
            hooks = [tf.train.StopAtStepHook(last_step=step)]

            config = tf.ConfigProto(
                allow_soft_placement=True,
                log_device_placement=False,
                #device_filters=["/job:ps", "/job:worker/task:%d" % FLAGS.task_index]
                device_filters=["/job:ps", "/job:worker/task:%d/gpu:%d" % (FLAGS.task_index, 0)]
            )

            # The MonitoredTrainingSession takes care of session initialization,
            # restoring from a checkpoint, saving to a checkpoint, and closing when done
            # or an error occurs.
            # master="grpc://" + worker_hosts[FLAGS.task_index]
            
            with tf.train.MonitoredTrainingSession(master=server.target,
                                                   config=config,
                                                   is_chief=(FLAGS.task_index == 0),
                                                   hooks=hooks,
                               max_wait_secs = 120) as mon_sess:
                while not mon_sess.should_stop():
                  
                    # Run a training step asynchronously.
                    # See `tf.train.SyncReplicasOptimizer` for additional details on how to
                    # perform *synchronous* training.
                    # mon_sess.run handles AbortedError in case of preempted PS.
                    #batch_x, batch_y = mnist.train.next_batch(64)
                    batch_x, batch_y = mnist.train.next_batch(FLAGS.batch_size)
#                    step, _ = mon_sess.run([global_step, train_op], feed_dict={
#                        x: batch_x,
#                        y_actual: batch_y,
#                        keep_prob: 0.8})
                    

                    #print("global_step: %f" % step)
                    #if step > 0 and step % 10 == 0:
                    step, _, loss, acc = mon_sess.run([global_step, train_op, cross_entropy, accuracy], feed_dict={
                        x: batch_x,
                        y_actual: batch_y,
                        keep_prob: 1.0})
                    print("step: %d, loss: %f, acc: %f" % (step, loss, acc))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.register("type", "bool", lambda v: v.lower() == "true")
    # Flags for defining the tf.train.ClusterSpec

    parser.add_argument(
        "--data_dir",
        type=str,
        default="/MNIST_data/",
        help="data directory"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=600,
        help="batch size"
    )
    parser.add_argument(
        "--max_step",
        type=int,
        default=100,
        help="step num"
    )

    parser.add_argument(
        "--ps_hosts",
        type=str,
        default="",
        help="Comma-separated list of hostname:port pairs"
    )
    parser.add_argument(
        "--worker_hosts",
        type=str,
        default="",
        help="Comma-separated list of hostname:port pairs"
    )
    parser.add_argument(
        "--job_name",
        type=str,
        default="",
        help="One of 'ps', 'worker'"
    )
    # Flags for defining the tf.train.Server
    parser.add_argument(
        "--task_index",
        type=int,
        default=0,
        help="Index of task within the job"
    )

    parser.add_argument(
#        "--checkpoint_dir",
        "--model_save_path",
        type=str,
        default="/wangdongmei/models/logs/log_tf_mnist_dist/",
        help="path to a directory where to restore variables."
    )

    parser.add_argument(
        "--learning_rate",
        type=float,
        default=0.001,
        help="learning rate"
    )

    FLAGS, _ = parser.parse_known_args()
    tf.app.run(main=main)
