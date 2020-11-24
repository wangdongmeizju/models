
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
 
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import argparse
import sys
import shutil
import os 
 
flags = tf.flags
flags.DEFINE_string('f', '', '')
flags.DEFINE_string('data_dir', '/MNIST_data/','dir')
#flags.DEFINE_string('log_out', '/wangdongmei/models/tensorflow/mnist/log_single/','dir')
flags.DEFINE_string('model_save_path', '/wangdongmei/models/logs/log_tf_mnist_single/','dir')
#flags.DEFINE_string('model_save_path', './log_single/','dir')
flags.DEFINE_integer('max_step', 20000, 'step')
flags.DEFINE_integer('batch_size', 64, 'batch')
flags.DEFINE_bool('isresume', False, 'isresume')

FLAGS = flags.FLAGS

model_save_path = FLAGS.model_save_path
isresume = FLAGS.isresume
if not isresume:
  if os.path.exists(model_save_path):
   #  if not os.listdir(model_save_path):  
 #      shutil.rmtree(model_save_path)
      model_path=model_save_path+'/*'
      os.system("rm -rf {}".format(model_path))


def main(_):
    # 获取数据集
    mnist = input_data.read_data_sets(FLAGS.data_dir, one_hot=True)
 
    # 定义输入和输出的占位符
    x = tf.compat.v1.placeholder(tf.float32, [None, 784])
    y_ = tf.compat.v1.placeholder(tf.float32, [None, 10])
 
    # 定义通用函数
    def weight_variable(shape):
        # 截断正态分布 标准方差为0.1
        initial = tf.random.truncated_normal(shape, stddev=0.1)
        return tf.Variable(initial)
 
    def bias_variable(shape):
        # 设为非零避免死神经元
        initial = tf.constant(0.1, shape=shape)
        return tf.Variable(initial)
 
    def conv2d(x, W):
        # 卷积不改变输入的shape
        return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')
 
    def max_pool_2x2(x):
        return tf.nn.max_pool2d(x, ksize=[1, 2, 2, 1],
                              strides=[1, 2, 2, 1], padding='SAME')
 
    sess = tf.compat.v1.InteractiveSession()
 
    # 构建模型
    # 把输入变换成一个4d的张量，第二三个对应的是图片的长和宽，第四个参数对应的颜色
    x_image = tf.reshape(x, [-1, 28, 28, 1])
 
    # 计算32个特征，每5*5patch,第一二个参数指的是patch的size，第三个参数是输入的channels，第四个参数是输出的channels
    W_conv1 = weight_variable([5, 5, 1, 32])
    # 偏差的shape应该和输出的shape一致，所以也是32
    b_conv1 = bias_variable([32])
 
    # 28*28的图片卷积时步长为1，随意卷积后大小不变，按2*2最大值池化，相当于从2*2块中提取一个最大值，
    # 所以池化后大小为[28/2,28/2] = [14,14]，第二次池化后为[14/2,14/2] = [7,7]
 
    # 对数据做卷积操作
    h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
    # max_pool_2x2之后，图片变成14*14
    h_pool1 = max_pool_2x2(h_conv1)
 
    # 在以前的基础上，生成了64个特征
    W_conv2 = weight_variable([5, 5, 32, 64])
    b_conv2 = bias_variable([64])
 
    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
    # max_pool_2x2之后，图片变成7*7
    h_pool2 = max_pool_2x2(h_conv2)
 
    # 构造一个全连接的神经网络，1024个神经元
    W_fc1 = weight_variable([7 * 7 * 64, 1024])
    b_fc1 = bias_variable([1024])
    # 输出为1024
    h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * 64])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
 
    # 做Dropout操作
    keep_prob = tf.compat.v1.placeholder(tf.float32)
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)
 
    W_fc2 = weight_variable([1024, 10])
    b_fc2 = bias_variable([10])
 
    y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2
 
    # 定义损失函数
    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv))
    # 定义优化函数
    train_step = tf.compat.v1.train.AdamOptimizer(1e-4).minimize(cross_entropy)
    # 计算准确率
    correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    # 初始化参数
    sess.run(tf.compat.v1.global_variables_initializer())
    # save the model    
    saver = tf.compat.v1.train.Saver(max_to_keep=5)

    ckpt = tf.train.get_checkpoint_state(FLAGS.model_save_path)
    ckpt_restore = FLAGS.isresume
    if ckpt_restore:
    #ckpt = tf.train.latest_checkpoint(FLAGS.log_out)
       if ckpt and ckpt.model_checkpoint_path:
          saver.restore(sess, ckpt.model_checkpoint_path)
          global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
          print("global_step: %s" %global_step )
          #print("left: %  " %(FLAGS.step_numbers - global_step))

    for i in range(FLAGS.max_step):
        batch = mnist.train.next_batch(FLAGS.batch_size)
        if i % 100 == 0:
            train_accuracy = accuracy.eval(feed_dict={x: batch[0], y_: batch[1], keep_prob: 1.0})
            print("step %d, training accuracy %g" % (i, train_accuracy))
            saver.save(sess, FLAGS.model_save_path+'model', global_step=i)             
        train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})
                  
    print("test accuracy %g" % accuracy.eval(feed_dict={
        x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0}))
 
 
if __name__ == '__main__':
    #parser = argparse.ArgumentParser()
    #parser.add_argument('--data_dir', type=str, default='data/mnist/input_data',
    #                    help='Directory for storing input data')
    #FLAGS, unparsed = parser.parse_known_args()
    #tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
    tf.compat.v1.app.run(main=main)
