from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf 
import numpy as np
import os 
import time

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0, 1"

flags = tf.flags
flags.DEFINE_string('f', '', '')
flags.DEFINE_string('data_dir', '/imagenet/ILSVRC2012_img_train_tfrecord','dir')
flags.DEFINE_string('log_out', '/inspur/models/tensorflow/log_imagenet/','dir')
flags.DEFINE_string('model_save_path', '/inspur/models/tensorflow/log_imagenet/','dir')
flags.DEFINE_integer('epoch', 2000, 'epoch')
flags.DEFINE_integer('batch_size', 64, 'batch')
flags.DEFINE_integer('gpu_num', 1, 'gpu')
FLAGS = flags.FLAGS

NUM_CLASSES = 1000
BATCH_SIZE = FLAGS.batch_size
EPOCHS = FLAGS.epoch
NUM_GPUS = FLAGS.gpu_num
TRAIN_SIZE = 1281167
'''
local meanstd = {
   mean = { 0.485, 0.456, 0.406 },
   std = { 0.229, 0.224, 0.225 },
}
'''

import shutil
model_dir = FLAGS.model_save_path
shutil.rmtree(model_dir)
os.makedirs(model_dir)
####### input pipeline 
def input_fn():
  train_files_names = os.listdir(FLAGS.data_dir)
  train_files = [FLAGS.data_dir+item for item in train_files_names[:20]]
  dataset_train = tf.data.TFRecordDataset(train_files, buffer_size=2048, num_parallel_reads=128)

  def _parse_data(example_proto):
    features = {'label': tf.FixedLenFeature([], tf.int64),
                'img_raw': tf.FixedLenFeature([], tf.string)}
    parsed_features = tf.parse_single_example(example_proto, features)
    image = tf.decode_raw(parsed_features['img_raw'], tf.uint8)
    label = tf.cast(parsed_features['label'], tf.int64)  
    image = tf.reshape(image, [224, 224, 3])
    image = tf.cast(image, tf.float32)/255
    return image, label

  dataset_train = dataset_train.repeat(FLAGS.epoch)
  dataset_train = dataset_train.shuffle(buffer_size=1024)
  dataset_train = dataset_train.map(_parse_data, num_parallel_calls=30)
  dataset_train = dataset_train.batch(BATCH_SIZE)
  dataset_train = dataset_train.prefetch(2)
  return dataset_train


class TimeHistory(tf.train.SessionRunHook):
  def begin(self):
    self.times = []
  def before_run(self, run_context):
    self.iter_time_start = time.time()
  def after_run(self, run_context, run_values):
    self.times.append(time.time() - self.iter_time_start)


def main():
  tf.logging.set_verbosity(tf.logging.INFO)
  ### classifier
  model = tf.keras.applications.resnet50.ResNet50(weights=None)
  model.summary()

  optimizer = tf.train.AdamOptimizer(learning_rate=0.01)
  model.compile(loss="sparse_categorical_crossentropy" , optimizer=optimizer,metrics=["accuracy"] )

  time_hist = TimeHistory()

  if NUM_GPUS==1: strategy = tf.contrib.distribute.OneDeviceStrategy(device='/GPU:0')
  else: strategy = tf.contrib.distribute.MirroredStrategy(num_gpus=NUM_GPUS)

  session_config = tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True))
  config = tf.estimator.RunConfig(model_dir=model_dir,
                                  train_distribute=strategy,save_checkpoints_steps=5000)#, session_config = session_config)                                 # distributed mode, fixed batch_size and fixed memory using
#                                  train_distribute=None, session_config = session_config, save_checkpoints_steps=5000)   # single mode, floating batch_size and memory using
#                                  train_distribute=None,save_checkpoints_steps=5000)                                      # single mode, fixed batch_size and fixed memory using

  estimator = tf.keras.estimator.model_to_estimator(model,config=config)
  estimator.train(input_fn = input_fn , hooks=[time_hist])

  ####################################
  total_time = sum(time_hist.times)
  print(f"total time with {NUM_GPUS} GPU(s): {total_time} seconds")
  avg_time_per_batch = np.mean(time_hist.times)
  print(f"{BATCH_SIZE*NUM_GPUS/avg_time_per_batch} images/second with {NUM_GPUS} GPU(s)")

if __name__ == '__main__':
  main()
