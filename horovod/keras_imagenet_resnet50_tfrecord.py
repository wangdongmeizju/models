# ResNet-50 model training using Keras and Horovod.
#
# This model is an example of a computation-intensive model that achieves good accuracy on an image
# classification task.  It brings together distributed training concepts such as learning rate
# schedule adjustments with a warmup, randomized data reading, and checkpointing on the first worker
# only.
#
# Note: This model uses Keras native ImageDataGenerator and not the sophisticated preprocessing
# pipeline that is typically used to train state-of-the-art ResNet-50 model.  This results in ~0.5%
# increase in the top-1 validation error compared to the single-crop top-1 validation error from
# https://github.com/KaimingHe/deep-residual-networks.

from __future__ import print_function

import argparse
#import keras
from tensorflow import keras
from keras import backend as K
from keras.preprocessing import image
import tensorflow as tf
import horovod.keras as hvd
import os
from tensorflow.python.client import timeline

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0"

parser = argparse.ArgumentParser(description='Keras ImageNet Example',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--train-dir', default=os.path.expanduser('/home/imagenet/ILSVRC2012_img_train_tfrecord/'),
                    help='path to training data')
parser.add_argument('--val-dir', default=os.path.expanduser('/home/imagenet/ILSVRC2012_img_val_tfrecord/'),
                    help='path to validation data')
parser.add_argument('--train', default=os.path.expanduser('/home/imagenet/ILSVRC2012_img_train/'),
                    help='path to training data')
parser.add_argument('--val', default=os.path.expanduser('/home/imagenet/ILSVRC2012_img_val_raw/'),
                    help='path to validation data')
parser.add_argument('--log-dir', default='./logs',
                    help='tensorboard log directory')
parser.add_argument('--checkpoint-format', default='./checkpoint-{epoch}.h5',
                    help='checkpoint file format')
parser.add_argument('--fp16-allreduce', action='store_true', default=True,
                    help='use fp16 compression during allreduce')

# Default settings from https://arxiv.org/abs/1706.02677.
parser.add_argument('--batch-size', type=int, default=32,
                    help='input batch size for training')
parser.add_argument('--val-batch-size', type=int, default=32,
                    help='input batch size for validation')
parser.add_argument('--epochs', type=int, default=4,
                    help='number of epochs to train')
parser.add_argument('--base-lr', type=float, default=0.0125,
                    help='learning rate for a single GPU')
parser.add_argument('--warmup-epochs', type=float, default=5,
                    help='number of warmup epochs')
parser.add_argument('--momentum', type=float, default=0.9,
                    help='SGD momentum')
parser.add_argument('--wd', type=float, default=0.00005,
                    help='weight decay')

args = parser.parse_args()

# Training data iterator for tfrecord
def tfrecord_loader(datasetname):
    train_files_names = os.listdir(datasetname)
    train_files = [os.path.join(datasetname, item) for item in train_files_names[:100]]
    dataset_train = tf.data.TFRecordDataset(train_files, buffer_size=2048, num_parallel_reads=128)

    def _parse_data(example_proto):
        features = {'label': tf.io.FixedLenFeature([], tf.int64),
                    'img_raw': tf.io.FixedLenFeature([], tf.string)}
        parsed_features = tf.io.parse_single_example(example_proto, features)
        image = tf.decode_raw(parsed_features['img_raw'], tf.uint8)
        label = tf.cast(parsed_features['label'], tf.int64)
        label = tf.one_hot(label, depth=1000)
        image = tf.reshape(image, [224, 224, 3])
        image = tf.cast(image, tf.float32) / 255
        return image, label

    dataset_train = dataset_train.repeat()
    dataset_train = dataset_train.shuffle(buffer_size=1024)
    dataset_train = dataset_train.map(_parse_data, num_parallel_calls=30)
    dataset_train = dataset_train.batch(args.batch_size)
    dataset_train = dataset_train.prefetch(2)
    return dataset_train

def iterator(datasetname):
    dataset = tfrecord_loader(datasetname)
    #iterator = dataset.make_one_shot_iterator()
    iterator = tf.compat.v1.data.make_one_shot_iterator(dataset)
    batch=iterator.get_next()
    K.batch_get_value(batch)
    return iterator


def main():
  # Horovod: initialize Horovod.
  hvd.init()
  
  # Horovod: pin GPU to be used to process local rank (one GPU per process)
  config = tf.compat.v1.ConfigProto(allow_soft_placement=True)
  config.graph_options.optimizer_options.global_jit_level = tf.compat.v1.OptimizerOptions.ON_1
  config.gpu_options.allow_growth = True
  config.gpu_options.visible_device_list = str(hvd.local_rank())
  K.set_session(tf.compat.v1.Session(config=config))
  
  # If set > 0, will resume training from a given checkpoint.
  resume_from_epoch = 0
  for try_epoch in range(args.epochs, 0, -1):
      if os.path.exists(args.checkpoint_format.format(epoch=try_epoch)):
          resume_from_epoch = try_epoch
          break
  
  # Horovod: broadcast resume_from_epoch from rank 0 (which will have
  # checkpoints) to other ranks.
  resume_from_epoch = hvd.broadcast(resume_from_epoch, 0, name='resume_from_epoch')
  
  # Horovod: print logs on the first worker.
  verbose = 1 if hvd.rank() == 0 else 0
  
  # Training data iterator.
  train_gen = image.ImageDataGenerator()
      #width_shift_range=0.33, height_shift_range=0.33, zoom_range=0.5, horizontal_flip=True,
      #preprocessing_function=keras.applications.resnet50.preprocess_input)
  train_iter = train_gen.flow_from_directory(args.train,
                                             batch_size=args.batch_size,
                                             target_size=(224, 224))
  
  # Validation data iterator.
  test_gen = image.ImageDataGenerator()
      #zoom_range=(0.875, 0.875), preprocessing_function=keras.applications.resnet50.preprocess_input)
  test_iter = test_gen.flow_from_directory(args.val,
                                           batch_size=args.val_batch_size,
                                           target_size=(224, 224))
  
  # train iterator for tfrecord 
  train_iter_tf = iterator(args.train_dir)
  val_iter_tf   = iterator(args.val_dir)
  
  # timeline
  #timeline = tf.train.ProfilerHook(save_steps=500, output_dir='./timeline')
  #run_options  = tf.compat.v1.RunOptions(trace_level=tf.compat.v1.RunOptions.FULL_TRACE)
  #run_metadata = tf.compat.v1.RunMetadata()
  
  # Set up standard ResNet-50 model.
  model = keras.applications.resnet50.ResNet50(weights=None)
  
  # Horovod: (optional) compression algorithm.
  compression = hvd.Compression.fp16 if args.fp16_allreduce else hvd.Compression.none
  
  # Restore from a previous checkpoint, if initial_epoch is specified.
  # Horovod: restore on the first worker which will broadcast both model and optimizer weights
  # to other workers.
  if resume_from_epoch > 0 and hvd.rank() == 0:
      model = hvd.load_model(args.checkpoint_format.format(epoch=resume_from_epoch),
                             compression=compression)
  else:
      # ResNet-50 model that is included with Keras is optimized for inference.
      # Add L2 weight decay & adjust BN settings.
      model_config = model.get_config()
      for layer, layer_config in zip(model.layers, model_config['layers']):
          if hasattr(layer, 'kernel_regularizer'):
              regularizer = keras.regularizers.l2(args.wd)
              layer_config['config']['kernel_regularizer'] = \
                  {'class_name': regularizer.__class__.__name__,
                   'config': regularizer.get_config()}
          if type(layer) == keras.layers.BatchNormalization:
              layer_config['config']['momentum'] = 0.9
              layer_config['config']['epsilon'] = 1e-5
  
      model = keras.models.Model.from_config(model_config)
  
      # Horovod: adjust learning rate based on number of GPUs.
      opt = keras.optimizers.SGD(lr=args.base_lr * hvd.size(),
                                 momentum=args.momentum)
  
      # Horovod: add Horovod Distributed Optimizer.
      opt = hvd.DistributedOptimizer(opt, compression=compression)
  
      model.compile(loss=keras.losses.categorical_crossentropy,
                    optimizer=opt,
                    metrics=['accuracy', 'top_k_categorical_accuracy'])
      #              options=run_options,
      #              run_metadata=run_metadata
      #              )
  
  callbacks = [
      # Horovod: broadcast initial variable states from rank 0 to all other processes.
      # This is necessary to ensure consistent initialization of all workers when
      # training is started with random weights or restored from a checkpoint.
      hvd.callbacks.BroadcastGlobalVariablesCallback(0),
  
      # Horovod: average metrics among workers at the end of every epoch.
      #
      # Note: This callback must be in the list before the ReduceLROnPlateau,
      # TensorBoard, or other metrics-based callbacks.
      hvd.callbacks.MetricAverageCallback(),
  
      # Horovod: using `lr = 1.0 * hvd.size()` from the very beginning leads to worse final
      # accuracy. Scale the learning rate `lr = 1.0` ---> `lr = 1.0 * hvd.size()` during
      # the first five epochs. See https://arxiv.org/abs/1706.02677 for details.
      hvd.callbacks.LearningRateWarmupCallback(warmup_epochs=args.warmup_epochs, verbose=verbose),
  
      # Horovod: after the warmup reduce learning rate by 10 on the 30th, 60th and 80th epochs.
      hvd.callbacks.LearningRateScheduleCallback(start_epoch=args.warmup_epochs, end_epoch=30, multiplier=1.),
      hvd.callbacks.LearningRateScheduleCallback(start_epoch=30, end_epoch=60, multiplier=1e-1),
      hvd.callbacks.LearningRateScheduleCallback(start_epoch=60, end_epoch=80, multiplier=1e-2),
      hvd.callbacks.LearningRateScheduleCallback(start_epoch=80, multiplier=1e-3),
  ]
  
  # Horovod: save checkpoints only on the first worker to prevent other workers from corrupting them.
  if hvd.rank() == 0:
      callbacks.append(keras.callbacks.ModelCheckpoint(args.checkpoint_format))
      callbacks.append(keras.callbacks.TensorBoard(args.log_dir))
  
  # Train the model. The training will randomly sample 1 / N batches of training data and
  # 3 / N batches of validation data on every worker, where N is the number of workers.
  # Over-sampling of validation data helps to increase probability that every validation
  # example will be evaluated.

  print('----  train  len------ :' ,  len(train_iter))
  print('----  test   len------ :' ,  len(test_iter)) 
  total_train_step = len(train_iter)
  total_val_step = len(test_iter)
 
  #model.fit_generator(train_iter,
  model.fit(train_iter_tf,
                      #steps_per_epoch=40037 // hvd.size(),
                      steps_per_epoch= total_train_step  // hvd.size(),
                      callbacks=callbacks,
                      epochs=args.epochs,
                      verbose=verbose,
                      workers=8,
                      initial_epoch=resume_from_epoch,
                      validation_data=val_iter_tf,
                      validation_steps=3 * total_val_step // hvd.size())
  
  # timeline tracing
  #trace = timeline.Timeline(step_stats=run_metadata.step_stats)
  #with open ('./timeline.keras.json','w') as f:
  #     f.write(trace.generate_chrome_trace_format())
   
  # Evaluate the model on the full data set.
  score = hvd.allreduce(model.evaluate_generator(test_iter, len(test_iter), workers=4))
  if verbose:
      print('Test loss:', score[0])
      print('Test accuracy:', score[1])

if __name__ == '__main__':
     main()
