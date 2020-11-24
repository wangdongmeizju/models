# Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserved
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import print_function


import requests
import hashlib
import os
import errno
import shutil
import six
import sys
import importlib
import paddle.dataset
import six.moves.cPickle as pickle
from six.moves import range
import glob
import numpy 
import gzip
import struct
import paddle as paddle
import paddle.fluid as fluid
import argparse

__all__ = [
    'DATA_HOME',
    'download',
    'md5file',
    'split',
    'cluster_files_reader',
    'convert',
]

parser = argparse.ArgumentParser(description='Paddle MNIST Example')
parser.add_argument('--data_dir', default='/MNIST_data/', metavar='L', help='dataset path, dataset name should be "mnist"')
parser.add_argument('--model_save_path', default='/wangdongmei/models/logs/log_paddle_mnist_single/', metavar='L', help='directory where summary logs are stored')

parser.add_argument('--GPU', default=True, metavar='L', help='directory where summary logs are stored')
parser.add_argument('--isresume', default=False, metavar='L', help='resumimg the training from the checkpoint')

args =parser.parse_args()
#args = parser.parse_args([])

# GPU using
use_GPU = args.GPU 
#DATA_HOME = os.path.expanduser('./dataset')
DATA_HOME = os.path.expanduser(args.data_dir)
MODEL_DIR = os.path.expanduser(args.model_save_path)

model_save_path = args.model_save_path
isresume = args.isresume
if not isresume:
  if os.path.exists(model_save_path):
   #  if not os.listdir(model_save_path):  
      #shutil.rmtree(model_save_path)
      model_path=model_save_path+'/*'
      os.system("rm -rf {}".format(model_path))


if not os.path.isdir(model_save_path):
        os.mkdir(model_save_path)

# When running unit tests, there could be multiple processes that
# trying to create DATA_HOME directory simultaneously, so we cannot
# use a if condition to check for the existence of the directory;
# instead, we use the filesystem as the synchronization mechanism by
# catching returned errors.
def must_mkdirs(path):
    try:
        os.makedirs(DATA_HOME)
    except OSError as exc:
        if exc.errno != errno.EEXIST:
            raise
        pass

must_mkdirs(DATA_HOME)


def md5file(fname):
    hash_md5 = hashlib.md5()
    f = open(fname, "rb")
    for chunk in iter(lambda: f.read(4096), b""):
        hash_md5.update(chunk)
    f.close()
    return hash_md5.hexdigest()


def download(url, module_name, md5sum, save_name=None):
    dirname = os.path.join(DATA_HOME, module_name)
    if not os.path.exists(dirname):
        os.makedirs(dirname)

    filename = os.path.join(dirname,
                            url.split('/')[-1]
                            if save_name is None else save_name)

    retry = 0
    retry_limit = 3
    while not (os.path.exists(filename) and md5file(filename) == md5sum):
        if os.path.exists(filename):
            sys.stderr.write("file %s  md5 %s" % (md5file(filename), md5sum))
        if retry < retry_limit:
            retry += 1
        else:
            raise RuntimeError("Cannot download {0} within retry limit {1}".
                               format(url, retry_limit))
        sys.stderr.write("Cache file %s not found, downloading %s" %
                         (filename, url))
        r = requests.get(url, stream=True)
        total_length = r.headers.get('content-length')

        if total_length is None:
            with open(filename, 'wb') as f:
                shutil.copyfileobj(r.raw, f)
        else:
            with open(filename, 'wb') as f:
                dl = 0
                total_length = int(total_length)
                for data in r.iter_content(chunk_size=4096):
                    if six.PY2:
                        data = six.b(data)
                    dl += len(data)
                    f.write(data)
                    done = int(50 * dl / total_length)
                    sys.stderr.write("\r[%s%s]" % ('=' * done,
                                                   ' ' * (50 - done)))
                    sys.stdout.flush()
    sys.stderr.write("\n")
    sys.stdout.flush()
    return filename


def fetch_all():
    for module_name in [
            x for x in dir(paddle.dataset) if not x.startswith("__")
    ]:
        if "fetch" in dir(
                importlib.import_module("paddle.dataset.%s" % module_name)):
            getattr(
                importlib.import_module("paddle.dataset.%s" % module_name),
                "fetch")()


def fetch_all_recordio(path):
    for module_name in [
            x for x in dir(paddle.dataset) if not x.startswith("__")
    ]:
        if "convert" in dir(
                importlib.import_module("paddle.dataset.%s" % module_name)) and \
                not module_name == "common":
            ds_path = os.path.join(path, module_name)
            must_mkdirs(ds_path)
            getattr(
                importlib.import_module("paddle.dataset.%s" % module_name),
                "convert")(ds_path)


def split(reader, line_count, suffix="%05d.pickle", dumper=pickle.dump):
    """
    you can call the function as:

    split(paddle.dataset.cifar.train10(), line_count=1000,
        suffix="imikolov-train-%05d.pickle")

    the output files as:

    |-imikolov-train-00000.pickle
    |-imikolov-train-00001.pickle
    |- ...
    |-imikolov-train-00480.pickle

    :param reader: is a reader creator
    :param line_count: line count for each file
    :param suffix: the suffix for the output files, should contain "%d"
                means the id for each file. Default is "%05d.pickle"
    :param dumper: is a callable function that dump object to file, this
                function will be called as dumper(obj, f) and obj is the object
                will be dumped, f is a file object. Default is cPickle.dump.
    """
    if not callable(dumper):
        raise TypeError("dumper should be callable.")
    lines = []
    indx_f = 0
    for i, d in enumerate(reader()):
        lines.append(d)
        if i >= line_count and i % line_count == 0:
            with open(suffix % indx_f, "w") as f:
                dumper(lines, f)
                lines = []
                indx_f += 1
    if lines:
        with open(suffix % indx_f, "w") as f:
            dumper(lines, f)


def cluster_files_reader(files_pattern,
                         trainer_count,
                         trainer_id,
                         loader=pickle.load):
    """
    Create a reader that yield element from the given files, select
    a file set according trainer count and trainer_id

    :param files_pattern: the files which generating by split(...)
    :param trainer_count: total trainer count
    :param trainer_id: the trainer rank id
    :param loader: is a callable function that load object from file, this
                function will be called as loader(f) and f is a file object.
                Default is cPickle.load
    """

    def reader():
        if not callable(loader):
            raise TypeError("loader should be callable.")
        file_list = glob.glob(files_pattern)
        file_list.sort()
        my_file_list = []
        for idx, fn in enumerate(file_list):
            if idx % trainer_count == trainer_id:
                print("append file: %s" % fn)
                my_file_list.append(fn)
        for fn in my_file_list:
            with open(fn, "r") as f:
                lines = loader(f)
                for line in lines:
                    yield line

    return reader


def convert(output_path, reader, line_count, name_prefix):
    import recordio
    """
    Convert data from reader to recordio format files.

    :param output_path: directory in which output files will be saved.
    :param reader: a data reader, from which the convert program will read
                   data instances.
    :param name_prefix: the name prefix of generated files.
    :param max_lines_to_shuffle: the max lines numbers to shuffle before
                                 writing.
    """

    assert line_count >= 1
    indx_f = 0

    def write_data(indx_f, lines):
        filename = "%s/%s-%05d" % (output_path, name_prefix, indx_f)
        writer = recordio.writer(filename)
        for l in lines:
            # FIXME(Yancey1989):
            # dumps with protocol: pickle.HIGHEST_PROTOCOL
            writer.write(pickle.dumps(l))
        writer.close()

    lines = []
    for i, d in enumerate(reader()):
        lines.append(d)
        if i % line_count == 0 and i >= line_count:
            write_data(indx_f, lines)
            lines = []
            indx_f += 1
            continue

    write_data(indx_f, lines)



ll__ = ['train', 'test', 'convert']                                                   
                                                                                         
URL_PREFIX = 'https://dataset.bj.bcebos.com/mnist/'                                      
TEST_IMAGE_URL = URL_PREFIX + 't10k-images-idx3-ubyte.gz'                                
TEST_IMAGE_MD5 = '9fb629c4189551a2d022fa330f9573f3'                                      
TEST_LABEL_URL = URL_PREFIX + 't10k-labels-idx1-ubyte.gz'                                
TEST_LABEL_MD5 = 'ec29112dd5afa0611ce80d1b7f02629c'                                      
TRAIN_IMAGE_URL = URL_PREFIX + 'train-images-idx3-ubyte.gz'                              
TRAIN_IMAGE_MD5 = 'f68b3c2dcbeaaa9fbdd348bbdeb94873'                                     
TRAIN_LABEL_URL = URL_PREFIX + 'train-labels-idx1-ubyte.gz'                              
TRAIN_LABEL_MD5 = 'd53e105ee54ea40749a09fcbcd1e9432'                                     
                                                                                         
                                                                                         
def reader_creator(image_filename, label_filename, buffer_size):                         
    def reader():                                                                        
        with gzip.GzipFile(image_filename, 'rb') as image_file:                          
            img_buf = image_file.read()                                                  
            with gzip.GzipFile(label_filename, 'rb') as label_file:                      
                lab_buf = label_file.read()                                              
                                                                                         
                step_label = 0                                                           
                                                                                         
                offset_img = 0                                                           
                # read from Big-endian                                                   
                # get file info from magic byte                                          
                # image file : 16B                                                       
                magic_byte_img = '>IIII'                                                 
                magic_img, image_num, rows, cols = struct.unpack_from(                   
                    magic_byte_img, img_buf, offset_img)                                 
                offset_img += struct.calcsize(magic_byte_img)                            
                offset_lab = 0                                                           


                # label file : 8B                                                                               
                magic_byte_lab = '>II'                                                                          
                magic_lab, label_num = struct.unpack_from(magic_byte_lab,                                       
                                                          lab_buf, offset_lab)                                  
                offset_lab += struct.calcsize(magic_byte_lab)                                                   
                                                                                                                
                while True:                                                                                     
                    if step_label >= label_num:                                                                 
                        break                                                                                   
                    fmt_label = '>' + str(buffer_size) + 'B'                                                    
                    labels = struct.unpack_from(fmt_label, lab_buf, offset_lab)                                 
                    offset_lab += struct.calcsize(fmt_label)                                                    
                    step_label += buffer_size                                                                   
                                                                                                                
                    fmt_images = '>' + str(buffer_size * rows * cols) + 'B'                                     
                    images_temp = struct.unpack_from(fmt_images, img_buf,                                       
                                                     offset_img)                                                
                    images = numpy.reshape(images_temp, (                                                       
                        buffer_size, rows * cols)).astype('float32')                                            
                    offset_img += struct.calcsize(fmt_images)                                                   
                                                                                                                
                    images = images / 255.0                                                                     
                    images = images * 2.0                                                                       
                    images = images - 1.0                                                                       
                                                                                                                
                    for i in range(buffer_size):                                                                
                        yield images[i, :], int(labels[i])                                                      
    return reader                                                                                               
                                                                                                                
def train():                                                                                                    
    """                                                                                                         
    MNIST training set creator.                                                                                 
    It returns a reader creator, each sample in the reader is image pixels in                                   
    [-1, 1] and label in [0, 9].                                                                                
    :return: Training reader creator                                                                            
    :rtype: callable                                                                                            
    """                                                                                                         
    return reader_creator(                                                                                      
        download(TRAIN_IMAGE_URL, 'mnist',                                                
                                       TRAIN_IMAGE_MD5),                                                        
        download(TRAIN_LABEL_URL, 'mnist',                                                
                                       TRAIN_LABEL_MD5), 100)                                                   

def test():                                                                          
    """                                                                              
    MNIST test set creator.                                                          
                                                                                     
    It returns a reader creator, each sample in the reader is image pixels in        
    [-1, 1] and label in [0, 9].                                                     
                                                                                     
    :return: Test reader creator.                                                    
    :rtype: callable                                                                 
    """                                                                              
    return reader_creator(                                                           
        download(TEST_IMAGE_URL, 'mnist', TEST_IMAGE_MD5),     
        download(TEST_LABEL_URL, 'mnist', TEST_LABEL_MD5),     
        100)                                                                         

def fetch():                                                                          
    download(TRAIN_IMAGE_URL, 'mnist', TRAIN_IMAGE_MD5)         
    download(TRAIN_LABEL_URL, 'mnist', TRAIN_LABEL_MD5)         
    download(TEST_IMAGE_URL, 'mnist', TEST_IMAGE_MD5)           
    download(TEST_LABEL_URL, 'mnist', TEST_LABEL_MD5)           
                                                                                      
                                                                                      
def convert(path):                                                                    
    """                                                                               
    Converts dataset to recordio format                                               
    """                                                                               
    convert(path, train(), 1000, "minist_train")                
    convert(path, test(), 1000, "minist_test")                  


########################################
###            training step 
########################################


#多层感知机                                                                                                                   
def multilayer_perceptron(input):                                                                                             
        #第一个全连接层，激活函数relu                                                                                         
        hidden1=fluid.layers.fc(input=input,size=100,act='relu')                                                              
        #第二个全连接层，激活函数relu                                                                                         
        hidden2=fluid.layers.fc(input=hidden1,size=100,act='relu')                                                            
        #以softmax为激活函数的全连接输出层，大小为label大小                                                                   
        fc=fluid.layers.fc(input=hidden2,size=10,act='softmax')                                                               
        return fc                                                                                                             
                                                                                                                              
#卷积神经网络                                                                                                                 
def convolutional_neural_network(input):                                                                                      
        #卷积层，卷积核大小为3*3，步长是1，一共有32个卷积核                                                                   
        conv_1=fluid.layers.conv2d(input=input,num_filters=32,filter_size=3,stride=1)                                         
        #池化层，池化核大小为2*2，步长是1，最大池化                                                                           
        pool_1=fluid.layers.pool2d(input=conv_1,pool_size=2,pool_stride=1,pool_type='max')                                    
        #第二个卷积层，卷积核大小为3*3，步长1，一共有64个卷积核                                                               
        conv_2=fluid.layers.conv2d(input=pool_1,num_filters=32,filter_size=3,stride=1)                                        
        #第二个池化层，池化核大小是2*2，步长1，最大池化                                                                       
        pool_2=fluid.layers.pool2d(input=conv_2,pool_size=2,pool_stride=1,pool_type='max')                                    
        #以softmax为激活函数的全连接输出层，大小为label的大小                                                                 
        #softmax一般用于多分类问题最后一层输出层的激活函数，作用是对输出归一化，这种情况下一般损失函数使用交叉熵              
        fc=fluid.layers.fc(input=pool_2,size=10,act='softmax')                                                                
        return fc                                                                                                             

#定义占位输入层和标签层                                                                                    
#图像是28*28的灰度图，所以输入的形状是[1,28,28]（灰度图是1通道，彩图3通道），理论上应该还有一个维度是Batch，
image=fluid.layers.data(name='image',shape=[1,28,28],dtype='float32')                                      
label=fluid.layers.data(name='label',shape=[1],dtype='int64')                                              
                                                                                                           
#获取前向传播网络结果                                                                                      
#result=multilayer_perceptron(image)    #使用多层感知机                                                    
result=convolutional_neural_network(image)                                                                 
                                                                                                           
#定义损失函数和准确率函数                                                                                  
cost=fluid.layers.cross_entropy(input=result,label=label)#交叉熵                                           
avg_cost=fluid.layers.mean(cost)#整个Batch的平均值                                                         
accuracy=fluid.layers.accuracy(input=result,label=label)                                                   
                                                                                                           
#克隆主程序，获得一个预测程序                                                                              
test_program=fluid.default_main_program().clone(for_test=True)                                             
                                                                                                           
#定义优化器，使用Adam优化器                                                                                
optimizer=fluid.optimizer.AdamOptimizer(learning_rate=0.001)                                               
opts=optimizer.minimize(avg_cost)                                                                          
                                                                                                           
#获取mnist数据集的reader，指定一个Batch的大小为128，一次喂入的张量shape是[128,1,28,28]                     
train_reader=paddle.batch(train(),batch_size=128)                                         
test_reader=paddle.batch(test(),batch_size=128)                                           

#定义执行器 
#place=fluid.CPUPlace()                                                                  
place=fluid.CUDAPlace(0) if use_GPU else fluid.CPUPlace()
  
exe=fluid.Executor(place)                                                                                                        
#初始化参数                                                                                                                      
exe.run(fluid.default_startup_program())                                                                                         

# 


# save the model or load the previous model's parameter
save_path = MODEL_DIR 
value_path = os.listdir(save_path)==[]
#print('------------')
#print(value_path)
#print('------------')
#if os.path.exists(save_path):
if not os.listdir(save_path)==[]:
   fluid.io.load_persistables(executor=exe, dirname=save_path)
   print('---  model loaded----- ')
#    fluid.io.load_params(executor=exe, dirname=save_path)

feeder=fluid.DataFeeder(place=place,feed_list=[image,label])                                                                     
#开始训练，5个pass                                                                                                               
for pass_id in range(5):                                                                                                         
        #开始训练                                                                                                                
        for batch_id,data in enumerate(train_reader()):                                                                          
                train_cost,train_acc=exe.run(program=fluid.default_main_program(),feed=feeder.feed(data),fetch_list=[avg_cost,accuracy])                                                                                                                          
                #每100个batch打印一次信息                                                                                        
                if batch_id%100==0:                                                                                              
                        print('Pass:%d Batch:%d Cost:%0.5f Accuracy:%0.5f'%(pass_id,batch_id,train_cost[0],train_acc[0]))        
                                                                                                                                 
        #每一个pass进行一次测试                                                                                                  
        test_accs=[]                                                                                                             
        test_costs=[]                                                                                                            
        for batch_id,data in enumerate(test_reader()):                                                                           
                test_cost,test_acc=exe.run(program=test_program,feed=feeder.feed(data),fetch_list=[avg_cost,accuracy])           
                test_accs.append(test_acc[0])                                                                                    
                test_costs.append(test_cost[0])                                                                                  
        #求测试结果的平均值                                                                                                      
        test_cost=(sum(test_costs)/len(test_costs))                                                                              
        test_acc=(sum(test_accs)/len(test_accs))                                                                                 
        print('Test:%d Cost:%0.5f Accuracy:%0.5f'%(pass_id,test_cost,test_acc))                                                  
        # 删除旧的模型文件
        #shutil.rmtree(save_path, ignore_errors=True)
        # 创建保持模型文件目录
        #os.makedirs(save_path)
        # 保存预测模型
        #fluid.io.save_inference_model(save_path, feeded_var_names=[image.name], target_vars=[model], executor=exe)
        fluid.io.save_persistables(executor=exe, dirname=save_path)
