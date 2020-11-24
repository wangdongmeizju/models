import numpy as np #numpy只保存数值，用于数值运算，解决Python标准库中的list只能保存对象的指针的问题
import os #本例子中没有使用到
import gzip #使用zlib来压缩和解压缩数据文件，读写gzip文件
import struct #通过引入struct模块来处理图片中的二进制数据
import mxnet as mx #引入MXNet包
import logging #引入logging包记录日志
import shutil
import argparse

parser = argparse.ArgumentParser(description='MXNet Gluon MNIST Example')
parser.add_argument('--batch_size', type=int, default=128,
                    help='batch size for training and testing (default: 100)')
parser.add_argument('--data_dir', type=str, default='/MNIST_data',
                    help='data dir')
parser.add_argument('--epochs', type=int, default=100,
                    help='number of epochs to train (default: 10)')
parser.add_argument('--lr', type=float, default=0.1,
                    help='learning rate (default: 0.1)')
parser.add_argument('--cuda', type=int, default=1,
                    help='Train on GPU with CUDA')
parser.add_argument('--log_save_path', type=str,   default="/wangdongmei/models/logs/log_mx_mnist_multi/",  help='log out')
parser.add_argument('--model_save_path', type=str, default="/wangdongmei/models/logs/log_mx_mnist_multi/", help='model path')
parser.add_argument('--isresume', type=bool, default=False, help='resumimg the training from the checkpoint')
parser.add_argument('--num_gpus', type=int, default=2, help='number of gpu')
#opt = parser.parse_args([])
opt = parser.parse_args()
                                      
log_save_path = opt.log_save_path    
model_save_path = opt.model_save_path
isresume = opt.isresume

if not isresume:
  if os.path.exists(model_save_path):
   #  if not os.listdir(model_save_path):  
#      shutil.rmtree(model_save_path)
       model_path=model_save_path+'/*'
       os.system("rm -rf {}".format(model_path))

if not os.path.isdir(model_save_path):
       os.mkdir(model_save_path)


#利用MNIST数据集进行训练
def read_data(label_url,image_url): #定义读取数据的函数
    with gzip.open(label_url) as flbl: #解压标签包
        magic, num = struct.unpack(">II",flbl.read(8)) #采用Big Endian的方式读取两个int类型的数据，且参考MNIST官方格式介绍，magic即为magic number (MSB first) 用于表示文件格式，num即为文件夹内包含的数据的数量
        label = np.fromstring(flbl.read(),dtype=np.int8) #将标签包中的每一个二进制数据转化成其对应的十进制数据，且转换后的数据格式为int8（-128 to 127）格式，返回一个数组
    with gzip.open(image_url,'rb') as fimg: #已只读形式解压图像包
        magic, num, rows, cols = struct.unpack(">IIII",fimg.read(16)) #采用Big Endian的方式读取四个int类型数据，且参考MNIST官方格式介绍，magic和num上同，rows和cols即表示图片的行数和列数
        image = np.fromstring(fimg.read(),dtype=np.uint8).reshape(len(label),rows,cols) #将图片包中的二进制数据读取后转换成无符号的int8格式的数组，并且以标签总个数，行数，列数重塑成一个新的多维数组
    return (label,image) #返回读取成功的label数组和image数组
#且fileobject.read(size)的时候是按照流的方式读取（可test）

(train_lbl, train_img) = read_data( opt.data_dir+'/train-labels-idx1-ubyte.gz', opt.data_dir+'/train-images-idx3-ubyte.gz') #构建训练数据
(val_lbl, val_img)     = read_data( opt.data_dir+'/t10k-labels-idx1-ubyte.gz',  opt.data_dir+'/t10k-images-idx3-ubyte.gz') #构建测试数据

def to4d(img): #定义一个函数用于生成四维矩阵
    return img.reshape(img.shape[0],1,28,28).astype(np.float32)/255 #将图像包中的数组以标签总个数，图像通道数（MNIST数据集为黑白数据集故只为1），行数，列数重塑后复制为一个数据类型为float32的新的四维矩阵，且其中的元素值都除以255后转化为0-1的浮点值

batch_size = opt.batch_size #定义每次处理数据的数量为10
train_iter = mx.io.NDArrayIter(to4d(train_img),train_lbl,batch_size,shuffle=True) #构建训练数据迭代器，且其中shuffle表示采用可拖动的方式，意味着可以将在早期已经训练过的数据在后面再次训练
val_iter = mx.io.NDArrayIter(to4d(val_img),val_lbl,batch_size) #构建测试数据迭代器

#创建多层网络模型
data = mx.sym.Variable('data') #创建一个用于输入数据的PlaceHolder变量（占位符）
data = mx.sym.Flatten(data=data) #将data中的四维数据转化为二维数据且其中一维为每次处理数据的数量，第二维即为每张图片的图像通道数×长×宽（即为其像素点个数×图像通道数）
fc1 = mx.sym.FullyConnected(data=data,name='fc1',num_hidden=128) #创建第一层全连接层，输入数据为data，num_hidden表示该隐藏层有128个用于输出的节点
act1 = mx.sym.Activation(data=fc1,name='relu1',act_type='relu') #为第一层全连接层设定一个Relu激活函数，输入数据为fc1
fc2 = mx.sym.FullyConnected(data=act1,name='fc2',num_hidden=64) #创建第二层全连接层，输入数据为act1，num_hidden表示该隐藏层有64个用于输出的节点
act2 = mx.sym.Activation(data=fc2,name='relu2',act_type='relu') #为第一层全连接层设定一个Relu激活函数，输入数据为fc2
fc3 = mx.sym.FullyConnected(data=act2,name='fc3',num_hidden=10) #创建第三层全连接层，输入数据为act2，num_hidden表示该隐藏层有10个用于输出的节点
mlp = mx.sym.SoftmaxOutput(data=fc3,name='softmax') #对输入的数据执行softmax变换，并且通过利用logloss执行BP算法

logging.getLogger().setLevel(logging.DEBUG) #返回作为层次结构根记录器的记录器，且记录等级作为DEBUG

model_prefix = opt.model_save_path+'mx_mlp'
checkpoint = mx.callback.do_checkpoint(model_prefix,20)
batch_end_callbacks = [mx.contrib.tensorboard.LogMetricsCallback(log_save_path)]


#构建前馈神经网络模型 - GPU training 
#model = mx.model.FeedForward(
model = mx.module.Module(
    symbol = mlp, #使网络结构为构建好的mlp
    context = [mx.gpu(i) for i in range(opt.num_gpus)] 
    #num_epoch = opt.epochs, #数据的训练次数为10
#    learning_rate = opt.lr #使模型按照学习率为0.1进行训练
)

# check the model dir and load the previous trained model parameters
if not os.listdir(opt.model_save_path)==[]:
   sym, arg_params, aux_params = mx.model.load_checkpoint(model_prefix,epoch=20)
   model.bind(data_shapes=train_iter.provide_data, label_shapes=train_iter.provide_label)
   model.set_params(arg_params, aux_params, allow_missing=True)
   print('---  model loaded----- ')

#数据拟合，训练模型
model.fit(
    train_iter, #设置训练迭代器
    val_iter, #设置测试迭代器
    #X = train_iter, #设置训练迭代器
    #eval_data = val_iter, #设置测试迭代器
    num_epoch = opt.epochs, #数据的训练次数为10
    optimizer_params={'learning_rate':opt.lr, 'momentum': 0.9},
    #batch_end_callback = mx.callback.Speedometer(batch_size,200),
    batch_end_callback =  batch_end_callbacks,
    epoch_end_callback=checkpoint
)
