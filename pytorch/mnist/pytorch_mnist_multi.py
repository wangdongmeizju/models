# -*- coding:utf-8 -*-
import torch
import torch.nn as nn
import torch.utils.data as Data
import torchvision
import argparse
# import visdom
import numpy as np
import os
from tensorboardX import SummaryWriter
import time
#time.sleep(10000)
import shutil

# 获取参数
parser = argparse.ArgumentParser(description='PyTorch Training')
parser.add_argument('--data_dir', default='/MNIST_pytorch/', help='dataset path')
parser.add_argument('--log_save_path',   default='/inspur/models/logs/log_pytorch_mnist_multi', help='log save path')
parser.add_argument('--model_save_path', default='/inspur/models/logs/log_pytorch_mnist_multi/ckpt.pth', help='log save path')
parser.add_argument('--epoch', default=10, type=int, help='dataset path')
parser.add_argument('--batch_size', default=64, type=int, help='dataset path')
parser.add_argument('--is_cuda', default="true", help='cuda')
parser.add_argument('--num_gpus', default=2, type=int, help='gpu num')
parser.add_argument('--isresume', type=bool, default=False, help='resumimg the training from the checkpoint')

args = parser.parse_args()
#args = parser.parse_args([])

log_save_path = args.log_save_path
model_save_path = args.model_save_path
isresume = args.isresume

if not isresume:
  if os.path.exists(log_save_path):
   #  if not os.listdir(model_save_path):  
#      shutil.rmtree(log_save_path)
      model_path=log_save_path+'/*'
      os.system("rm -rf {}".format(model_path))


if not os.path.isdir(log_save_path):
       os.mkdir(log_save_path)

torch.manual_seed(1)
EPOCH = args.epoch
BATCH_SIZE = args.batch_size
LR = 0.001
DOWNLOAD_MNIST = False

train_data = torchvision.datasets.MNIST(
    root=args.data_dir,
    train=True,
    transform=torchvision.transforms.ToTensor(),
    download=DOWNLOAD_MNIST
)

test_data = torchvision.datasets.MNIST(
    root=args.data_dir, train=False
)

train_loader = Data.DataLoader(
    dataset=train_data, batch_size=BATCH_SIZE,
    shuffle=True,
)

train_x = torch.unsqueeze(train_data.train_data, dim=1).type(torch.FloatTensor)[:2000] / 255
train_y = train_data.train_labels[:2000]
test_x = torch.unsqueeze(test_data.test_data, dim=1).type(torch.FloatTensor)[:2000] / 255
test_y = test_data.test_labels[:2000]


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=16,
                      kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=32,
                      kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )

        self.out = nn.Linear(32 * 7 * 7, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.size(0), -1)
        output = self.out(x)
        return output

cnn = CNN()
with SummaryWriter(log_save_path) as writer:
    inputs = torch.Tensor(1, 1, 28, 28)
    writer.add_graph(cnn, (inputs,))

print("$$$$$$$$$$$$$$$$$$$")
print(args.is_cuda)
if args.is_cuda is "true":
    device_ids = []
    for i in range(args.num_gpus):
        device_ids.append(i)
    cnn = cnn.cuda()
    cnn = nn.DataParallel(cnn, device_ids=device_ids)
optimizer = torch.optim.Adam(cnn.parameters(), lr=LR)
loss_func = nn.CrossEntropyLoss()
# vis=visdom.Visdom()


model_path = os.path.isfile(args.model_save_path)
#print('---------')  
#print(log_save_path)
#print('---------')  
if model_path:
   checkpoint = torch.load(args.model_save_path)
   cnn.load_state_dict(checkpoint['model_state_dict'])
   optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
   epoch = checkpoint['epoch']
   loss = checkpoint['loss']
   acc = checkpoint['acc']
   print('-----------------  ckpt loaded  --------------------!')

for epoch in range(EPOCH):
    writer = SummaryWriter(log_save_path)
    for step, (b_x, b_y) in enumerate(train_loader):
        if args.is_cuda is "true":
            b_x = b_x.cuda()
            b_y = b_y.cuda()
        output = cnn(b_x)
        loss = loss_func(output, b_y)
        # print('train_loader', train_loader)
        # print('acc', acc)
        current_output = cnn(train_x)
        current_pred = torch.max(current_output, 1)[1].data.cpu().numpy().squeeze()
        real_labels = train_y.numpy()
        acc = sum(current_pred == real_labels) / 2000
        print('loss', loss.data.cpu().numpy(), '|acc:', acc)
        print(loss.data)

        # vis.line(X=torch.FloatTensor([step]),
        #          Y=loss.data.view([1]),
        #          win='loss', update='append' if step > 0 else None)
        # vis.line(X=torch.FloatTensor([step]),
        #          Y=[acc],
        #          win='acc', update='append' if step > 0 else None)
        # y=[[step, step]]
        
        y = np.array([[loss.data.cpu().numpy(), acc]])

        # vis.line(Y=y,X=np.array([[step,step]]),
        # win='acc', update='append', opts=dict(legend=['Sine', 'Cosine']))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        niter = epoch * len(train_loader) + step
        writer.add_scalar('Train_loss', loss.data, niter)
        writer.add_scalar('Train_acc', acc, niter)
    writer.close()

    torch.save({
            'epoch': epoch,
            'model_state_dict': cnn.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss,
            'acc': acc,
            }, args.model_save_path)

test_output = cnn(test_x[:2000])
pred_y = torch.max(test_output, 1)[1].data.cpu().numpy().squeeze()
real_out = test_y[:2000].numpy()
print(pred_y, 'prediction number')
print(real_out, 'real number')
print('test_set_acc:', sum(pred_y == real_out) / 2000)
