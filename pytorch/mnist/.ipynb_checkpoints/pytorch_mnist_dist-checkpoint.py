from __future__ import print_function

import argparse
import os
import shutil

from tensorboardX import SummaryWriter
from torchvision import datasets, transforms
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

WORLD_SIZE = int(os.environ.get('WORLD_SIZE', 1))


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 20, 5, 1)
        self.conv2 = nn.Conv2d(20, 50, 5, 1)
        self.fc1 = nn.Linear(4 * 4 * 50, 500)
        self.fc2 = nn.Linear(500, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 4 * 4 * 50)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


def train(args, model, device, train_loader, optimizer, epoch, writer, model_file):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tloss={:.4f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), loss.item()))
            niter = epoch * len(train_loader) + batch_idx
            writer.add_scalar('loss', loss.item(), niter)
        if batch_idx % 400 == 0 and args.rank == "1" and batch_idx != 0:
            torch.save({'epoch': epoch,
                       'model_state_dict': model.module.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict()},
                      model_file)


def test(args, model, device, test_loader, writer, epoch):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.max(1, keepdim=True)[1]  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    print('\naccuracy={:.4f}\n'.format(float(correct) / len(test_loader.dataset)))
    writer.add_scalar('accuracy', float(correct) / len(test_loader.dataset), epoch)


def should_distribute():
    return dist.is_available() and WORLD_SIZE > 1


def is_distributed():
    return dist.is_available() and dist.is_initialized()


def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--data_dir', default='/MNIST_pytorch', help='dataset path')
    parser.add_argument('--batch_size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test_batch_size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=1, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                        help='learning rate (default: 0.01)')
    parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                        help='SGD momentum (default: 0.5)')
    parser.add_argument('--no_cuda', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log_interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--save_model', default=True,
                        help='For Saving the current Model')
    parser.add_argument('--model_save_path', default='/inspur/models/logs/log_pytorch_mnist_dist',
                        help='model save path')
    parser.add_argument('--log_dir', default='logs', metavar='L',
                        help='directory where summary logs are stored')
    parser.add_argument('--rank', default='0', metavar='R', help='rank')
    parser.add_argument('--isresume', type=bool, default=False, help='resumimg the training from the checkpoint')

    epoch_checkpoint = 1
    if dist.is_available():
        parser.add_argument('--backend', type=str, help='Distributed backend',
                            choices=[dist.Backend.GLOO, dist.Backend.NCCL, dist.Backend.MPI],
                            default=dist.Backend.GLOO)
    
    args = parser.parse_args()
    #args = parser.parse_args([])

    model_save_path = args.model_save_path
    isresume = args.isresume

    if not isresume:
      if os.path.exists(model_save_path):
     #  if not os.listdir(model_save_path):  
    #     shutil.rmtree(model_save_path)
          model_path=model_save_path+'/*'
          os.system("rm -rf {}".format(model_path))


    if not os.path.isdir(model_save_path):
         os.mkdir(model_save_path)


    use_cuda = not args.no_cuda and torch.cuda.is_available()
    if use_cuda:
        print('Using CUDA')

    model_file = args.model_save_path + "/mnist_cnn.pt"
    # model_file = '/opt/mnist_cnn.pt"

    writer = SummaryWriter(args.log_dir)

    torch.manual_seed(args.seed)

    device = torch.device("cuda" if use_cuda else "cpu")

    if should_distribute():
        print('Using distributed PyTorch with {} backend'.format(args.backend))
        dist.init_process_group(backend=args.backend)

    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST(args.data_dir, train=True, download=False,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])),
        batch_size=args.batch_size, shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST(args.data_dir, train=False, transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])),
        batch_size=args.test_batch_size, shuffle=False, **kwargs)

    model = Net().to(device)

    if is_distributed():
        Distributor = nn.parallel.DistributedDataParallel if use_cuda \
            else nn.parallel.DistributedDataParallelCPU
        model = Distributor(model)

    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)

    #if os.path.exists(model_file) and args.rank == "1":
    if os.path.exists(model_file):
        model.module.load_state_dict(torch.load(model_file)['model_state_dict'])
        optimizer.load_state_dict(torch.load(model_file)['optimizer_state_dict'])

        epoch_checkpoint = torch.load(model_file)['epoch']
        print("It has been traing in {} epoch".format(epoch_checkpoint))

    for epoch in range(epoch_checkpoint, args.epochs + 1):
        train(args, model, device, train_loader, optimizer, epoch, writer, model_file)
        test(args, model, device, test_loader, writer, epoch)

    # if (args.save_model):
    #     if not os.path.isdir(args.model_save_path):
    #         os.mkdir(args.model_save_path)
    #     # torch.save(model.state_dict(), args.model_save_path + "/mnist_cnn.pt")
    #     torch.save({'epoch':epoch,
    #         'model_state_dict':model.state_dict(),
    #         'optimizer_state_dict':optimizer.state_dict()},
    #         model_file)


if __name__ == '__main__':
    main()
