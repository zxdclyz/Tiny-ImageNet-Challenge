# search "path to" and fix it
import argparse
import os
import random
import shutil
import time
import warnings

import numpy as np
import pandas as pd
import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.datasets as datasets
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image

# parser
model_names = sorted(name for name in models.__dict__
                     if name.islower() and not name.startswith("__")
                     and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')

parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet18',
                    choices=model_names,
                    help='model architecture: ' +
                    ' | '.join(model_names) +
                    ' (default: resnet18)')
parser.add_argument('-b', '--batch-size', default=256, type=int,
                    metavar='N',
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--epochs', default=90, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--gpu', default=None, type=int,
                    help='GPU id to use.')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)',
                    dest='weight_decay')
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. ')
parser.add_argument('-p', '--print-freq', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')

# main

best_acc1 = 0

##############

class inception_resnet_64(nn.Module):
    def __init__(self, num_classes=100, transform_input=False,
                 inception_blocks=None):
        super(inception_resnet_64, self).__init__()
        if inception_blocks is None:
            inception_blocks = [
                BasicConv2d, InceptionA, InceptionB, InceptionC,
                InceptionReductionA, InceptionReductionB
            ]
        assert len(inception_blocks) == 6
        conv_block = inception_blocks[0]
        inception_a = inception_blocks[1]
        inception_b = inception_blocks[2]
        inception_c = inception_blocks[3]
        inception_d = inception_blocks[4]
        inception_e = inception_blocks[5]

        self.transform_input = transform_input
        self.Conv2d_1a_3x3 = conv_block(3, 32, kernel_size=3, padding=4, stride=2)
        self.Conv2d_1b_1x1 = conv_block(32, 256, kernel_size=1)

        self.Mixed_2a = inception_a(256)
        self.Mixed_2b = inception_a(256)
        self.Mixed_2c = inception_a(256)
        self.Mixed_2d = inception_a(256)
        self.Mixed_2e = inception_a(256)

        self.Mixed_3a = inception_d(256)

        self.Mixed_4a = inception_b(896)
        self.Mixed_4b = inception_b(896)
        self.Mixed_4c = inception_b(896)
        self.Mixed_4d = inception_b(896)
        self.Mixed_4e = inception_b(896)
        self.Mixed_4f = inception_b(896)
        self.Mixed_4g = inception_b(896)
        self.Mixed_4h = inception_b(896)
        self.Mixed_4i = inception_b(896)
        self.Mixed_4j = inception_b(896)

        self.Mixed_5a = inception_e(896)

        self.Mixed_6a = inception_c(1792)
        self.Mixed_6b = inception_c(1792)
        self.Mixed_6c = inception_c(1792)
        self.Mixed_6d = inception_c(1792)
        self.Mixed_6e = inception_c(1792)

        self.fc = nn.Linear(1792, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                import scipy.stats as stats
                stddev = m.stddev if hasattr(m, 'stddev') else 0.1
                X = stats.truncnorm(-2, 2, scale=stddev)
                values = torch.as_tensor(X.rvs(m.weight.numel()), dtype=m.weight.dtype)
                values = values.view(m.weight.size())
                with torch.no_grad():
                    m.weight.copy_(values)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _transform_input(self, x):
        if self.transform_input:
            x_ch0 = torch.unsqueeze(x[:, 0], 1) * (0.229 / 0.5) + (0.485 - 0.5) / 0.5
            x_ch1 = torch.unsqueeze(x[:, 1], 1) * (0.224 / 0.5) + (0.456 - 0.5) / 0.5
            x_ch2 = torch.unsqueeze(x[:, 2], 1) * (0.225 / 0.5) + (0.406 - 0.5) / 0.5
            x = torch.cat((x_ch0, x_ch1, x_ch2), 1)
        return x

    def _forward(self, x):
        x = self.Conv2d_1a_3x3(x)

        x = self.Conv2d_1b_1x1(x)
        x = self.Mixed_2a(x)
        x = self.Mixed_2b(x)
        x = self.Mixed_2c(x)
        x = self.Mixed_2d(x)
        x = self.Mixed_2e(x)

        x = self.Mixed_3a(x)

        x = self.Mixed_4a(x)
        x = self.Mixed_4b(x)
        x = self.Mixed_4c(x)
        x = self.Mixed_4d(x)
        x = self.Mixed_4e(x)
        x = self.Mixed_4f(x)
        x = self.Mixed_4g(x)
        x = self.Mixed_4h(x)
        x = self.Mixed_4i(x)
        x = self.Mixed_4j(x)

        x = self.Mixed_5a(x)

        x = self.Mixed_6a(x)
        x = self.Mixed_6b(x)
        x = self.Mixed_6c(x)
        x = self.Mixed_6d(x)
        x = self.Mixed_6e(x)
        # Adaptive average pooling
        x = F.adaptive_avg_pool2d(x, (1, 1))
        # N x 2048 x 1 x 1
        x = F.dropout(x, training=self.training)
        # N x 2048 x 1 x 1
        x = torch.flatten(x, 1)
        # N x 2048
        x = self.fc(x)
        # N x 1000 (num_classes)
        return x

    # @torch.jit.unused
    # def eager_outputs(self, x, aux):
    #     # type: (Tensor, Optional[Tensor]) -> InceptionOutputs
    #     if self.training and self.aux_logits:
    #         return InceptionOutputs(x, aux)
    #     else:
    #         return x

    def forward(self, x):
        x = self._transform_input(x)
        x = self._forward(x)
        return x

class BasicConv2d(nn.Module):

    def __init__(self, in_channels, out_channels, **kwargs):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)
        self.bn = nn.BatchNorm2d(out_channels, eps=0.001)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return F.relu(x, inplace=True)

class InceptionA(nn.Module):
    def __init__(self, in_channels, conv_block=None):
        super(InceptionA, self).__init__()
        if conv_block is None:
            conv_block = BasicConv2d
        self.branch1x1 = conv_block(in_channels, 32, kernel_size=1)

        self.branch1x1_3x3_1 = conv_block(in_channels, 32, kernel_size=1)
        self.branch1x1_3x3_2 = conv_block(32, 32, kernel_size=3, padding=1)

        self.branch1x1_3x3dbl_1 = conv_block(in_channels, 32, kernel_size=1)
        self.branch1x1_3x3dbl_2 = conv_block(32, 32, kernel_size=3, padding=1)
        self.branch1x1_3x3dbl_3 = conv_block(32, 32, kernel_size=3, padding=1)

        self.conv1x1 = conv_block(96, 256, kernel_size=1)

    def _forward(self, x):
        branch1x1 = self.branch1x1(x)

        branch1x1_3x3 = self.branch1x1_3x3_1(x)
        branch1x1_3x3 = self.branch1x1_3x3_2(branch1x1_3x3)

        branch1x1_3x3dbl = self.branch1x1_3x3dbl_1(x)
        branch1x1_3x3dbl = self.branch1x1_3x3dbl_2(branch1x1_3x3dbl)
        branch1x1_3x3dbl = self.branch1x1_3x3dbl_3(branch1x1_3x3dbl)

        branch_inception = torch.cat([branch1x1, branch1x1_3x3, branch1x1_3x3dbl], 1)
        branch_inception = self.conv1x1(branch_inception)

        outputs = x + branch_inception
        return outputs

    def forward(self, x):
        output = self._forward(x)
        return F.leaky_relu(output, inplace=True)

class InceptionB(nn.Module):

    def __init__(self, in_channels, conv_block=None):
        super(InceptionB, self).__init__()
        if conv_block is None:
            conv_block = BasicConv2d

        self.branch1x1 = conv_block(in_channels, 128, kernel_size=1)

        self.branch7x7_1 = conv_block(in_channels, 128, kernel_size=1)
        self.branch7x7_2 = conv_block(128, 128, kernel_size=(1, 7), padding=(0, 3))
        self.branch7x7_3 = conv_block(128, 128, kernel_size=(7, 1), padding=(3, 0))

        self.conv1x1 = conv_block(256, 896, kernel_size=1)

    def _forward(self, x):
        branch1x1 = self.branch1x1(x)

        branch7x7 = self.branch7x7_1(x)
        branch7x7 = self.branch7x7_2(branch7x7)
        branch7x7 = self.branch7x7_3(branch7x7)

        branch_inception = torch.cat([branch1x1, branch7x7], 1)
        branch_inception = self.conv1x1(branch_inception)

        outputs = x + branch_inception
        return outputs

    def forward(self, x):
        output = self._forward(x)
        return F.leaky_relu(output, inplace=True)

class InceptionC(nn.Module):

    def __init__(self, in_channels, conv_block=None):
        super(InceptionC, self).__init__()
        if conv_block is None:
            conv_block = BasicConv2d
        self.branch1x1 = conv_block(in_channels, 192, kernel_size=1)

        self.branch3x3_1 = conv_block(in_channels, 192, kernel_size=1)
        self.branch3x3_2 = conv_block(192, 192, kernel_size=(1, 3), padding=(0, 1))
        self.branch3x3_3 = conv_block(192, 192, kernel_size=(3, 1), padding=(1, 0))

        self.conv1x1 = conv_block(384, 1792, kernel_size=1)


    def _forward(self, x):
        branch1x1 = self.branch1x1(x)

        branch3x3 = self.branch3x3_1(x)
        branch3x3 = self.branch3x3_2(branch3x3)
        branch3x3 = self.branch3x3_3(branch3x3)

        branch_inception = torch.cat([branch1x1, branch3x3], 1)
        branch_inception = self.conv1x1(branch_inception)

        outputs = x + branch_inception
        return outputs

    def forward(self, x):
        output = self._forward(x)
        return F.leaky_relu(output, inplace=True)

class InceptionReductionA(nn.Module):
    def __init__(self, in_channels,  conv_block=None):
        super(InceptionReductionA, self).__init__()
        if conv_block is None:
            conv_block = BasicConv2d
        
        # 3x3 Maxpool stride 2

        self.branch3x3 = conv_block(in_channels, 384, kernel_size=3, stride=2)

        self.branch3x3dbl_1 = conv_block(in_channels, 192, kernel_size=1)
        self.branch3x3dbl_2 = conv_block(192, 192, kernel_size=3, padding=1)
        self.branch3x3dbl_3 = conv_block(192, 256, kernel_size=3, stride=2)

    def _forward(self, x):
        branchpool = F.max_pool2d(x, kernel_size=3, stride=2)

        branch3x3 = self.branch3x3(x)

        branch3x3dbl = self.branch3x3dbl_1(x)
        branch3x3dbl = self.branch3x3dbl_2(branch3x3dbl)
        branch3x3dbl = self.branch3x3dbl_3(branch3x3dbl)

        outputs = [branchpool, branch3x3, branch3x3dbl]
        return outputs

    def forward(self, x):
        outputs = self._forward(x)
        return torch.cat(outputs, 1)

class InceptionReductionB(nn.Module):
    def __init__(self, in_channels, conv_block=None):
        super(InceptionReductionB, self).__init__()
        if conv_block is None:
            conv_block = BasicConv2d
        
        # 3x3 Maxpool stride 2

        self.branch3x3_11 = conv_block(in_channels, 256, kernel_size=1)
        self.branch3x3_12 = conv_block(256, 384, kernel_size=3, stride=2)

        self.branch3x3_21 = conv_block(in_channels, 256, kernel_size=1)
        self.branch3x3_22 = conv_block(256, 256, kernel_size=3, stride=2)

        self.branch3x3dbl_1 = conv_block(in_channels, 256, kernel_size=1)
        self.branch3x3dbl_2 = conv_block(256, 256, kernel_size=3, padding=1)
        self.branch3x3dbl_3 = conv_block(256, 256, kernel_size=3, stride=2)

    def _forward(self, x):
        branch_pool = F.max_pool2d(x, kernel_size=3, stride=2)

        branch3x3_1 = self.branch3x3_11(x)
        branch3x3_1 = self.branch3x3_12(branch3x3_1)

        branch3x3_2 = self.branch3x3_21(x)
        branch3x3_2 = self.branch3x3_22(branch3x3_2)

        branch3x3dbl = self.branch3x3dbl_1(x)
        branch3x3dbl = self.branch3x3dbl_2(branch3x3dbl)
        branch3x3dbl = self.branch3x3dbl_3(branch3x3dbl)

        outputs = [branch_pool, branch3x3_1, branch3x3_2, branch3x3dbl]
        return outputs

    def forward(self, x):
        outputs = self._forward(x)
        return torch.cat(outputs, 1)

##############

def main():
    args = parser.parse_args()

    if args.gpu is not None:
        warnings.warn('You have chosen a specific GPU. This will completely '
                      'disable data parallelism.')

    ngpus_per_node = torch.cuda.device_count()  # to be consider
    main_worker(args.gpu, ngpus_per_node, args)


def main_worker(gpu, ngpus_per_node, args):
    global best_acc1

    args.gpu = gpu
    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))

    # create model
    # print("=> creating model '{}'".format(args.arch))
    # model = models.__dict__[args.arch]()
    print("inception resnet test")
    model = inception_resnet_64()
    
#     model.aux_logits=False

    if args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)
    else:
        # DataParallel will divide and allocate batch_size to all available GPUs
        if args.arch.startswith('alexnet') or args.arch.startswith('vgg'):
            model.features = torch.nn.DataParallel(model.features)
            model.cuda()
        else:
            model = torch.nn.DataParallel(model).cuda()

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda(args.gpu)

    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)

    # accelerate
    cudnn.benchmark = True

    # set dataset
    train_dataset = TinyImageNetDataset(
        './TinyImageNet', './TinyImageNet/train.txt')
    val_dataset = TinyImageNetDataset(
        './TinyImageNet', './TinyImageNet/val.txt')
    test_dataset = TinyImageNetDataset(
        './TinyImageNet', './TinyImageNet/test.txt')

    train_sampler = None

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=(
            train_sampler is None),
        num_workers=args.workers, pin_memory=True, sampler=train_sampler)

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)

    for epoch in range(args.start_epoch, args.epochs):
        adjust_learning_rate(optimizer, epoch, args)

        # train for one epoch
        train(train_loader, model, criterion, optimizer, epoch, args)

        # evaluate on validation set
        acc1 = validate(val_loader, model, criterion, args)

        # remember best acc@1 and save
        best_acc1 = max(acc1, best_acc1)
        
        test(test_loader, model, args)

    test(test_loader, model, args)


def train(train_loader, model, criterion, optimizer, epoch, args):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, losses, top1, top5],
        prefix="Epoch: [{}]".format(epoch))

    # switch to train mode
    model.train()

    end = time.time()
    for i, (images, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        if args.gpu is not None:
            images = images.cuda(args.gpu, non_blocking=True)
        target = target.cuda(args.gpu, non_blocking=True)

        # compute output
        output = model(images)
        loss = criterion(output, target.long())

        # measure accuracy and record loss
        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        losses.update(loss.item(), images.size(0))
        top1.update(acc1[0], images.size(0))
        top5.update(acc5[0], images.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            progress.display(i)


def validate(val_loader, model, criterion, args):
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(
        len(val_loader),
        [batch_time, losses, top1, top5],
        prefix='Test: ')

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, (images, target) in enumerate(val_loader):
            if args.gpu is not None:
                images = images.cuda(args.gpu, non_blocking=True)
            target = target.cuda(args.gpu, non_blocking=True)

            # compute output
            output = model(images)
            loss = criterion(output, target)

            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), images.size(0))
            top1.update(acc1[0], images.size(0))
            top5.update(acc5[0], images.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                progress.display(i)

        # TODO: this should also be done with the ProgressMeter
        print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
              .format(top1=top1, top5=top5))

    return top1.avg


prediction = []


def test(test_loader, model, args):
    model.eval()
    global prediction

    if prediction:
        prediction = []

    with torch.no_grad():
        for i, images in enumerate(test_loader):
            if args.gpu is not None:
                images = images.cuda(args.gpu, non_blocking=True)

            output = model(images)
            prediction += output.argmax(dim=1).tolist()

    index = []
    labels = open('./TinyImageNet/test.txt').readlines()
    for line in labels:
        index.append(line.split('/')[1].strip())

    df = pd.DataFrame(prediction, index=index)
    df.to_csv('./prediction.csv')


def default_loader(path):
    return Image.open(path).convert('RGB')


class TinyImageNetDataset(torch.utils.data.Dataset):
    """data loader, considering change PIL to cv"""

    def __init__(self, root, data_list, transform=None, loader=default_loader):
        # root: your_path/TinyImageNet/
        # data_list: your_path/TinyImageNet/train.txt etc.
        images = []
        labels = open(data_list).readlines()
        for line in labels:
            items = line.strip('\n').split()
            img_name = items[0]

            # test list contains only image name
            test_flag = True if len(items) == 1 else False
            label = None if test_flag == True else np.array(int(items[1]))

            if os.path.isfile(os.path.join(root, img_name)):
                images.append((img_name, label))
            else:
                print(os.path.join(root, img_name) + 'Not Found.')

        self.root = root
        self.images = images
        self.transform = transform
        self.loader = loader

    def __getitem__(self, index):
        img_name, label = self.images[index]
        img = self.loader(os.path.join(self.root, img_name))
        if self.transform is not None:
            img = self.transform(img)
        img = np.array(img, dtype='float32')
        img = img.transpose(2, 0, 1)
        return (img, label) if label is not None else img

    def __len__(self):
        return len(self.images)


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


def adjust_learning_rate(optimizer, epoch, args):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = args.lr * (0.1 ** (epoch // 30))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


if __name__ == '__main__':
    main()