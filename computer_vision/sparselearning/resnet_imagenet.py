'''ResNet in PyTorch.

For Pre-activation ResNet, see 'preact_resnet.py'.

Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import get_num_classes


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, builder,in_planes, planes, stride=1,downsample=False,base_width=64):

        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNetBuilder(object):
    def __init__(self, config):
        self.config = config

    def conv(self, kernel_size, in_planes, out_planes, stride=1):
        if kernel_size == 3:
            conv = self.config['conv'](
                in_planes, out_planes, kernel_size=3, stride=stride,
                padding=1, bias=False)
        elif kernel_size == 1:
            conv = nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride,
                             bias=False)
        elif kernel_size == 7:
            conv = nn.Conv2d(in_planes, out_planes, kernel_size=7, stride=stride,
                             padding=3, bias=False)
        else:
            return None

        nn.init.kaiming_normal_(conv.weight,
                                mode=self.config['conv_init'],
                                nonlinearity='relu')

        return conv

    def conv3x3(self, in_planes, out_planes, stride=1):
        """3x3 convolution with padding"""
        c = self.conv(3, in_planes, out_planes, stride=stride)
        return c

    def conv1x1(self, in_planes, out_planes, stride=1):
        """1x1 convolution with padding"""
        c = self.conv(1, in_planes, out_planes, stride=stride)
        return c

    def conv7x7(self, in_planes, out_planes, stride=1):
        """7x7 convolution with padding"""
        c = self.conv(7, in_planes, out_planes, stride=stride)
        return c

    def batchnorm(self, planes):
        bn = nn.BatchNorm2d(planes)
        nn.init.constant_(bn.weight, 1)
        nn.init.constant_(bn.bias, 0)

        return bn

    def activation(self):
        return nn.ReLU(inplace=True)


builder = ResNetBuilder({'conv': nn.Conv2d, 'conv_init': 'fan_out'})


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, builder, inplanes, planes, stride=1, downsample=None, base_width=64):
        super(Bottleneck, self).__init__()
        width = int(planes * base_width / 64)
        self.conv1 = builder.conv1x1(inplanes, width)
        self.bn1 = builder.batchnorm(width)
        self.conv2 = builder.conv3x3(width, width, stride=stride)
        self.bn2 = builder.batchnorm(width)
        self.conv3 = builder.conv1x1(width, planes * self.expansion)
        self.bn3 = builder.batchnorm(planes * self.expansion)
        self.relu = builder.activation()
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        if self.bn1 is not None:
            out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        if self.bn2 is not None:
            out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        if self.bn3 is not None:
            out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual

        out = self.relu(out)

        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes, args):
        super(ResNet, self).__init__()
        self.in_planes = 64
        self.base_width = 64
        self.num_ensemble = args.num_ensemble
        self.num_classes = num_classes
        self.blocks_in_head = args.blocks_in_head
        self.conv1 = builder.conv7x7(3, 64, stride=2)
        self.bn1 = builder.batchnorm(64)
        self.relu = builder.activation()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.total_blocks = sum(num_blocks)
        self.blocks_shared = self.total_blocks - self.blocks_in_head
        if self.blocks_shared < 0:
            raise ValueError(f'Too many blocks_in_head. Can be at most the total number of blocks ({self.total_blocks})')

        print(f'total blocks: {self.total_blocks}')
        print(f'blocks shared: {self.blocks_shared}')
        print(f'blocks in head: {self.blocks_in_head}')

        # for each head we make a complete copy of the network at first, then remove the shared blocks
        for head in range(self.num_ensemble + 1):  # one extra, which will be the shared backbone
            self.in_planes = 64  # crucial to reset this for each head, as it's updated in _make_layer
            superblocks = []
            superblocks.extend(self._make_layer(builder,block, 64, num_blocks[0]))
            superblocks.extend(self._make_layer(builder,block, 128, num_blocks[1], stride=2))
            superblocks.extend(self._make_layer(builder,block, 256, num_blocks[2], stride=2))
            superblocks.extend(self._make_layer(builder,block, 512, num_blocks[3], stride=2))

            if head != self.num_ensemble:  # rm the shared blocks that are not needed in the heads
                for block_num in range(self.blocks_shared):
                    del superblocks[0]
                setattr(self, f'blocks_head{head}', nn.Sequential(*superblocks))
            else:
                for block_num in range(self.blocks_in_head):
                    del superblocks[-1]
                setattr(self, f'shared_blocks', nn.Sequential(*superblocks))

        for head in range(self.num_ensemble):
            setattr(self, f'fc_{head}', nn.Linear(512*block.expansion, self.num_classes, bias=False))

    def _make_layer(self, builder, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.in_planes != planes * block.expansion:
            dconv = builder.conv1x1(self.in_planes, planes * block.expansion,
                                    stride=stride)
            dbn = builder.batchnorm(planes * block.expansion)
            if dbn is not None:
                downsample = nn.Sequential(dconv, dbn)
            else:
                downsample = dconv

        layers = []
        layers.append(block(builder, self.in_planes, planes, stride, downsample, base_width=self.base_width))
        self.in_planes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(builder, self.in_planes, planes, base_width=self.base_width))

        return nn.Sequential(*layers)

        layers = []
        layers.append(block(self.in_planes, planes, stride,downsample))
        self.in_planes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.in_planes, planes, stride,downsample))

        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.maxpool(out)
        out = self.shared_blocks(out)
        head_out_list = []
        for head in range(self.num_ensemble):
            head_out = getattr(self, f'blocks_head{head}')(out)
            head_out = F.avg_pool2d(head_out, 4)
            head_out = head_out.view(head_out.size(0), -1)
            head_out = getattr(self, f'fc_{head}')(head_out)
            head_out_list.append(head_out)

        if self.num_ensemble > 1:
            out = torch.stack(head_out_list, dim=1)
        else:
            out = head_out_list[0]

        return out


def ResNet18_imagenet(args):
    c = get_num_classes(args.data)
    return ResNet(BasicBlock, [2,2,2,2], c, args)

def ResNet34(args):
    c = get_num_classes(args.data)
    return ResNet(BasicBlock, [3,4,6,3], c, args)

def ResNet50_imagenet(args):
    c = get_num_classes(args.data)
    return ResNet(Bottleneck, [3,4,6,3], c, args)

def ResNet101(args):
    c = get_num_classes(args.data)
    return ResNet(Bottleneck, [3,4,23,3], c, args)

def ResNet152(args):
    c = get_num_classes(args.data)
    return ResNet(Bottleneck, [3,8,36,3], c, args)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    args.data = 'cifar10'
    args.num_ensemble = 3
    args.blocks_in_head = 2

    net = ResNet50_imagenet(args)
    y = net(torch.randn(1, 3, 32, 32))

    # print a nice overview of the network
    print(net)

    # print('\n\nTorch summary of the network:')
    # from torchinfo import summary
    # summary(net, input_size=(1, 3, 32, 32))
