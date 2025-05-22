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

    def __init__(self, in_planes, planes, stride=1):
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


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion*planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes,multiplier, args):
        super(ResNet, self).__init__()
        self.in_planes = 64
        self.num_ensemble = args.num_ensemble
        self.num_classes = num_classes
        self.blocks_in_head = args.blocks_in_head
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
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
            superblocks.extend(self._make_layer(block, round(64*multiplier), num_blocks[0], stride=1))
            superblocks.extend(self._make_layer(block, round(128*multiplier), num_blocks[1], stride=2))
            superblocks.extend(self._make_layer(block, round(256*multiplier), num_blocks[2], stride=2))
            superblocks.extend(self._make_layer(block, round(512*multiplier), num_blocks[3], stride=2))

            if head != self.num_ensemble:  # rm the shared blocks that are not needed in the heads
                for block_num in range(self.blocks_shared):
                    del superblocks[0]
                setattr(self, f'blocks_head{head}', nn.Sequential(*superblocks))
            else:
                for block_num in range(self.blocks_in_head):
                    del superblocks[-1]
                setattr(self, f'shared_blocks', nn.Sequential(*superblocks))

        for head in range(self.num_ensemble):
            setattr(self, f'fc_{head}', nn.Linear(round(512*block.expansion*multiplier), self.num_classes, bias=False))

    def _make_layer(self, block, planes, num_blocks, stride):

        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))

            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
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


def ResNet18(args):
    c = get_num_classes(args.data)
    return ResNet(BasicBlock, [2,2,2,2], c,args.multiplier, args)

def ResNet34(args):
    c = get_num_classes(args.data)
    return ResNet(BasicBlock, [3,4,6,3], c,args.multiplier, args)

def ResNet50(args):
    c = get_num_classes(args.data)
    return ResNet(Bottleneck, [3,4,6,3], c,args.multiplier, args)

def ResNet101(args):
    c = get_num_classes(args.data)
    return ResNet(Bottleneck, [3,4,23,3], c,args.multiplier, args)

def ResNet152(args):
    c = get_num_classes(args.data)
    return ResNet(Bottleneck, [3,8,36,3], c,args.multiplier, args)