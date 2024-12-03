import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torch.nn.parallel import DistributedDataParallel as DDP

from typing import *

def init_model(device : torch.device, args):
    '''
    Load the correct model for each dataset.
    '''
    if args.exp_name == 'SupCon':
        if hasattr(args, 'world_size') and args.world_size > 1:
            model = DDP(SupConResNet(args).to(device))
        else:
            model = SupConResNet(args).to(device)
    elif args.exp_name == 'ERM':
        if hasattr(args, 'world_size') and args.world_size > 1:
            model = DDP(SupCEResNet(args)).to(device)
        else:
            model = SupCEResNet(args).to(device)
    model = torch.compile(model)
    return model
    
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, is_last=False):
        super(BasicBlock, self).__init__()
        self.is_last = is_last
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(
            planes, planes, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_planes,
                    self.expansion * planes,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(self.expansion * planes),
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        preact = out
        out = F.relu(out)
        if self.is_last:
            return out, preact
        else:
            return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1, is_last=False):
        super(Bottleneck, self).__init__()
        self.is_last = is_last
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(
            planes, planes, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(
            planes, self.expansion * planes, kernel_size=1, bias=False
        )
        self.bn3 = nn.BatchNorm2d(self.expansion * planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_planes,
                    self.expansion * planes,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(self.expansion * planes),
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        preact = out
        out = F.relu(out)
        if self.is_last:
            return out, preact
        else:
            return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, in_channel=3, zero_init_residual=False):
        super(ResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(
            in_channel, 64, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves
        # like an identity. This improves the model by 0.2~0.3% according to:
        # https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for i in range(num_blocks):
            stride = strides[i]
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x, layer=100):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        return out


class SupCEResNet(nn.Module):
    """encoder + classifier"""
    def __init__(self, args):
        super(SupCEResNet, self).__init__()
        model_fun, dim_in = model_dict[args.model_name]
        self.args = args
        self.encoder = model_fun()
        self.fc = nn.Linear(dim_in, args.n_cls)
        self.normalize = args.normalize

    def forward(self, x):
        features = self.encoder(x)
        if self.normalize: 
            features =  F.normalize(features, dim=1)
        return features, self.fc(features)

class SupConResNet(nn.Module):
    """encoder + head"""
    def __init__(self, args, multiplier = 1):
        super(SupConResNet, self).__init__()
        model_fun, dim_in = model_dict[args.model_name]
        self.args = args
        if args.dataset == 'IMAGENET100':
            # fine-tune
            if args.model_name == 'resnet18':
                model = models.resnet18(pretrained=True)
            elif args.model_name == 'resnet34':
                model = models.resnet34(pretrained=True)
            elif args.model_name == 'resnet50':
                model = models.resnet50(pretrained=True)
            elif args.model_name == 'resnet101':
                model = models.resnet101(pretrained=True)
            for name, p in model.named_parameters():
                if not name.startswith('layer4'):
                    p.requires_grad = False
            modules = list(model.children())[:-1] # remove last linear layer
            self.encoder = nn.Sequential(*modules)
        else:
            self.encoder = model_fun()
        self.fc = nn.Linear(dim_in, args.n_cls)
        self.multiplier = multiplier
        self.head = nn.Sequential(
            nn.Linear(dim_in, dim_in),
            nn.ReLU(inplace=True),
            nn.Linear(dim_in, 128)
        )
    
    
    def forward(self, x):
        feat = self.encoder(x).squeeze()
        if self.args.normalize == 0: # official SupCon codebase
            # proj-L2
            unnorm_features = self.head(feat)
            features = F.normalize(unnorm_features, dim=1) 
        elif self.args.normalize == 1:
            # feat-L2-proj-L2
            feat = F.normalize(feat, dim=1) # Following paper setting, normalize twice
            unnorm_features = self.head(feat)
            features = F.normalize(unnorm_features, dim=1)
        elif self.args.normalize == 2:
            # proj-max
            unnorm_features = self.head(feat)
            norms = torch.norm(unnorm_features, dim=1)
            features = unnorm_features/torch.max(norms) # learn on unit disk for projection head
        elif self.args.normalize == 3:
            # none
            unnorm_features = self.head(feat)
            features = unnorm_features
        elif self.args.normalize == 4:
            # feat-L2
            feat = F.normalize(feat, dim=1)
            unnorm_features = self.head(feat)
            features = unnorm_features
        elif self.args.normalize == 5:
            # feat-L2-proj-max
            feat = F.normalize(feat, dim=1)
            unnorm_features = self.head(feat)
            norms = torch.norm(unnorm_features, dim=1)
            features = unnorm_features/torch.max(norms)
        elif self.args.normalize == 6:
            # feat-max
            feat_norms = torch.norm(feat, dim=1)
            feat = feat/torch.max(feat_norms)
            unnorm_features = self.head(feat)
            features = unnorm_features
        elif self.args.normalize == 7:
            # feat-max-proj-L2
            feat_norms = torch.norm(feat, dim=1)
            feat = feat/torch.max(feat_norms)
            unnorm_features = self.head(feat)
            features = F.normalize(unnorm_features, dim=1)
        elif self.args.normalize == 8:
            # feat-max-proj-max
            feat_norms = torch.norm(feat, dim=1)
            feat = feat/torch.max(feat_norms)
            unnorm_features = self.head(feat)
            norms = torch.norm(unnorm_features, dim=1)
            features = unnorm_features/torch.max(norms)

        return feat, features # first is unnormalized feat, second is projected+normalize result

class LinearClassifier(nn.Module):
    """Linear classifier"""
    def __init__(self, name='resnet50', num_classes=10):
        super(LinearClassifier, self).__init__()
        _, feat_dim = model_dict[name]
        self.fc = nn.Linear(feat_dim, num_classes)

    def forward(self, features):
        return self.fc(features)
    
class MLPClassifier(nn.Module):
    """MLP classifier"""
    def __init__(self, name='resnet50', num_classes=10):
        super(MLPClassifier, self).__init__()
        _, feat_dim = model_dict[name]
        hidden_dim = 128
        self.mlp = nn.Sequential(nn.Linear(feat_dim, hidden_dim),
                                  nn.ReLU(),
                                  nn.Linear(hidden_dim, num_classes))

    def forward(self, features):
        return self.mlp(features)
    



def resnet18(**kwargs):
    return ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)


def resnet34(**kwargs):
    return ResNet(BasicBlock, [3, 4, 6, 3], **kwargs)


def resnet50(**kwargs):
    return ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)


def resnet101(**kwargs):
    return ResNet(Bottleneck, [3, 4, 23, 3], **kwargs)


model_dict = {
    'resnet18': [resnet18, 512],
    'resnet34': [resnet34, 512],
    'resnet50': [resnet50, 2048],
    'resnet101': [resnet101, 2048],
}
