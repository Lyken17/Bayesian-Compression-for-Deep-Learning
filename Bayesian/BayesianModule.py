from __future__ import print_function

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.autograd import Variable

from Bayesian import BayesianLayers


class BayesianModule(nn.Module):
    def __init__(self):
        self.kl_list = []
        self.layers = []
        super(BayesianModule, self).__init__()

    def __setattr__(self, name, value):
        super(BayesianModule, self).__setattr__(name, value)
        # simple hack to collect bayesian layer automatically
        if isinstance(value, BayesianLayers.BayesianLayers) and not isinstance(value, BayesianLayers._ConvNdGroupNJ):
            self.kl_list.append(value)
            self.layers.append(value)

    def get_masks(self, thresholds):
        weight_masks = []
        mask = None
        for i, (layer, threshold) in enumerate(zip(self.kl_list, thresholds)):
            # compute dropout mask
            if mask is None:
                log_alpha = layer.get_log_dropout_rates().cpu().data.numpy()
                mask = log_alpha < threshold
            else:
                mask = np.copy(next_mask)

            try:
                log_alpha = self.layers[i + 1].get_log_dropout_rates().cpu().data.numpy()
                next_mask = log_alpha < thresholds[i + 1]
            except:
                # must be the last mask
                next_mask = np.ones(10)

            weight_mask = np.expand_dims(mask, axis=0) * np.expand_dims(next_mask, axis=1)
            weight_masks.append(weight_mask.astype(np.float))
        return weight_masks

    def kl_divergence(self):
        KLD = 0
        for layer in self.kl_list:
            KLD += layer.kl_divergence()
        return KLD


class MLP_Cifar(BayesianModule):
    def __init__(self, num_classes=10, use_cuda=True):
        super(MLP_Cifar, self).__init__()

        self.fc1 = BayesianLayers.LinearGroupNJ(3 * 32 * 32, 300, clip_var=0.04, cuda=use_cuda)
        self.fc2 = BayesianLayers.LinearGroupNJ(300, 100, cuda=use_cuda)
        self.fc3 = BayesianLayers.LinearGroupNJ(100, num_classes, cuda=use_cuda)

    def forward(self, x):
        x = x.view(-1, 3 * 32 * 32)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class MLP_MNIST(BayesianModule):
    def __init__(self, num_classes=10, use_cuda=True):
        super(MLP_MNIST, self).__init__()

        self.fc1 = BayesianLayers.LinearGroupNJ(28 * 28, 300, clip_var=0.04, cuda=use_cuda)
        self.fc2 = BayesianLayers.LinearGroupNJ(300, 100, cuda=use_cuda)
        self.fc3 = BayesianLayers.LinearGroupNJ(100, num_classes, cuda=use_cuda)

    def forward(self, x):
        x = x.view(-1, 28 * 28)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class LeNet_Cifar(BayesianModule):
    def __init__(self, num_classes=10, use_cuda=True):
        super(LeNet_Cifar, self).__init__()

        self.conv1 = BayesianLayers.Conv2dGroupNJ(3, 6, 5, cuda=use_cuda)
        self.conv2 = BayesianLayers.Conv2dGroupNJ(6, 16, 5, cuda=use_cuda)

        self.fc1 = BayesianLayers.LinearGroupNJ(16 * 5 * 5, 120, clip_var=0.04, cuda=use_cuda)
        self.fc2 = BayesianLayers.LinearGroupNJ(120, 84, cuda=use_cuda)
        self.fc3 = BayesianLayers.LinearGroupNJ(84, num_classes, cuda=use_cuda)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        out = x.view(x.size(0), -1)
        out = F.relu(self.fc1(out))
        out = F.relu(self.fc2(out))
        out = self.fc3(out)
        return out


class LeNet_MNIST(BayesianModule):
    def __init__(self, num_classes=10, use_cuda=True):
        super(LeNet_MNIST, self).__init__()

        self.conv1 = BayesianLayers.Conv2dGroupNJ(1, 10, 5, cuda=use_cuda)
        self.conv2 = BayesianLayers.Conv2dGroupNJ(10, 20, 5, cuda=use_cuda)

        self.fc1 = BayesianLayers.LinearGroupNJ(320, 50, clip_var=0.04, cuda=use_cuda)
        self.fc2 = BayesianLayers.LinearGroupNJ(50, num_classes, cuda=use_cuda)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        out = x.view(x.size(0), -1)
        out = F.relu(self.fc1(out))
        out = self.fc2(out)
        return out


if __name__ == "__main__":
    net = LeNet_Cifar(use_cuda=False)
    data = torch.randn([1, 3, 32, 32])
    data = Variable(data)
    print(net(data).size())
