from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torchvision import datasets, transforms

from compression import compute_compression_rate, compute_reduced_weights
from Bayesian import BayesianModule

N = 60000.  # number of data points in the training set


def main(FLAGS):
    # import data
    kwargs = {'num_workers': 1, 'pin_memory': True} if FLAGS.cuda else {}

    if FLAGS.dataset == "cifar10":
        proj_dst = datasets.CIFAR10
        num_classes = 10
    elif FLAGS.dataset == "cifar100":
        proj_dst = datasets.CIFAR100
        num_classes = 100
    elif FLAGS.dataset == "mnist":
        proj_dst = datasets.MNIST
        num_classes = 10

    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=True, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           lambda x: 2 * (x - 0.5),
                       ])),
        batch_size=FLAGS.batchsize, shuffle=True, **kwargs)

    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=False, transform=transforms.Compose([
            transforms.ToTensor(),
            lambda x: 2 * (x - 0.5),
        ])),
        batch_size=FLAGS.batchsize, shuffle=True, **kwargs)

    if FLAGS.dataset.startswith("cifar"):
        if FLAGS.nettype == "lenet":
            model = BayesianModule.LeNet_Cifar(num_classes)
        elif FLAGS.nettype == "mlp":
            model = BayesianModule.MLP_Cifar(num_classes)
    elif FLAGS.dataset == "mnist":
        if FLAGS.nettype == "lenet":
            model = BayesianModule.LeNet_MNIST(num_classes)
        elif FLAGS.nettype == "mlp":
            model = BayesianModule.MLP_MNIST(num_classes)

    print(FLAGS.dataset, FLAGS.nettype)
    if FLAGS.cuda:
        model.cuda()

    # init optimizer
    optimizer = optim.Adam(model.parameters())

    # we optimize the variational lower bound scaled by the number of data
    # points (so we can keep our intuitions about hyper-params such as the learning rate)
    discrimination_loss = nn.functional.cross_entropy

    class objection(object):
        def __init__(self, N, use_cuda=True):
            self.d_loss = nn.functional.cross_entropy
            self.N = N
            self.use_cuda = use_cuda

        def __call__(self, output, target, kl_divergence):
            d_error = self.d_loss(output, target)
            variational_bound = d_error + kl_divergence / self.N  # TODO: why divide by N?
            if self.use_cuda:
                variational_bound = variational_bound.cuda()
            return variational_bound

    objective = objection(len(train_loader.dataset))

    from trainer import Trainer
    trainer = Trainer(model, train_loader, test_loader, optimizer, objective)
    # train the model and save some visualisations on the way
    for epoch in range(1, FLAGS.epochs + 1):
        trainer.train(epoch)
        trainer.test()

    # compute compression rate and new model accuracy
    layers = model.layers
    thresholds = FLAGS.thresholds
    compute_compression_rate(layers, model.get_masks(thresholds))

    print("Test error after with reduced bit precision:")

    weights = compute_reduced_weights(layers, model.get_masks(thresholds))
    for layer, weight in zip(layers, weights):
        if FLAGS.cuda:
            layer.post_weight_mu.data = torch.Tensor(weight).cuda()
        else:
            layer.post_weight_mu.data = torch.Tensor(weight)

    for layer in layers:
        layer.deterministic = True
    trainer.test()


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batchsize', type=int, default=256)
    parser.add_argument('--thresholds', type=float, nargs='*', default=[-2.8, -3., -5., ])

    parser.add_argument('--dataset', type=str, choices=["cifar10", "cifar100", "mnist"], default="mnist")
    parser.add_argument('--nettype', type=str, choices=["mlp", "lenet"], default="mlp")

    FLAGS = parser.parse_args()
    FLAGS.cuda = torch.cuda.is_available()  # check if we can put the net on the GPU

    main(FLAGS)
