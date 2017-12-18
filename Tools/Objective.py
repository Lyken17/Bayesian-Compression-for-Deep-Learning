import torch.nn as nn

class objection(object):
    def __init__(self, data_loader, use_cuda=True):
        self.d_loss = nn.functional.cross_entropy
        self.N = 0
        if isinstance(data_loader, list):
            self.N += len(data_loader)
        else:
            self.N = len(data_loader)
        self.use_cuda = use_cuda

    def __call__(self, output, target, kl_divergence):
        d_error = self.d_loss(output, target)
        variational_bound = d_error + kl_divergence / self.N  # TODO: why divide by N?
        if self.use_cuda:
            variational_bound = variational_bound.cuda()
        return variational_bound