import torch
import torch.nn as nn


class Snake(nn.Module):
    def __init__(self, in_features, alpha=1.0, alpha_trainable=True, alpha_logscale=False):
        super().__init__()
        self.in_features = in_features

        # initialize alpha
        self.alpha_logscale = alpha_logscale
        if self.alpha_logscale:  # log scale alphas initialized to 0
            self.alpha = nn.Parameter(torch.zeros(in_features) * alpha)
        else:  # linear scale alphas initialized to alpha
            self.alpha = nn.Parameter(torch.ones(in_features) * alpha)

        self.alpha.requires_grad = alpha_trainable

        self.no_div_by_zero = 0.000000001

    def forward(self, x):
        if x.shape[-1] == self.in_features:
            alpha = self.alpha
        elif x.shape[1] == self.in_features:
            shape = [1] * x.dim()
            shape[1] = self.in_features
            alpha = self.alpha.view(shape)
        else:
            raise ValueError(f"Input shape {x.shape} does not match alpha shape {self.in_features}")

        if self.alpha_logscale:
            alpha = torch.exp(alpha)
        x = x + (1.0 / (alpha + self.no_div_by_zero)) * torch.pow(torch.sin(x * alpha), 2)

        return x
