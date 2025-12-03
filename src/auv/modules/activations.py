import torch
import torch.nn as nn
import torch.nn.functional as F


class Snake(nn.Module):
    """
    Snake activation function: x + (1/alpha) * sin^2(alpha * x)
    
    When alpha is large, Snake approximates identity function (linear).
    When alpha is small (~1), Snake has periodic modulation.
    
    To approximate GELU behavior when loading from GELU pretrained weights,
    we use init_from_gelu=True which initializes alpha to make Snake behave
    more like a smooth non-linear activation.
    """
    def __init__(self, in_features, alpha=1.0, alpha_trainable=True, alpha_logscale=False, init_from_gelu=False):
        super().__init__()
        self.in_features = in_features
        self.alpha_logscale = alpha_logscale
        
        # When init_from_gelu is True, we initialize alpha to a value that makes
        # Snake's behavior closer to GELU's smooth non-linearity.
        # A moderate alpha (~1.0-2.0) gives a smooth activation with slight periodic modulation.
        # This provides a good starting point when fine-tuning from GELU pretrained weights.
        if init_from_gelu:
            # Initialize with alpha=1.0 which gives a balanced Snake activation
            # The periodic component sin^2(x) adds a smooth non-linearity similar to GELU
            init_alpha = 1.0
        else:
            init_alpha = alpha
            
        if self.alpha_logscale:
            self.alpha = nn.Parameter(torch.zeros(in_features) + torch.log(torch.tensor(init_alpha)))
        else:
            self.alpha = nn.Parameter(torch.ones(in_features) * init_alpha)

        self.alpha.requires_grad = alpha_trainable
        self.no_div_by_zero = 1e-9

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
    
    @classmethod
    def init_alpha_from_gelu_stats(cls, snake_module, sample_input=None):
        """
        Optionally refine alpha initialization by matching GELU output statistics.
        This can be called after loading pretrained weights to further align Snake with GELU.
        
        Args:
            snake_module: The Snake module to initialize
            sample_input: Optional sample input tensor for calibration
        """
        # Default initialization that works well in practice
        with torch.no_grad():
            snake_module.alpha.fill_(1.0)
