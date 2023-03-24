import torch
from torch import Tensor

from oml.interfaces.distances import IDistance


class EucledianDistance(IDistance):
    """"
    Default Eucledian norm distance. Basically torch.cdist.

    """

    def __init__(self, p: float = 2):
        """
        Args:
            p: p-norm to calculate metric with
        
        """
        self.p = p

    def elementwise(self, x1: Tensor, x2: Tensor) -> Tensor:
        """
        Args:
            x1: tensor with the shape of [N, D]
            x2: tensor with the shape of [N, D]

        Returns: elementwise distances with the shape of [N]

        """
        assert len(x1.shape) == len(x2.shape) == 2
        assert x1.shape == x2.shape

        # we need an extra dim here to avoid pairwise behaviour of torch.cdist
        if len(x1.shape) == 2:
            x1 = x1.unsqueeze(1)
            x2 = x2.unsqueeze(1)

        return torch.cdist(x1, x2, p=self.p).view(len(x1))

    def pairwise(self, x1: Tensor, x2: Tensor) -> Tensor:
        """
        Args:
            x1: tensor with the shape of [N, D]
            x2: tensor with the shape of [M, D]

        Returns: pairwise distances with the shape of [N, M]

        """
        assert len(x1.shape) == len(x2.shape) == 2
        assert x1.shape[-1] == x2.shape[-1]

        return torch.cdist(x1, x2, p=self.p)
