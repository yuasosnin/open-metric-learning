from abc import ABC, abstractmethod

import torch
from torch import Tensor
import torch.nn as nn


class IDistance(nn.Module, ABC):
    """
    A base interface for difference distance metrics 
    that implement calculating elementwise and pairwise (matrix) distances.

    """

    def forward(self, x1: Tensor, x2: Tensor) -> Tensor:
        return self.elementwise(x1, x2)
    
    @abstractmethod
    def elementwise(self, x1: Tensor, x2: Tensor) -> Tensor:
        """
        Args:
            x1: tensor with the shape of [N, D]
            x2: tensor with the shape of [N, D]

        Returns: elementwise distances with the shape of [N]

        """
        raise NotImplementedError()
    
    def pairwise(self, x1: Tensor, x2: Tensor) -> Tensor:
        """
        Calculates distance matrix between two batched inputs.
        Reimplement this method if there is a more efficient way
        to do it than a loop.

        Args:
            x1: tensor with the shape of [N, D]
            x2: tensor with the shape of [M, D]

        Returns: pairwise distances with the shape of [N, M]

        """
        n = x1.shape[0]
        m = x2.shape[0]
        inner = torch.empty((n, m), device=x1.device)
        for i in range(n):
            x1_i = x1[i].unsqueeze(0)
            inner[i, :] = self.elementwise(x1_i, x2)
        return inner
