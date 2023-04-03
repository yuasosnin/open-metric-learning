from abc import ABC, abstractmethod

from torch import Tensor
import torch.nn as nn


class IDistance(nn.Module, ABC):
    """
    A base interface for difference distance metrics 
    that implement calculating elementwise and pairwise (matrix) distances.

    """

    @abstractmethod
    def elementwise(self, x1: Tensor, x2: Tensor) -> Tensor:
        """
        Args:
            x1: tensor with the shape of [N, D]
            x2: tensor with the shape of [N, D]

        Returns: elementwise distances with the shape of [N]

        """
        raise NotImplementedError()
    
    @abstractmethod
    def pairwise(self, x1: Tensor, x2: Tensor) -> Tensor:
        """
        Args:
            x1: tensor with the shape of [N, D]
            x2: tensor with the shape of [M, D]

        Returns: pairwise distances with the shape of [N, M]

        """
        raise NotImplementedError()

    def forward(self, x1: Tensor, x2: Tensor) -> Tensor:
        return self.elementwise(x1, x2)
