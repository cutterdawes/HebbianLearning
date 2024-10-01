"""
Script defining Hebbian learning rules
"""
from typing import Any
import torch
import torch.nn.functional as F


class HebbsRule:
    def __call__(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        W: torch.Tensor
    ) -> torch.Tensor:
        dW = y.unsqueeze(-1) * x.unsqueeze(-2)  # dW = y x^T
        return dW


class OjasRule:
    def __call__(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        W: torch.Tensor
    ) -> torch.Tensor:
        dW = y.unsqueeze(-1) * (x.unsqueeze(-2) - y.unsqueeze(-1) * W.unsqueeze(0))  # dW_ij = y_i (x_j - y_i W_ij)
        return dW


class HardWTA:
    def __init__(
            self,
            K: int = 1,
            delta: float = 0.4
    ) -> None:
        self.K = K
        self.delta = delta

    def __call__(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        W: torch.Tensor
    ) -> torch.Tensor:
        
        # compute plasticity rule
        dW = y.unsqueeze(-1) * x.unsqueeze(-2) - (y**2).unsqueeze(-1) * W.unsqueeze(0)

        # compute winner and Kth most activated
        winner = torch.argmax(y, -1)
        wta = F.one_hot(winner, y.shape[-1]).float()
        if self.K > 1:
            _, topK = torch.topk(y, self.K, -1)
            # two_to_K = topK[1:] if topK.dim() == 1 else topK[:,1:]
            # wta -= self.delta * torch.sum(F.one_hot(two_to_K, y.shape[-1]), -2)
            Kth = topK[-1] if topK.dim() == 1 else topK[:,-1]
            wta -= self.delta * F.one_hot(Kth, y.shape[-1])

        # modify dW according to WTA
        dW *= wta.unsqueeze(-1)
        
        return dW


class SoftWTA:
    def __init__(
            self,
            temp: float = 1
    ) -> None:
        self.temp = temp

    def __call__(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        W: torch.Tensor
    ) -> torch.Tensor:

        # compute softmaxed activations (anti-Hebbian for losers)
        winner = torch.argmax(y, -1)
        wta = F.one_hot(winner, y.shape[-1]).float()
        wta = 2 * wta - torch.ones_like(wta)
        y_soft = wta * torch.softmax(self.temp * y, -1)

        # compute SoftHebb update
        dW = y_soft.unsqueeze(-1) * (x.unsqueeze(-2) - y.unsqueeze(-1) * W.unsqueeze(0))
        # import pdb; pdb.set_trace()

        return dW
    

class RandomW:
    def __call__(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        W: torch.Tensor
    ) -> torch.Tensor:
        dW = torch.zeros_like(W)
        return dW


class LearningRule:
    def __init__(
        self,
        rule: str,
        **kwargs
    ) -> None:
        rules = {
            'hebbs_rule': HebbsRule,
            'ojas_rule': OjasRule,
            'hard_WTA': HardWTA,
            'soft_WTA': SoftWTA,
            'random_W': RandomW
        }
        if rule not in rules.keys():
            raise ValueError(f'Argument rule={rule} must be among {rules.keys()}')
        self.rule = rules[rule](**kwargs)

    def __call__(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        W: torch.Tensor
    ) -> torch.Tensor:
        
        # compute specified learning rule
        dW = self.rule(x, y, W)

        # take mean if computed over batch
        if dW.dim() > 2:
            dW = torch.mean(dW, 0)

        return dW


if __name__ == "__main__":
    rule = HardWTA(K=2)
    W = torch.randn(5, 3)
    x = torch.randn(3)
    y = torch.matmul(x, W.T)
    rule(x, y, W)