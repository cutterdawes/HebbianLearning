"""
Script defining Hebbian learning rules
"""
import torch
import torch.nn.functional as F


class Plasticity:
    def __init__(self, rule: str
    ) -> None:
        """
        Plasticity component of learning rule
        """
        possible_rules = ['hebbs_rule', 'ojas_rule', 'random_W']
        if rule not in possible_rules:
            raise ValueError(f'Argument rule={rule} must be among {possible_rules}')
        self.rule = rule

    def __call__(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        W: torch.Tensor
    ) -> torch.Tensor:
        # compute dW according to specified rule
        if self.rule == 'hebbs_rule':  # dW = y x^T
            dW = y.unsqueeze(-1) * x.unsqueeze(-2)

        elif self.rule == 'ojas_rule':  # dW_ij = y_i (x_j - y_i W_ij)
            dW = y.unsqueeze(-1) * x.unsqueeze(-2) - (y**2).unsqueeze(-1) * W.unsqueeze(0)

        elif self.rule == 'random_W':  # dW = 0 (alt. baseline test)
            dW = torch.zeros_like(W)

        # take mean if computed over batch
        if dW.dim() > 2:
            dW = torch.mean(dW, 0)

        return dW
    
    def __str__(self) -> str:
        return f'Plasticity:\nrule={self.rule}'


class WTA:
    def __init__(
        self,
        rule: str = 'none',
        K: int = 1,
        delta: float = 0.4,
        temp: float = 1
    ) -> None:
        """
        Competitive component of learning rule
        """
        possible_rules = ['hard', 'soft', 'none']
        if rule not in possible_rules:
            raise ValueError(f'Argument rule={rule} must be among {possible_rules}')
        self.rule = rule
        self.K = K
        self.delta = delta
        self.temp = temp

    def __call__(self, y: torch.Tensor) -> torch.Tensor:
        # compute wta according to specified rule
        if self.rule == 'hard':
            winner = torch.argmax(y, -1)
            wta = F.one_hot(winner, y.shape[-1]).float()
            if self.K > 1:
                _, topK = torch.topk(y, self.K, -1)
                two_to_K = topK[1:] if topK.dim() == 1 else topK[:,1:]
                wta -= self.delta * torch.sum(F.one_hot(two_to_K, y.shape[-1]), -2)

        if self.rule == 'soft':
            winner = torch.argmax(y, -1)
            wta = F.one_hot(winner, y.shape[-1]).float()
            wta = 2 * wta - torch.ones_like(wta)
            wta *= torch.softmax(self.temp * y, -1)  # following implementation of SoftHebb
        
        return wta
    
    def __str__(self) -> str:
        return f'WTA:\nrule={self.rule},\nK={self.K}, delta={self.delta}, temp={self.temp}'


class LearningRule:
    def __init__(
        self,
        plasticity: str = 'hebbs_rule',
        wta: str = 'none',
        **kwargs
    ) -> None:
        """
        Customizable Hebbian learning rule that includes customizable WTA and plasticity components
        """
        # pass appropriate kwargs to components
        plasticity_kwargs = {k: kwargs[k] for k in kwargs if k in {}}
        wta_kwargs = {k: kwargs[k] for k in kwargs if k in {'K', 'delta'}}

        # set plasticity and competitive components
        self.plasticity = Plasticity(plasticity, **plasticity_kwargs)
        self.wta = WTA(wta, **wta_kwargs)

    def __call__(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        W: torch.Tensor
    ) -> torch.Tensor:
        # compute weight change (dW) according to plasticity rule
        dW = self.plasticity(x, y, W)

        # optionally add competitive component (WTA)
        if self.wta.rule != 'none':
            wta = self.wta(y)
            dW = wta.unsqueeze(-1) * dW.unsqueeze(0)

        # take mean if computed over batch
        if dW.dim() > 2:
            dW = torch.mean(dW, 0)

        return dW
