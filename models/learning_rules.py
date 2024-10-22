"""
Script defining Hebbian learning rules
"""
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
        Wx = torch.matmul(x, W.T)
        dW = y.unsqueeze(-1) * (x.unsqueeze(-2) - Wx.unsqueeze(-1) * W.unsqueeze(0))  # dW_ij = y_i (x_j - (Wx)_i W_ij)
        return dW


class HardWTA:
    def __init__(
            self,
            N_hebb: int = 1,
            N_anti: int = 0,
            K_anti: int = 1,
            delta: float = 0.4,
    ) -> None:
        if N_hebb >= K_anti and N_anti > 0:
            raise ValueError('Invalid arguments N_hebb={H_hebb}, N_anti={N_anti}, K_anti={K_anti}')
        self.N_hebb = N_hebb
        self.N_anti = N_anti
        self.K_anti = K_anti
        self.delta = delta

    def __call__(
            self,
            x: torch.Tensor,
            y: torch.Tensor,
            W: torch.Tensor
    ) -> torch.Tensor:
        
        # compute Hebb's rule and first-order normalization
        Wx = torch.matmul(x, W.T)
        dW_hebb = y.unsqueeze(-1) * x.unsqueeze(-2)
        dW_norm = Wx.unsqueeze(-1) * W.unsqueeze(0)

        # compute neurons to be updated (Hebbian and anti-Hebbian)
        if self.N_hebb > 1:
            _, hebb = torch.topk(y, self.N_hebb, -1)
            hebb = torch.sum(F.one_hot(hebb, y.shape[-1]), -2)
        else:
            hebb = torch.argmax(y, -1)
            hebb = F.one_hot(hebb, y.shape[-1]).float()

        if self.N_anti > 0:
            _, anti = torch.topk(y, self.K_anti+self.N_anti, -1)
            if self.N_anti > 1:
                anti = anti[self.K_anti:] if anti.dim() == 1 else anti[:,self.K_anti:]
                anti = torch.sum(F.one_hot(anti, y.shape[-1]), -2)
            elif self.N_anti == 1:
                anti = anti[-1] if anti.dim() == 1 else anti[:,-1]
                anti = F.one_hot(anti, y.shape[-1])
        else:
            anti = torch.zeros_like(hebb)
        wta_hebb = hebb - self.delta * anti
        wta_norm = hebb + self.delta * anti

        # modify dW according to WTA
        dW = wta_hebb.unsqueeze(-1) * dW_hebb - wta_norm.unsqueeze(-1) * dW_norm

        return dW


class SoftWTA:
    def __init__(
            self,
            temp: float = 1000,
            beta: float = 1
    ) -> None:
        self.temp = temp
        self.beta = beta  # TODO: revert to time-independent version
        
        # initialize time-dependent variables
        self.x_prev = torch.tensor(0)
        self.x_mem = torch.tensor(0)

    def __call__(
            self,
            x: torch.Tensor,
            y: torch.Tensor,
            W: torch.Tensor
    ) -> torch.Tensor:
        
        if isinstance(self.beta, float):
            # sample beta for each neuron uniformly from [0, beta]
            self.beta = self.beta * torch.rand(x.shape[-1])

        try:
            # update time dependent variables
            x_dot = x - self.x_prev
            self.x_prev = x
            self.x_mem = x_dot + self.beta * self.x_mem
        except:  # handles "leftovers" of batch
            return torch.zeros_like(W)

        # compute softmaxed activations (anti-Hebbian for losers)
        winner = torch.argmax(y, -1)
        wta = F.one_hot(winner, y.shape[-1]).float()
        wta = 2 * wta - torch.ones_like(wta)
        y_soft = wta * torch.softmax(self.temp * y, -1)
        
        # compute SoftHebb update
        Wx = torch.matmul(self.x_mem, W.T)
        dW = y_soft.unsqueeze(-1) * (self.x_mem.unsqueeze(-2) - Wx.unsqueeze(-1) * W.unsqueeze(0))
    
        return dW
    

class STDP:
    def __init__(
            self,
            beta: float = 1.0
    ) -> None:
        self.beta = beta

        # initialize time-dependent variables
        self.x_prev = torch.tensor(0)
        self.x_mem = torch.tensor(0)

    def __call__(
            self,
            x: torch.Tensor,
            y: torch.Tensor,
            W: torch.Tensor
    ) -> torch.Tensor:
        
        if isinstance(self.beta, float):
            # sample beta for each neuron uniformly from [0, beta]
            self.beta = self.beta * torch.rand(x.shape[-1])

        try:
            # update time dependent variables
            x_dot = x - self.x_prev
            self.x_prev = x
            self.x_mem = x_dot + self.beta * self.x_mem
        except:  # handles "leftovers" of batch
            return torch.zeros_like(W)
        
        # compute STDP rule
        Wx = torch.matmul(self.x_mem, W.T)
        dW = y.unsqueeze(-1) * (self.x_mem.unsqueeze(-2) - Wx.unsqueeze(-1) * W.unsqueeze(0))
        # dW += 0.3 * (y.max() - y).unsqueeze(-1) * torch.randn_like(W)

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
            'STDP': STDP,
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