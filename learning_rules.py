"""
Script defining various Hebbian learning rules
"""
import torch
import torch.nn.functional as F


def hebbs_rule(x, y, W):
    """
    return dW according to Hebb's rule: dW = y x^T
    """
    dW = y.unsqueeze(-1) * x.unsqueeze(-2)
    if dW.dim() > 2:
        dW = torch.mean(dW, 0)
    return dW


def ojas_rule(x, y, W):
    """
    return dW according to Oja's rule: dW_ij = y_i (x_j - y_i W_ij)
    """
    dW = y.unsqueeze(-1) * x.unsqueeze(-2) - (y**2).unsqueeze(-1) * W.unsqueeze(0)
    if dW.dim() > 2:
        dW = torch.mean(dW, 0)
    return dW


def hard_WTA(learning_rule):
    """
    decorator to add hard WTA to specified learning rule (i.e., only change weights of "winning" neuron)
    """
    def hard_WTA_learning_rule(x, y, W):

        # find winning neuron and create indicator tensor
        ind_win = torch.zeros_like(y)
        winners = torch.argmax(y, -1)
        ind_win = F.one_hot(winners, y.shape[-1]).float()

        # modify dW to only change weights of winning neuron
        dW = learning_rule(x, y, W)
        dW = ind_win.unsqueeze(-1) * dW.unsqueeze(0)
        if dW.dim() > 2:
            dW = torch.mean(dW, 0)

        return dW
    
    return hard_WTA_learning_rule


@hard_WTA
def hard_WTA_hebbs_rule(x, y, W):
    """
    hard WTA added to Hebb's rule
    """
    return hebbs_rule(x, y, W)


@hard_WTA
def hard_WTA_ojas_rule(x, y, W):
    """
    hard WTA added to Oja's rule
    """
    return ojas_rule(x, y, W)


def random_W(x, y, W):
    """
    return dW = 0 so that weights remain at random initialization (alt. baseline test)
    """
    dW = torch.zeros_like(W)
    return dW


learning_rules = {
    'hebbs_rule': hebbs_rule,
    'ojas_rule': ojas_rule,
    'hard_WTA_hebbs_rule': hard_WTA_hebbs_rule,
    'hard_WTA_ojas_rule': hard_WTA_ojas_rule,
    'random_W': random_W
}