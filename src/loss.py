import torch
import torch.nn as nn

class DiceLoss(nn.Module):  
    def __init__(self):
        super(DiceLoss, self).__init__()
        self.epsilon = 1e-6
    def forward(self, pred, target):
        numerator = 2 * torch.sum(target * pred) + self.epsilon
        denominator = torch.sum(torch.square(target)) + torch.sum(torch.square(pred)) + self.epsilon
        return 1 - (numerator / denominator)

def get_loss_func(loss_name):
    if loss_name == 'dice':
        return DiceLoss()
    elif loss_name == 'ce':
        return nn.CrossEntropyLoss()
    else:
        raise NotImplementedError