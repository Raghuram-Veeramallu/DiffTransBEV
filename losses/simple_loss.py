import torch

class SimpleLoss(torch.nn.Module):
    def __init__(self, pos_weight, device='cpu'):
        super(SimpleLoss, self).__init__()
        self.loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=torch.Tensor([pos_weight]).to(device))

    def forward(self, ypred, ytgt):
        loss = self.loss_fn(ypred, ytgt)
        return loss
