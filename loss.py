import torch
import torch.nn as nn


def weighted_CrossEntropy(reduction):
    device = torch.device('cuda:0' if torch.cuda.is_available() else "cpu")
    pt = torch.tensor([0.0501, 0.0248, 0.0344, 0.0654, 0.0356, 0.1110, 0.0478, 0.1621, 0.0921,
                           0.2281, 0.0019, 0.1131, 0.0105, 0.0137, 0.0093], dtype=torch.float32)
    weight = (1/pt).to(device)
    loss = nn.CrossEntropyLoss(weight=weight, reduction=reduction)

    return loss

class WeightFocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, input, target):
        ce_criterion = weighted_CrossEntropy(reduction=self.reduction)
        ce_loss = ce_criterion(input, target)
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss

        return focal_loss

if __name__ == '__main__':
    # criterion = weighted_CrossEntropy()
    # input = torch.randn(3, 15, requires_grad=True)
    # target = torch.empty(3, dtype=torch.long).random_(15)
    # loss = criterion(input, target)
    # print(loss.item())

    focalLoss = WeightFocalLoss()
    input = torch.randn(3, 15, requires_grad=True)
    target = torch.empty(3, dtype=torch.long).random_(15)
    loss = focalLoss(input, target)
    print(loss.item())