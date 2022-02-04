import torch
import torch.nn.functional as F

torch.manual_seed(1)

z = torch.FloatTensor([1, 2, 3])
hypothesis = F.softmax(z, dim=0)

z = torch.rand(3, 5, requires_grad=True)
hypothesis = F.softmax(z, dim=1)

y = torch.randint(5, (3,)).long()
# one-hot incoding
y_one_hot = torch.zeros_like(hypothesis) 
y_one_hot.scatter_(1, y.unsqueeze(1), 1)

# 비용
cost = (y_one_hot * -torch.log(hypothesis)).sum(dim=1).mean()