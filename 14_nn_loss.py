import torch
from torch import nn

input = torch.tensor([1, 2, 3], dtype=torch.float32)
target = torch.tensor([1, 2, 6], dtype=torch.float32)

input = torch.reshape(input, (1, 1, 1, 3))
target = torch.reshape(target, (1, 1, 1, 3))

loss1 = nn.L1Loss(reduction='sum')
result1 = loss1(input, target)
print(result1)

loss2 = nn.MSELoss()
result2 = loss2(input, target)
print(result2)

loss3 = nn.CrossEntropyLoss()
x = torch.tensor([0.1, 0.2, 0.3])
y = torch.tensor([1])
x = torch.reshape(x, (1, 3))
result3 = loss3(x, y)
print(result3)
