import torch
from torch import nn


class mynn(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input):
        output = input + 1
        return output


test = mynn()
x = torch.tensor(1.0)
output = test(x)
print(output)
