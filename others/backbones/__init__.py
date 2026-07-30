from torch import nn


class Wrapper(nn.Module):
    def __init__(self, model1, model2):
        super(Wrapper, self).__init__()
        self.model1 = model1
        self.model2 = model2

    def forward(self, x, t):
        out = self.model1(x, t)
        out = self.model2(out)
        return out