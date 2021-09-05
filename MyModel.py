import torch
import torch.nn as nn
from torch.nn.modules.activation import ReLU
from torch.utils.data import DataLoader
from torch.nn import Conv2d, MaxPool2d, Softmax, BatchNorm2d, Linear, Dropout


class AlexNet(nn.Module):
    def __init__(self):
        super(AlexNet, self).__init__()
        self.model = nn.Sequential(
            Conv2d(3, 32, 5, 1, 2),
            ReLU(),
            MaxPool2d(2),
            ReLU(),
            BatchNorm2d(32),
            Dropout(0.2),
            Conv2d(32, 32, 5, 1, 2),
            ReLU(),
            MaxPool2d(2),
            ReLU(),
            BatchNorm2d(32),
            Conv2d(32, 64, 5, 1, 2),
            ReLU(),
            MaxPool2d(2),
            ReLU(),
            nn.Flatten(),
            Linear(64*4*4, 64),
            ReLU(),
            Linear(64, 10)
        )

    def forward(self, x):
        x = self.model(x)
        return x


# 检查模型的输入和输出维度是否匹配
if __name__ == "__main__":
    # bacch_size, channels, h, w
    input = torch.ones((64, 3, 32, 32))
    alexnet = AlexNet()
    output = alexnet(input)
    print("Output: ", output.size())

  