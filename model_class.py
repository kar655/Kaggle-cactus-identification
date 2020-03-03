import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):

    def __init__(self):
        super().__init__()
        # img size 32 * 32
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels= 128, kernel_size=3)
        # 
        # 32 - 3 + 1 = 30
        # 30 / 2 = 15
        # 15 - 3 + 1 = 13
        # 13 / 2 = 6
        # 6 - 3 + 1 = 4
        # 4 / 2 = 2
        self.fc1 = nn.Linear(in_features=128 * 2 * 2, out_features=512)
        self.fc2 = nn.Linear(in_features=512, out_features=2)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        x = F.max_pool2d(F.relu(self.conv2(x)), (2, 2))
        x = F.max_pool2d(F.relu(self.conv3(x)), (2, 2))

        x = x.view(-1, 128 * 2 * 2)

        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        x = F.softmax(x, dim=1)

        return x