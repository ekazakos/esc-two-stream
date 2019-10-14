import torch
from torch import nn


class ESCModel(nn.Module):
    def __init__(self):
        super(ESCModel, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, stride=1, padding=1)
        self.batch_norm1 = nn.BatchNorm2d(32)
        self.relu1 = nn.ReLU()

        self.conv2 = nn.Conv2d(32, 32, 3, stride=1, padding=1)
        self.batch_norm2 = nn.BatchNorm2d(32)
        self.relu2 = nn.ReLU()
        self.max_pool2 = nn.MaxPool2d(2)
        self.dropout2 = nn.Dropout(p=0.5)

        self.conv3 = nn.Conv2d(32, 64, 3, stride=1, padding=1)
        self.batch_norm3 = nn.BatchNorm2d(64)
        self.relu3 = nn.ReLU()

        self.conv4 = nn.Conv2d(64, 64, 3, stride=1, padding=1)
        self.batch_norm4 = nn.BatchNorm2d(64)
        self.relu4 = nn.ReLU()
        self.max_pool4 = nn.MaxPool2d(2)
        self.dropout4 = nn.Dropout(p=0.5)


        self.fc1 = nn.Linear(64*10*21, 1024)
        self.sigm = nn.Sigmoid()
        self.dropout_fc1 = nn.Dropout(p=0.5)
        self.fc2 = nn.Linear(1024, 10)

    def forward(self, input):
        x = self.conv1(input)
        x = self.batch_norm1(x)
        x = self.relu1(x)

        x = self.conv2(x)
        x = self.batch_norm2(x)
        x = self.relu2(x)
        x = self.max_pool2(x)
        x = self.dropout2(x)

        x = self.conv3(x)
        x = self.batch_norm3(x)
        x = self.relu3(x)

        x = self.conv4(x)
        x = self.batch_norm4(x)
        x = self.relu4(x)
        x = self.max_pool4(x)
        x = self.dropout4(x)

        x = x.flatten(1)
        x = self.fc1(x)
        x = self.sigm(x)
        x = self.dropout_fc1(x)

        output = self.fc2(x)

        return output


# class FusionModel(nn.Module):
#
#     def __init__(self):
#         super(FusionModel, self).__init__()
#
#
#
#     def forward(self, input):
#
#         return output
