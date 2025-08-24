import torch.nn as nn
import torch.nn.functional as F


class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN,self).__init__()

        self.conv1 = nn.Conv2d(in_channels=3,out_channels=16,kernel_size=3, padding=1)
        self.conv2= nn.Conv2d(in_channels=16, out_channels=32, padding=1, kernel_size=3)
        self.conv3= nn.Conv2d(in_channels=32, out_channels=64, padding=1, kernel_size=3)

        self.fc1 = nn.Linear(64*4*4,128)  #basically the outchannels of the conv2d and the width and height
        self.fc2 = nn.Linear(128,10)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout = nn.Dropout(0.2)

        self.bn1 = nn.BatchNorm2d(16)
        self.bn2= nn.BatchNorm2d(32)
        self.bn3= nn.BatchNorm2d(64)


    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        x = x.view(-1  , 64*4*4)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x


#Feature Maps
    def extract_feature_maps(self,x):
        feature_maps = []

        # Conv Layer 1
        fmap1 = self.pool(F.relu(self.bn1(self.conv1(x))))
        feature_maps.append(fmap1)

        # Conv Layer 2
        fmap2 = self.pool(F.relu(self.bn2(self.conv2(fmap1))))
        feature_maps.append(fmap2)

        # Conv Layer 3
        fmap3 = self.pool(F.relu(self.bn3(self.conv3(fmap2))))
        feature_maps.append(fmap3)

        return feature_maps



