import torch.nn as nn
import torch.nn.functional as F


class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN,self).__init__()
        #This is just defining the layers here , the model will learn
        #data from the forward method
        #convo layers
        #channels change based on developer
        #pixels gets halved for every conv layer

        #out channels == no of features u want or richness
        #more the better ,but performance issue
        self.conv1 = nn.Conv2d(in_channels=3,out_channels=16,kernel_size=3, padding=1)
        self.conv2= nn.Conv2d(in_channels=16, out_channels=32, padding=1, kernel_size=3)

        #fully connected layers
        self.fc1 = nn.Linear(32*8*8,64)
        self.fc2 = nn.Linear(64,10)
        #max pooling layer(basically keeps only imp info like PCA)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)


    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))  #Converts chanels from 3 to 16 as mentioned in the architecture
        #Also reduces spatial size from 32*32 to 16*16 (pooling) , relu introduces non linearity
        x = self.pool(F.relu(self.conv2(x)))  # same thing but reduced spatial size to 8*8 , and chanels to 32
        x = x.view(-1  , 32*8*8)  #basically flattens out the pixels to a line of nums (-1 tells it to auto find the dimensions)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x



