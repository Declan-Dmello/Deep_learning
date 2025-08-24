import numpy as np
import torch
import torchvision
from  torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10
from model import SimpleCNN
import torch.optim as optim
import torch.nn as nn
import matplotlib.pyplot as plt



#DATA LOADING AND PREPROCESSING
#for transformation basically
transformation_cifar = (transforms.Compose
    ([transforms.ToTensor(),
     transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
     ]))

train_data = CIFAR10(download=True ,transform=transformation_cifar, root="./data", train=True)
test_data = CIFAR10(download=True , train=False , transform=transformation_cifar, root="./data")


train_loader = DataLoader(train_data , shuffle=True, batch_size=32)
test_loader = DataLoader(test_data, shuffle=False , batch_size=32)


#THE MODEL BUILDING
model = SimpleCNN()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
print(model)


# THE LOSS FUNCTION AND OOT

#Defining the loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr = 0.001)



#The training loop

loss_per_epoch =[]

no_epoches = 15
for epoch in range(no_epoches):
    model.train()
    running_loss = 0.0

    for images , labels in train_loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)

        loss = criterion(outputs, labels) # basically loss using true vs predicted

        loss.backward()

        optimizer.step()

        running_loss += loss.item()

    loss_per_epoch.append(running_loss)
    print(f"Epoch No : {epoch+1} /{no_epoches} , Loss:{running_loss} ")





model.eval()

correct = 0
total = 0

with torch.no_grad():

    for images, labels in test_loader:
        images,labels= images.to(device),labels.to(device)

        outputs = model(images)

        _, predicted = torch.max(outputs.data , 1) #1 is the dimemsion

        total +=labels.size(0)

        correct += (predicted==labels).sum().item() #item() basically gives a number from a tensor ([11]) -> 11 but only single elements


accuracy = correct/total * 100
print("Accuracy : {:.2f}%".format(accuracy))


#THE VISUALIZATION

plt.plot(loss_per_epoch)
plt.title('Loss per epoch')
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.grid(True)
plt.show()




classes = train_data.classes  # ['airplane', 'automobile', ..., 'truck']

def imshow(img):
    img = img / 2 + 0.5  # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

# Get a batch of test images
dataiter = iter(test_loader)
images, labels = next(dataiter)

# Predict
images = images.to(device)
outputs = model(images)
_, predicted = torch.max(outputs, 1)

# Show images
imshow(torchvision.utils.make_grid(images.cpu()[:8]))
print("Predicted:", ' | '.join(classes[predicted[i]] for i in range(8)))
print("Actual:   ", ' | '.join(classes[labels[i]] for i in range(8)))


