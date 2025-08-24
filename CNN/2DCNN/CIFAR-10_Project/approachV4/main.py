import torchvision.models as models
import torch.nn as nn
import torch
from torchvision import transforms
import torchvision
from  torch.utils.data import DataLoader

resnet = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)

for param in resnet.parameters():
    param.requires_grad=False

resnet.fc = nn.Sequential(
    nn.Linear(resnet.fc.in_features, 256),
    nn.ReLU(),
    nn.Dropout(0.2),
    nn.Linear(256, 10)
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
resnet = resnet.to(device)

resnet_transforms =  transforms.Compose([

    transforms.RandomHorizontalFlip(),
    transforms.Resize(224),
transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

train_data = torchvision.datasets.CIFAR10( root = r"C:/Users/decla/PycharmProjects/pythonProject2/Deep_learning/CNN/2DCNN/CIFAR-10_Project/data",
                                           transform=resnet_transforms, train=True)
test_data = torchvision.datasets.CIFAR10( root = r"C:/Users/decla/PycharmProjects/pythonProject2/Deep_learning/CNN/2DCNN/CIFAR-10_Project/data",
                                           transform=resnet_transforms, train=False)


train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
test_loader = DataLoader(test_data, batch_size = 32, shuffle=False)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(resnet.fc.parameters(), lr=0.001)


#The training loop

loss_per_epoch = []
train_accu = []
test_accu = []
best_acc=  0

patience = 5
patience_counter = 0
epochs = 15
for epoch in range(epochs):
    resnet.train()

    running_loss  = 0.0
    correct, total = 0, 0

    for images, labels in train_loader:


        images = images.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        outputs = resnet(images)
        loss =  criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted==labels).sum().item()



    train_acc = correct / total * 100
    train_accu.append(train_acc)
    loss_per_epoch.append(running_loss)

    resnet.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for images ,labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = resnet(images)

            _, predicted = torch.max(outputs, 1)

            total += labels.size(0)
            correct += (predicted==labels).sum().item()


        test_acc = correct / total * 100
        test_accu.append(test_acc)

        if test_acc > best_acc:
            patience_counter = 0
            best_acc = test_acc
            torch.save(resnet.state_dict(), "best_model.pth")
            print(f"Model Checkpoint Saved at {epoch+1}!!")
        elif best_acc > test_acc:
            patience_counter += 1
            print(f"Model Accuracy didnt increase for {patience_counter} epochs")
            if patience_counter == patience:
                print("Early Stopping Triggered")
                break
        print(f"Epoch:  {epoch}/{epochs}  | Train Accuracy: {train_acc:.2f}  | Test Accuracy: {test_acc:.2f}")







