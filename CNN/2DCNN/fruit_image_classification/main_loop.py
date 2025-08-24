import torch
import torch.nn as nn
from torchvision import transforms
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torch.optim import Adam , lr_scheduler
from model_architecture import Img_classifier
import os
import multiprocessing
from tqdm import tqdm

train_path = r"C:\Users\decla\PycharmProjects\pythonProject2\data\Fruit_dataset\train1"
test_path =  r"C:\Users\decla\PycharmProjects\pythonProject2\data\Fruit_dataset\test1"
val_path = r"C:\Users\decla\PycharmProjects\pythonProject2\data\Fruit_dataset\val1"

device = "cuda" if torch.cuda.is_available() else "cpu"
torch.backends.cudnn.benchmark = True  # Add this line here

#sorted_classes = sorted(os.listdir(train_path))
#classes_to_idx = {cls_name : idx for idx, cls_name in enumerate(sorted_classes)}

#Loading the data and stuff


train_transformation = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor()
    ,transforms.Normalize((0.485, 0.456, 0.406),
                     (0.229, 0.224, 0.225))])


test_transformation = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406),
                         (0.229, 0.224, 0.225))])


train_data = ImageFolder(root=train_path, transform=train_transformation)
test_data = ImageFolder(root=test_path, transform=test_transformation)
val_data = ImageFolder(root=val_path, transform=test_transformation)
batch_size = 32

if __name__ == "__main__":
    train_dataloader = DataLoader(train_data , shuffle=True, batch_size=batch_size, num_workers=6, pin_memory=True)
    test_dataloader = DataLoader(test_data, shuffle=False, batch_size=batch_size, num_workers=6, pin_memory=True)
    val_dataloader = DataLoader(val_data, shuffle=False, batch_size=batch_size, num_workers=6, pin_memory=True)



    model =  Img_classifier().to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=0.001)
    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5, verbose=True)

    #training loop
    epochs = 50
    train_acc = []
    val_acc = []
    test_acc  = []
    loss_per_epoch = []
    patience_counter = 8
    patience = 0
    best_acc = 0.0
    best_loss = float('inf')


    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train =0


        for images, labels in tqdm(train_dataloader, desc="Training", leave=False):
            images = images.to(device)
            labels =  labels.to(device)

            optimizer.zero_grad()

            outputs = model(images)
            _ ,prediction = torch.max(outputs, 1)

            total_train += labels.size(0)
            correct_train += (prediction == labels).sum().item()
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        scheduler.step()

        print(f"The learning rate is {scheduler.get_last_lr()}")

        training_accuracy = 100 * (correct_train / total_train)
        train_acc.append(training_accuracy)

        loss_per_epoch.append(running_loss)

        correct = 0
        total = 0
        model.eval()
        total_test_loss = 0

        with torch.no_grad():

            for images, labels in tqdm(val_dataloader, desc="Validating", leave=False):
                images =images.to(device)
                labels= labels.to(device)

                outputs = model(images)

                _,prediction = torch.max(outputs, 1)
                correct += (prediction ==  labels).sum().item()
                total += labels.size(0)
                loss = criterion(outputs, labels)
                total_test_loss += loss.item()

        validation_accuracy = (correct / total) * 100

        #early stopping

        if epoch > 9:
            if validation_accuracy > best_acc:
                best_acc = validation_accuracy
                torch.save(model.state_dict(), "best_model.pth")
                print("The best model is saved {}".format(validation_accuracy))
                patience = 0
            else:
                patience += 1
                print(f"The model accuracy didnt increase for {patience} epochs")
                if patience > patience_counter:
                    print("Early Stopping Triggered!")
                    break



        val_acc.append(validation_accuracy)
        accuracy =  correct / total * 100

        print(
            f"Epoch {epoch + 1}/{epochs} | Loss: {running_loss:.2f} | Train Acc: {training_accuracy:.2f}% | Val Acc: {validation_accuracy:.2f}%")

    model.load_state_dict(torch.load("best_model.pth"))
    model.eval()

    correct_test = 0
    total_test = 0

    with torch.no_grad():
        for images, labels in tqdm(test_dataloader, desc="Testing", leave=False):
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, prediction = torch.max(outputs, 1)

            total_test +=labels.size(0)
            correct_test += prediction.eq(labels).sum().item()
    testing_accuracy = (correct_test / total_test) * 100
    test_acc.append(testing_accuracy)
    print(testing_accuracy)