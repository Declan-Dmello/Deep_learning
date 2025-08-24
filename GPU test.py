import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# Check if CUDA is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")


# Simple Neural Network (for illustration purposes)
class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(784, 128)  # 784 input features (e.g., 28x28 image)
        self.fc2 = nn.Linear(128, 10)  # 10 output classes (e.g., for MNIST)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x


# Initialize model and move to GPU
model = SimpleNN().to(device)

# Example dataset (dummy data for illustration)
X = torch.randn(1000, 784)  # 1000 samples, 784 features
y = torch.randint(0, 10, (1000,))  # 1000 labels for 10 classes

# Create DataLoader
dataset = TensorDataset(X, y)
train_loader = DataLoader(dataset, batch_size=32, shuffle=True)

# Loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
num_epochs = 5
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for data, target in train_loader:
        data, target = data.to(device), target.to(device)  # Move data to GPU
        optimizer.zero_grad()
        output = model(data)  # Forward pass on GPU
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {running_loss / len(train_loader)}")

print("Training complete!")
