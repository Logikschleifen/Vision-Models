import torch
from torch import nn, optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# 1. Prepare data
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))  # mean, std
])

train_dataset = datasets.FashionMNIST(root='./data',
                                      train=True,
                                      download=True,
                                      transform=transform)
test_dataset = datasets.FashionMNIST(root='./data',
                                     train=False,
                                     download=True,
                                     transform=transform)

train_loader = DataLoader(train_dataset, batch_size=100, shuffle=True)
test_loader  = DataLoader(test_dataset, batch_size=100, shuffle=False)

# 2. Define model
class FashionClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.layers = nn.Sequential(
            nn.Linear(28*28, 128),
            nn.ReLU(),
            nn.TransformerEncoderLayer(d_model=128, nhead=8),
            nn.TransformerDecoderLayer(d_model=80, nhead=10),
        )
    def forward(self, x):
        x = self.flatten(x)
        logits = self.layers(x)
        return logits

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = FashionClassifier().to(device)

# 3. Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# 4. Training loop
epochs = 15
for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * images.size(0)
        _, predicted = torch.max(outputs, 1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)
        
    epoch_loss = running_loss / total
    epoch_acc = correct / total
    print(f"Epoch {epoch+1}/{epochs} — loss: {epoch_loss:.4f}, acc: {epoch_acc:.4f}")

# 5. Evaluation
model.eval()
test_loss = 0.0
correct = 0
total = 0
with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        test_loss += loss.item() * images.size(0)
        _, predicted = torch.max(outputs, 1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)

test_loss = test_loss / total
test_acc = correct / total
print(f"Test — loss: {test_loss:.4f}, acc: {test_acc:.4f}")

torch.save(model.state_dict(), "model_weights.pth")
