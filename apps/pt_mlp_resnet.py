import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import time
import numpy as np

def ResidualBlock(dim, hidden_dim, norm=nn.BatchNorm1d, drop_prob=0.1):
    class Residual(nn.Module):
        def __init__(self, fn):
            super().__init__()
            self.fn = fn
        def forward(self, x):
            return self.fn(x) + x

    fn = nn.Sequential(
        nn.Linear(dim, hidden_dim),
        norm(hidden_dim),
        nn.ReLU(),
        nn.Dropout(drop_prob),
        nn.Linear(hidden_dim, dim),
        norm(dim)
    )
    return nn.Sequential(
        Residual(fn),
        nn.ReLU()
    )

class MLPResNet(nn.Module):
    def __init__(self, dim, hidden_dim=100, num_blocks=3, num_classes=10, norm=nn.BatchNorm1d, drop_prob=0.1):
        super().__init__()
        layers = [
            nn.Linear(dim, hidden_dim),
            nn.ReLU()
        ]
        for _ in range(num_blocks):
            layers.append(ResidualBlock(hidden_dim, hidden_dim // 2, norm, drop_prob))
        layers.append(nn.Linear(hidden_dim, num_classes))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        return self.model(x)

def train_mnist(batch_size=100, epochs=10, lr=0.001, weight_decay=0.001, hidden_dim=150):
    device = torch.device("cpu")
    
    import gzip
    import struct

    def load_mnist_images(filename):
        with gzip.open(filename, 'rb') as f:
            magic, num, rows, cols = struct.unpack(">IIII", f.read(16))
            images = np.frombuffer(f.read(), dtype=np.uint8).reshape(num, 784)
            return images.astype(np.float32) / 255.0

    def load_mnist_labels(filename):
        with gzip.open(filename, 'rb') as f:
            magic, num = struct.unpack(">II", f.read(8))
            labels = np.frombuffer(f.read(), dtype=np.uint8)
            return labels.astype(np.int64)

    X_train = load_mnist_images("data/train-images-idx3-ubyte.gz")
    y_train = load_mnist_labels("data/train-labels-idx1-ubyte.gz")
    X_test = load_mnist_images("data/t10k-images-idx3-ubyte.gz")
    y_test = load_mnist_labels("data/t10k-labels-idx1-ubyte.gz")

    train_set = torch.utils.data.TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True)
    
    test_set = torch.utils.data.TensorDataset(torch.from_numpy(X_test), torch.from_numpy(y_test))
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False)

    model = MLPResNet(784, hidden_dim=hidden_dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss()

    print(f"Starting PyTorch MLPResNet training on MNIST...")
    for epoch in range(epochs):
        model.train()
        start_time = time.time()
        total_loss = 0
        correct = 0
        total = 0
        for X, y in train_loader:
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()
            outputs = model(X)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item() * X.size(0)
            _, predicted = outputs.max(1)
            total += y.size(0)
            correct += predicted.eq(y).sum().item()
        
        end_time = time.time()
        print(f"Epoch {epoch+1} train_corr = {correct/total:.4f}, train_loss = {total_loss/total:.4f}, time = {end_time-start_time:.2f}s")

    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for X, y in test_loader:
            X, y = X.to(device), y.to(device)
            outputs = model(X)
            _, predicted = outputs.max(1)
            total += y.size(0)
            correct += predicted.eq(y).sum().item()
    
    print(f"Final Test: test_corr = {correct/total:.4f}")

if __name__ == "__main__":
    train_mnist()
