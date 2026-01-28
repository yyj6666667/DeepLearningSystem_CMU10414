import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import time
import numpy as np

class MoE(nn.Module):
    def __init__(self, input_size, output_size, num_experts, top_k=1):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.experts = nn.ModuleList([nn.Linear(input_size, output_size) for _ in range(num_experts)])
        self.gate = nn.Linear(input_size, num_experts, bias=False)

    def forward(self, x):
        gate_logits = self.gate(x)
        gate_weights = torch.softmax(gate_logits, dim=1)
        
        # Top-1 routing
        max_weights, _ = torch.max(gate_weights, dim=1, keepdim=True)
        mask = (gate_weights >= max_weights).float()
        selected_weights = gate_weights * mask
        
        final_output = 0
        for i in range(self.num_experts):
            expert_output = self.experts[i](x)
            w = selected_weights[:, i:i+1]
            final_output += expert_output * w
        return final_output, gate_weights

class ImportanceLoss(nn.Module):
    def __init__(self, w=0.01):
        super().__init__()
        self.w = w

    def forward(self, gate_weights):
        batch_size = gate_weights.size(0)
        importance = torch.sum(gate_weights, dim=0) / batch_size
        loss = torch.sum(importance * importance) * gate_weights.size(1)
        return loss * self.w

class MoEMNIST(nn.Module):
    def __init__(self, num_experts=4):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear1 = nn.Linear(784, 128)
        self.relu1 = nn.ReLU()
        self.moe = MoE(128, 128, num_experts=num_experts)
        self.relu2 = nn.ReLU()
        self.linear2 = nn.Linear(128, 10)
        self.importance_loss_fn = ImportanceLoss(w=0.01)

    def forward(self, x):
        x = self.flatten(x)
        x = self.linear1(x)
        x = self.relu1(x)
        x, gate_weights = self.moe(x)
        x = self.relu2(x)
        x = self.linear2(x)
        return x, gate_weights

def train_mnist_moe(batch_size=100, epochs=10, lr=0.001, weight_decay=0.001, num_experts=4):
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

    model = MoEMNIST(num_experts=num_experts).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss()

    print(f"Starting PyTorch MoE training on MNIST with {num_experts} experts...")
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        total_correct = 0
        total_samples = 0
        start_time = time.time()
        
        for X, y in train_loader:
            X, y = X.to(device), y.to(device)
            
            optimizer.zero_grad()
            logits, gate_weights = model(X)
            
            main_loss = criterion(logits, y)
            aux_loss = model.importance_loss_fn(gate_weights)
            
            loss = main_loss + aux_loss
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item() * y.size(0)
            _, predicted = logits.max(1)
            total_correct += predicted.eq(y).sum().item()
            total_samples += y.size(0)

        end_time = time.time()
        print(f"Epoch {epoch}: Acc: {total_correct/total_samples:.4f}, Loss: {total_loss/total_samples:.4f}, Time: {end_time-start_time:.2f}s")
    
    model.eval()
    test_correct = 0
    test_samples = 0
    with torch.no_grad():
        for X, y in test_loader:
            X, y = X.to(device), y.to(device)
            logits, _ = model(X)
            _, predicted = logits.max(1)
            test_correct += predicted.eq(y).sum().item()
            test_samples += y.size(0)
    
    print(f"Final Test Accuracy: {test_correct / test_samples:.4f}")

if __name__ == "__main__":
    train_mnist_moe()
