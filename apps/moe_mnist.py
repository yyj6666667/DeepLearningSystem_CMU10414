import sys
sys.path.append('./python')
import needle as ndl
import needle.nn as nn
import needle.ops as ops
import numpy as np
import time

def train_mnist_moe(batch_size=100, epochs=3, lr=0.001, weight_decay=0.001, num_experts=4):
    device = ndl.cpu()
    
    # Load MNIST data
    train_dataset = ndl.data.MNISTDataset(
        "data/train-images-idx3-ubyte.gz",
        "data/train-labels-idx1-ubyte.gz"
    )
    train_loader = ndl.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    # Define MoE Model
    class MoEMNIST(nn.Module):
        def __init__(self, num_experts=4, device=None):
            super().__init__()
            self.model = nn.Sequential(
                nn.Flatten(),
                nn.Linear(784, 128, device=device),
                nn.ReLU(),
                nn.MoE(128, 128, num_experts=num_experts, device=device),
                nn.ReLU(),
                nn.Linear(128, 10, device=device)
            )
            self.importance_loss = nn.ImportanceLoss(w=0.01)

        def forward(self, x):
            return self.model(x)

    model = MoEMNIST(num_experts=num_experts, device=device)
    opt = ndl.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.SoftmaxLoss()

    print(f"Starting MoE training on MNIST with {num_experts} experts...")
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        total_correct = 0
        total_samples = 0
        start_time = time.time()
        
        # Track expert usage
        expert_usage = np.zeros(num_experts)
        
        for batch in train_loader:
            X, y = batch
            X, y = ndl.Tensor(X, device=device), ndl.Tensor(y, device=device)
            
            opt.reset_grad()
            logits = model(X)
            
            # Main loss + Importance loss
            main_loss = loss_fn(logits, y)
            # Find the MoE layer to get gate weights
            moe_layer = model.model.modules[3]
            aux_loss = model.importance_loss(moe_layer.gate_weights)
            
            total_loss_val = main_loss + aux_loss
            total_loss_val.backward()
            opt.step()
            
            # Stats
            total_loss += total_loss_val.data.numpy() * y.shape[0]
            total_correct += np.sum(np.argmax(logits.numpy(), axis=1) == y.numpy())
            total_samples += y.shape[0]
            
            # Track usage (which expert had max weight)
            weights = moe_layer.gate_weights.numpy()
            chosen_experts = np.argmax(weights, axis=1)
            for e in chosen_experts:
                expert_usage[e] += 1

        end_time = time.time()
        avg_acc = total_correct / total_samples
        avg_loss = total_loss / total_samples
        usage_dist = expert_usage / total_samples
        
        print(f"Epoch {epoch}: Acc: {avg_acc:.4f}, Loss: {avg_loss:.4f}, Time: {end_time-start_time:.2f}s")
        print(f"Expert Usage Distribution: {usage_dist}")

if __name__ == "__main__":
    train_mnist_moe()
