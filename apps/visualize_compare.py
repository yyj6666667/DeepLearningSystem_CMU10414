import matplotlib.pyplot as plt
import re
import numpy as np

def parse_log(filename):
    epochs = []
    accs = []
    losses = []
    times = []
    
    with open(filename, 'r', encoding='utf-16') as f:
        content = f.read()
        
    # Pattern for needle_mlp_resnet and pt_mlp_resnet
    # Epoch 1  train_corr = 0.8968, train_loss= 0.3373, time = 9.70s
    matches = re.findall(r'Epoch\s+(\d+)\s+train_corr\s*=\s*([\d.]+),\s*train_loss\s*=\s*([\d.]+),\s*time\s*=\s*([\d.]+)s', content)
    if not matches:
        # Pattern for needle_moe_mnist and pt_moe_mnist
        # Epoch 0: Acc: 0.9135, Loss: 0.3294, Time: 6.10s
        matches = re.findall(r'Epoch\s+(\d+):\s+Acc:\s*([\d.]+),\s+Loss:\s*([\d.]+),\s+Time:\s*([\d.]+)s', content)
        
    for m in matches:
        epochs.append(int(m[0]))
        accs.append(float(m[1]))
        losses.append(float(m[2]))
        times.append(float(m[3]))
        
    return epochs, accs, losses, times

def plot_comparison(needle_log, pt_log, title, save_name):
    n_e, n_a, n_l, n_t = parse_log(needle_log)
    p_e, p_a, p_l, p_t = parse_log(pt_log)
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle(title, fontsize=16)
    
    # Accuracy
    axes[0].plot(n_e, n_a, 'o-', label='Needle')
    axes[0].plot(p_e, p_a, 's-', label='PyTorch')
    axes[0].set_title('Training Accuracy')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Accuracy')
    axes[0].legend()
    axes[0].grid(True)
    
    # Loss
    axes[1].plot(n_e, n_l, 'o-', label='Needle')
    axes[1].plot(p_e, p_l, 's-', label='PyTorch')
    axes[1].set_title('Training Loss')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Loss')
    axes[1].legend()
    axes[1].grid(True)
    
    # Time
    axes[2].bar(['Needle', 'PyTorch'], [np.mean(n_t), np.mean(p_t)], color=['blue', 'orange'])
    axes[2].set_title('Average Time per Epoch (s)')
    axes[2].set_ylabel('Seconds')
    axes[2].grid(axis='y')
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(save_name)
    print(f"Saved {save_name}")

if __name__ == "__main__":
    plot_comparison('needle_mlp_resnet.log', 'pt_mlp_resnet.log', 'MLP ResNet Performance Comparison', 'mlp_resnet_comparison.png')
    plot_comparison('needle_moe_mnist.log', 'pt_moe_mnist.log', 'MoE Performance Comparison', 'moe_mnist_comparison.png')
