import sys
import os

cur_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(cur_dir, "../python"))
import needle as ndl
import needle.nn as nn
import numpy as np
import time

np.random.seed(0)
# MY_DEVICE = ndl.backend_selection.cuda()


def ResidualBlock(dim, hidden_dim, norm=nn.BatchNorm1d, drop_prob=0.1):
    ### BEGIN YOUR SOLUTION
    sequence_1 =  nn.Sequential(
                      nn.Linear(dim, hidden_dim),
                      norm(hidden_dim),
                      nn.ReLU(),
                      nn.Dropout(drop_prob),
                      nn.Linear(hidden_dim, dim),
                      norm(dim)
                  )
    sequence_2 = nn.Sequential(
                      nn.Residual(sequence_1),
                      nn.ReLU()
    )
    return sequence_2
    ### END YOUR SOLUTION


def MLPResNet(
    dim,
    hidden_dim=100,
    num_blocks=3,
    num_classes=10,
    norm=nn.BatchNorm1d,
    drop_prob=0.1,
):
    ### BEGIN YOUR SOLUTION
 #   tuple_ResidualBlock = []
 #   for i in range(num_blocks):
 #       tuple_ResidualBlock.append(ResidualBlock(hidden_dim, #hidden_dim//2, norm, drop_prob))
 #   
 #   sequence = nn.Sequential(
 #                       nn.Linear(dim, hidden_dim),
 #                       nn.ReLU(),
 #                       *tuple_ResidualBlock,
 #                       nn.Linear(hidden_dim, num_classes)
 #   )
 #   return sequence

    sequence = nn.Sequential(
                nn.Linear(dim, hidden_dim),
                nn.ReLU(),
                *[ResidualBlock(hidden_dim, hidden_dim//2, norm, drop_prob) for _ in range(num_blocks)],
                nn.Linear(hidden_dim, num_classes)
    )
    return sequence
    ### END YOUR SOLUTION


def epoch(dataloader: ndl.data.DataLoader, model :nn.Module, opt: ndl.optim.Optimizer =None):
    np.random.seed(4)
    ### BEGIN YOUR SOLUTION
    model.eval() if opt is None else model.train()
    loss_fn = nn.SoftmaxLoss()
    sum_loss = 0.0
    sum_err = 0.0


    for i, batch in enumerate(dataloader):
        X, y  = batch
        hypothesis = model(X.reshape((X.shape[0], -1)))
        loss = loss_fn(hypothesis, y)

        if model.training:
            opt.reset_grad()
            loss.backward()
            opt.step()

        sum_loss += loss.data.numpy() * X.shape[0]
        y_pred = np.argmax(hypothesis.detach().numpy(), axis = 1)
        sum_err += np.sum(y_pred != y.numpy())

    aver_loss = sum_loss / len(dataloader.dataset) * 1.0
    aver_err  = sum_err / len(dataloader.dataset) * 1.0
    return aver_err, aver_loss
        
    ### END YOUR SOLUTION


def train_mnist(
    batch_size=100,
    epochs=10,
    optimizer=ndl.optim.Adam,
    lr=0.001,
    weight_decay=0.001,
    hidden_dim=100,
    data_dir="data",
):
    np.random.seed(4)
    ### BEGIN YOUR SOLUTION
    optional_transforms_1 = [
    ]
    data_train = ndl.data.MNISTDataset(data_dir + "/train-images-idx3-ubyte.gz",
                                           data_dir + "/train-labels-idx1-ubyte.gz",
                                           optional_transforms_1)
    print("train with transforms:", optional_transforms_1)
    data_test = ndl.data.MNISTDataset(data_dir + "/t10k-images-idx3-ubyte.gz",
                                      data_dir + "/t10k-labels-idx1-ubyte.gz"
                                      )
    train_loader = ndl.data.DataLoader(data_train, batch_size, shuffle=True)
    test_loader  = ndl.data.DataLoader(data_test,  batch_size, shuffle = False)

    model = MLPResNet(28 ** 2, hidden_dim = hidden_dim )
    opt = optimizer(model.parameters(), lr = lr, weight_decay=weight_decay)

    for e in range(epochs):
        train_err, train_loss = epoch(train_loader, model, opt=opt)
        print(f"Epoch {e + 1}  train_err = {train_err:.4f}, train_loss= {train_loss:.4f}")
    
    test_err, test_loss = epoch(test_loader, model, None)
    print(f"Final Test: test_err = {test_err:.4f}, test_loss= {test_loss:.4f}")

    return train_err, train_loss, test_err, test_loss
    ### END YOUR SOLUTION


if __name__ == "__main__":
    train_mnist(data_dir="../data")
