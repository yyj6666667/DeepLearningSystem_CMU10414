import numpy as np
import python.needle as ndl

np.random.seed(0)
device = ndl.cpu()
dataset = ndl.data.CIFAR10Dataset("./data/cifar-10-batches-py", train=True)
dataloader = ndl.data.DataLoader(
    dataset=dataset,
    batch_size=128,
    shuffle=False
)

from apps.models import ResNet9
np.random.seed(0)

def one_iter_of_cifar10_training(dataloader, model, niter=1, loss_fn=None, opt=None):
    if loss_fn is None:
        loss_fn = ndl.nn.SoftmaxLoss()
    if opt is None:
        opt = ndl.optim.Adam(model.parameters())
    
    np.random. seed(4)
    model.train()
    correct, total_loss = 0, 0
    i = 1
    for batch in dataloader:
        opt.reset_grad()
        X, y = batch
        #X, y = ndl.Tensor(X, device=device), ndl.Tensor(y, device=device)
        out = model(X)
        correct += np.sum(np.argmax(out.numpy(), axis=1) == y.numpy())
        loss = loss_fn(out, y)
        total_loss += loss.data.numpy() * y.shape[0]
        loss.backward()
        opt.step()
        if i >= niter: 
            break
        i += 1
    return correct/(y.shape[0]*niter), total_loss/(y.shape[0]*niter)

model = ResNet9(device=device, dtype="float32")
out = one_iter_of_cifar10_training(
    dataloader, 
    model, 
    opt=ndl.optim.Adam(model.parameters(), lr=0.001, weight_decay=0.001)
)
print(f"one epoch: correct is {out[0]}, total_loss is {out[1]}")