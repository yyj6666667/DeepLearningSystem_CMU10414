import sys
sys.path.append('./python')
sys.path.append('./apps')

import needle as ndl
from models import ResNet9
from simple_ml import train_cifar10, evaluate_cifar10

device = ndl.cpu()
train_dataset = ndl.data.CIFAR10Dataset("data/cifar-10-batches-py", train = True)
train_dataloader = ndl.data.DataLoader(dataset = train_dataset,
                                       batch_size = 128,
                                       shuffle = True,
                                       )
test_dataset = ndl.data.CIFAR10Dataset("data/cifar-10-batches-py", train = False)
test_dataloader = ndl.data.DataLoader(dataset = test_dataset,
                                      batch_size = 128,
                                      shuffle = False,
                                      )

model = ResNet9(device = device, dtype = "float32")
best_acc = -1
train_cifar10(model, train_dataloader, n_epochs = 10, optimizer = ndl.optim.Adam, 
                  lr = 0.0001, weight_decay=0.0001)
evaluate_cifar10(model, test_dataloader)