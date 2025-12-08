import struct
import numpy as np
import gzip
try:
    from simple_ml_ext import *
except:
    pass


def add(x, y):
    """ A trivial 'add' function you should implement to get used to the
    autograder and submission system.  The solution to this problem is in the
    the homework notebook.

    Args:
        x (Python number or numpy array)
        y (Python number or numpy array)

    Return:
        Sum of x + y
    """
    ### BEGIN YOUR CODE
    return x + y
    ### END YOUR CODE


def parse_mnist(image_filename, label_filename):
    """ Read an images and labels file in MNIST format.  See this page:
    http://yann.lecun.com/exdb/mnist/ for a description of the file format.

    Args:
        image_filename (str): name of gzipped images file in MNIST format
        label_filename (str): name of gzipped labels file in MNIST format

    Returns:
        Tuple (X,y):
            X (numpy.ndarray[np.float32]): 2D numpy array containing the loaded 
                data.  The dimensionality of the data should be 
                (num_examples x input_dim) where 'input_dim' is the full 
                dimension of the data, e.g., since MNIST images are 28x28, it 
                will be 784.  Values should be of type np.float32, and the data 
                should be normalized to have a minimum value of 0.0 and a 
                maximum value of 1.0 (i.e., scale original values of 0 to 0.0 
                and 255 to 1.0).

            y (numpy.ndarray[dtype=np.uint8]): 1D numpy array containing the
                labels of the examples.  Values should be of type np.uint8 and
                for MNIST will contain the values 0-9.
    """
    ### BEGIN YOUR CODE
    with gzip.open(label_filename, 'rb') as lbl_f: ##
        ##why do we need to read magic number?
        magic, num_labels = struct.unpack(">II", lbl_f.read(8))
        y = np.frombuffer(lbl_f.read(), np.uint8)
    with gzip.open(image_filename, 'rb') as img_f:
        magic, num_images, rows, cols = struct.unpack(">IIII", img_f.read(16))
        x = np.frombuffer(img_f.read(), np.uint8).reshape(num_images, rows*cols)
        x = x.astype(np.float32) / 255.0
        #if(x.dtype != np.float32):
        #    panic("aaah")
        return x, y


    ### END YOUR CODE


def loss_ce_1(Z, y):
    ##写法一：直接使用Z_hot,即loss_ce = -log(Z_hot)
    ##       注：Z 即 softmax(hypothesis(x))
    Z_hot = np.zeros(Z.shape[0])
    for i in range(Z.shape[0]):
        index = y[i]
        Z_hot[i] = np.exp(Z[i, index]) / np.sum(np.exp(Z[i]))
    Loss = np.mean(-np.log(Z_hot))
    return Loss

def loss_ce_2(Z, y):
    ##写法二：loss_ce = -h_y(x) + log(\sum exp(h_i)) 
    batch_size = Z.shape[0]
    hypo_y = Z[np.arange(batch_size), y]
    sum = 0
    for j in range(Z.shape[1]):
        sum += np.exp(Z[np.arange(batch_size), j])
    Loss_vector = -(hypo_y) + np.log(sum)
    Loss = np.mean(Loss_vector)
    return Loss

def loss_ce_3(Z, y):
    #探索更短的写法哈哈哈
    Loss_vector = -Z[np.arange(Z.shape[0]), y] + np.log(np.exp(Z).sum(axis=1))
    return Loss_vector.mean()
    ##一行解决问题，优雅！


def softmax_loss(Z, y):
    """ Return softmax loss.  Note that for the purposes of this assignment,
    you don't need to worry about "nicely" scaling the numerical properties
    of the log-sum-exp computation, but can just compute this directly.

    Args:
        Z (np.ndarray[np.float32]): 2D numpy array of shape
            (batch_size, num_classes), containing the logit predictions for
            each class.
        y (np.ndarray[np.uint8]): 1D numpy array of shape (batch_size, )
            containing the true label of each example.

    Returns:
        Average softmax loss over the sample.
    """
    ### BEGIN YOUR CODE
    return loss_ce_3(Z, y)
    ### END YOUR CODE

def loss(H, y): #写了没屁用，吐血
    Loss = -H[np.arange(H.shape[0]), y] + np.log(np.sum(np.exp(H), axis=1))
    return Loss #(batch_size,)

def solve_1(X, y, theta, lr, batch):
    for i in range(0, X.shape[0], batch):
        Xb = X[i:i+batch]
        Yb = y[i:i+batch]
        Hypo = Xb @ theta
        Z_tem = np.exp(Hypo)
        Z = Z_tem / np.sum(Z_tem ,axis = 1,  keepdims = True) #normalize
        P = Z
        P[np.arange(Z.shape[0]), Yb] -= 1 # - Iy
        grad = Xb.T @ P
        theta -= lr * grad / batch

def solve_2(X, y, theta, lr, batch):
    for i in range(0, X.shape[0], batch):
        Xb = X[i:i+batch]
        yb = y[i:i+batch] 
        hypothesis = Xb @ theta
        Z = np.exp(hypothesis) / np.sum(np.exp(hypothesis), axis = 1, keepdims = True)  #keepdims 吐血
        Iy_b = np.zeros((yb.shape[0], theta.shape[1]), dtype = np.float32)  #dtype 吐血
        Iy_b[np.arange(yb.shape[0]), yb] = 1   
        grad = Xb.T @ (Z - Iy_b)
        theta -= lr / (batch) * grad

def solve_3(X, y, theta, lr, batch):
    n = X.shape[0]
    for i in range(0, n, batch):
        Xb = X[i:i+batch]
        yb = y[i: i+batch]
        Z = Xb @ theta
        Z = Z - Z.max(axis = 1, keepdims = True)
        P = np.exp(Z) / np.exp(Z).sum(axis = 1, keepdims = True)
        P[np.arange(yb.shape[0]), yb] -= 1
        grad = Xb.T @ P 
        theta -= lr /yb.shape[0] * grad



def softmax_regression_epoch(X, y, theta, lr = 0.1, batch=100):
    """

    Args:
        X (np.ndarray[np.float32]): 2D input array of size
            (num_examples x input_dim).
        y (np.ndarray[np.uint8]): 1D class label array of size (num_examples,)
        theta (np.ndarrray[np.float32]): 2D array of softmax regression
            parameters, of shape (input_dim, num_classes)
        lr (float): step size (learning rate) for SGD
        batch (int): size of SGD minibatch

    Returns:
        None
    """
    ### BEGIN YOUR CODE
    #产生一个随机数，选择solve_{1,2,3}()
    choice = np.random.randint(1,4)
    if choice == 1:
        solve_1(X, y, theta, lr, batch)
    elif choice == 2:
        solve_2(X, y, theta, lr, batch)
    else:
        solve_3(X, y, theta, lr, batch)

    ### END YOUR CODE
    


def ReLU(x):
    return np.maximum(0, x)

def sigma(x):
    return ReLU(x)

def nn_epoch(X, y, W1, W2, lr = 0.1, batch=100):
    """ Run a single epoch of SGD for a two-layer neural network defined by the
    weights W1 and W2 (with no bias terms):
        logits = ReLU(X * W1) * W2
    The function should use the step size lr, and the specified batch size (and
    again, without randomizing the order of X).  It should modify the
    W1 and W2 matrices in place.

    Args:
        X (np.ndarray[np.float32]): 2D input array of size
            (num_examples x input_dim).
        y (np.ndarray[np.uint8]): 1D class label array of size (num_examples,)
        W1 (np.ndarray[np.float32]): 2D array of first layer weights, of shape
            (input_dim, hidden_dim)
        W2 (np.ndarray[np.float32]): 2D array of second layer weights, of shape
            (hidden_dim, num_classes)
        lr (float): step size (learning rate) for SGD
        batch (int): size of SGD minibatch

    Returns:
        None
    """
    ### BEGIN YOUR CODE
    for i in range(0, X.shape[0], batch):
        Xb = X[i:i+batch]
        yb = y[i:i+batch]
        # below is for grad_w2 
        Z_2 = sigma(Xb @ W1)
        Hypo = Z_2 @ W2
        S = np.exp(Hypo) / np.sum(np.exp(Hypo), axis = 1, keepdims = True) 
        P = S                                                                #请注意这个安全操作哦
        P[np.arange(P.shape[0]), yb] -= 1
        grad_w2 = Z_2.T @ P
        #below is for grad_w1
        #对sigma 求导
        dZ_2 = (Z_2 > 0) * 1 #太妙了
        Tem1 = P @ W2.T
        Tem2 = Tem1 * dZ_2 # element-wise
        grad_w1 = Xb.T @ Tem2
        #descent
        W1 -= lr * grad_w1 / batch
        W2 -= lr * grad_w2 / batch

"""
    n = X.shape[0]
    for i in range(0, n,batch):
        Xb = X[i:i+batch]
        yb = y[i:i+batch]
        # Forward pass
        Z1 = Xb @ W1 # shape (batch, hidden_dim)
        A1 = np.maximum(Z1, 0) # is 0 here a matrix or scalar?
        Z2 = A1 @ W2
        Z2 -= Z2.max(axis=1, keepdims=True)
        expZ2 = np.exp(Z2) # shape (batch, num_classes)
        div = np.sum(expZ2, axis = 1, keepdims= True) # shape (batch, 1)
        P = expZ2 / div
        P[np.arange(yb.shape[0]), yb] -= 1  # shape (batch, num_classes)
        # Backward pass
        grad_W2 = (A1.T @ P) / yb.shape[0] 
        dA1 = P @ W2.T
        dZ1 = dA1 * (Z1 >0)
        grad_W1 = (Xb.T @ dZ1) / yb.shape[0]
        # Update weights
        W2 -= lr * grad_W2
        W1 -= lr * grad_W1
"""       

    ### END YOUR CODE



### CODE BELOW IS FOR ILLUSTRATION, YOU DO NOT NEED TO EDIT

def loss_err(h,y):
    """ Helper funciton to compute both loss and error"""
    return softmax_loss(h,y), np.mean(h.argmax(axis=1) != y)


def train_softmax(X_tr, y_tr, X_te, y_te, epochs=10, lr=0.5, batch=100,
                  cpp=False):
    """ Example function to fully train a softmax regression classifier """
    theta = np.zeros((X_tr.shape[1], y_tr.max()+1), dtype=np.float32)
    print("| Epoch | Train Loss | Train Err | Test Loss | Test Err |")
    for epoch in range(epochs):
        if not cpp:
            softmax_regression_epoch(X_tr, y_tr, theta, lr=lr, batch=batch)
        else:
            softmax_regression_epoch_cpp(X_tr, y_tr, theta, lr=lr, batch=batch)
        train_loss, train_err = loss_err(X_tr @ theta, y_tr)
        test_loss, test_err = loss_err(X_te @ theta, y_te)
        print("|  {:>4} |    {:.5f} |   {:.5f} |   {:.5f} |  {:.5f} |"\
              .format(epoch, train_loss, train_err, test_loss, test_err))


def train_nn(X_tr, y_tr, X_te, y_te, hidden_dim = 500,
             epochs=10, lr=0.5, batch=100):
    """ Example function to train two layer neural network """
    n, k = X_tr.shape[1], y_tr.max() + 1
    np.random.seed(0)
    W1 = np.random.randn(n, hidden_dim).astype(np.float32) / np.sqrt(hidden_dim)
    W2 = np.random.randn(hidden_dim, k).astype(np.float32) / np.sqrt(k)

    print("| Epoch | Train Loss | Train Err | Test Loss | Test Err |")
    for epoch in range(epochs):
        nn_epoch(X_tr, y_tr, W1, W2, lr=lr, batch=batch)
        train_loss, train_err = loss_err(np.maximum(X_tr@W1,0)@W2, y_tr)
        test_loss, test_err = loss_err(np.maximum(X_te@W1,0)@W2, y_te)
        print("|  {:>4} |    {:.5f} |   {:.5f} |   {:.5f} |  {:.5f} |"\
              .format(epoch, train_loss, train_err, test_loss, test_err))



if __name__ == "__main__":
    X_tr, y_tr = parse_mnist("data/train-images-idx3-ubyte.gz",
                             "data/train-labels-idx1-ubyte.gz")
    X_te, y_te = parse_mnist("data/t10k-images-idx3-ubyte.gz",
                             "data/t10k-labels-idx1-ubyte.gz")

    print("Training softmax regression")
    train_softmax(X_tr, y_tr, X_te, y_te, epochs=10, lr = 0.1)

    print("\nTraining two layer neural network w/ 100 hidden units")
    train_nn(X_tr, y_tr, X_te, y_te, hidden_dim=100, epochs=20, lr = 0.2)
