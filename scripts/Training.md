# Training a Neural Network

In machine learning, We need to train a system with Training Datasets to minimize the loss.
It means the system get the values from ```Loss Function``` for each train-dataset and find out the parameters(weights & biases) make the summation of the values minimize.

* mini-batch  
* get gradients  
* adjust parameters(weights, to the gradient direction)
* feedback  


## Loss Function

### Mean Squared Error(MSE)

```python
def meanSquaredError(yTrue, yPred):
    return np.sum((yTrue - yPred)**2) / 2

```

### Cross Entropy Error(CEE)

```python
def crossEntropyError(yTrue, yPred):
    delta = 1e-7
    return -np.sum(yPred * np.log(yTrue + delta))

```


## Mini-batch Training

(pick random 100ea from total 60,000ea)

```python
### Load a Dataset
import sys
import numpy as np
from PIL import Image

sys.path.append('/media/dawkiny/M/Git/DeepLearning/books/DeepLearningFromScratch')
from dataset.mnist import load_mnist

def showImage(image):
    pilImg = Image.fromarray(np.uint8(image))
    pilImg.show()

# Load Images: Flatten=True then it is loaded with 1D numpy
(x_train, t_train), (x_test, t_test) = load_mnist(flatten=True, normalize=False)

x_train.shape
x_test.shape

# Pick 100s
trainSize = x_train.shape[0]
batchSize = 100
batchMask = np.random.choice(trainSize, batchSize)
xBatch = x_train[batchMask]
yBatch = t_train[batchMask]



# Cross Entropy Error Function for batch
def crossEntropyError(yTrue, yPred):
    if y.ndim == 1:
        yTrue = yTrue.reshape(1, yTrue.size)
        yPred = yPred.reshape(1, yPred.size)
    
    batch_Size = y.shape[0]
    
    # Check if the number of label classes are more than 2(0,1)
    if len(yPred.unique()) == 2
        res = -np.sum(yPred * np.log(yTrue)) / batch_Size
    else len(yPred.unique()) > 2:
        res = -np.sum(yPred * np.log(yTrue[np.arange(batch_Size), t])) / batch_Size
    return 


img = x_train[0]
label = t_train[0]

print(label)
print(img.shape)

# Change 1D numpy to an image
img = img.reshape(28, 28)
print(img.shape)

```

## Gradients

### Numerical Differentiation

```python
def numericalDifferentiation(function, x):
    h = 1e-50
    return (function(x + h) - function(x)) / h
```

the above has 2 problems.

At first, if ```h``` is too small, it cannot be calculated by computer properly:
```python
np.float64(1e-50)
# Out []:
1e-50

np.float32(1e-50)
# Out []:
0.0
```

so it is recommended to use ```1e-5```


Secondary, its ```difference``` implies an error because the ```h``` is cannot be **_0_**.  
Instead, we can use ```central difference```: ```function(x+h) - function(x-h)```, not ```function(x+h) - function(x)```  

```python
def numericalDifferentiation(function, x):
    h = 1e-5
    return (function(x + h) - function(x - h)) / (2 * h)
```

### Example: Numerical Differentiation

Define a function:
```python
def functionA(x):
    return (.01 * x **2) + (.1 * x)
```

Plot it:
```python
import numpy as np
import matplotlib.pyplot as plt

X = np.arange(.0, 20., .1)
y = functionA(x)

plt.xlabel('x')
plt.ylabel('f(x)')
plt.plot(X, y)
plt.show()
```

Differentiate it:
```python
numericalDifferentiation(functionA, 5)
# Out []:
0.19999999998909776

numericalDifferentiation(functionA, 10)
# Out []:
0.29999999997532
```

Plot it:
```python
def tangentLine(function, x):
    d = numericalDifferentiation(function, x)
    print(d)
    y = function(x) - d*x
    return lambda t: d*t + y

tf = tangentLine(functionA, 5)
y2 = tf(x)

# Plot lines
plt.plot(X, y)
plt.plot(X, y2)

# the solution
solY = y[np.where(y2 == y)] # y = .75
solX = X[np.where(y2 == y)] # X = 5.
plt.plot(solX,solY, lw=2, c='k', marker='o')

plt.show()
```


### Partial Differentiation

Define a Partial Function:
>f(X0, X1) = X0\*\*2 + X1\*\*2

```python
def partialFunctionA(X):
    assert len(X) == 2
    return X[0]**2 + X[1]**2
```

If ```x0 = 3.```, ```x1 = 4.```,

Get the partial difference of ```X0```:
```python
def partialFunctionX0(X0):
    return X0**2 + 4.**2

def partialFunctionX1(X1):
    return 3.**2 + X1**2

numericalDifferentiation(partialFunctionX0, 3.)
# Out []:
6.000000000128124

numericalDifferentiation(partialFunctionX1, 4.)
# Out []:
7.999999999874773
```

### Gradients

Define a Gradient Function:
```python
def numericalGradient(function, X): 

    h = 1e-5
    grad = np.zeros_like(X)
    
    for _ in range(X.size):
        tmp_val = X[_]
        
        # function(x + h)
        X[_] = tmp_val + h
        fxh1 = function(X)
        
        # function(x - h)
        X[_] = tmp_val - h 
        fxh2 = function(X) # f(x-h)
        
        grad[_] = (fxh1 - fxh2) / (2*h)
    
    return grad
```

Get Gradients:
```python
numericalGradient(partialFunctionA, np.array([3., 4.]))
# Out []:
array([ 6.,  8.])

numericalGradient(partialFunctionA, np.array([0., 2.]))
# Out []:
array([ 0.,  4.])

numericalGradient(partialFunctionA, np.array([3., 0.]))
# Out []:
array([ 6.,  0.])

```


#### Gradient Descent
The optimal parameter means the parameter value that make ```Loss Function``` minimize. We usually apply ```Gradient Descent``` on finding the optimals and it premises that the gradient points to the optimal.  

Gradient Descent Function:
```python

def gradientDescent(function, initX, learningRate=.01, repeatStep=100):
    x = initX
    for _ in range(repeatStep):
        x -= learningRate * numericalGradient(function, x)
    return x

```

Use it:
```python
def gradientDescent(function, initX, learningRate=.01, repeatStep=100):
    x = initX
    for _ in range(repeatStep):
        x -= learningRate * numericalGradient(function, x)
    return x


def partialFunctionA(X):
    assert len(X) == 2
    return X[0]**2 + X[1]**2

def numericalGradient(function, X): 
    
    h = 1e-5
    grad = np.zeros_like(X)
    
    for _ in range(X.size):
        tmp_val = X[_]
        
        # function(x + h)
        X[_] = tmp_val + h
        fxh1 = function(X)
        
        # function(x - h)
        X[_] = tmp_val - h 
        fxh2 = function(X) # f(x-h)
        
        grad[_] = (fxh1 - fxh2) / (2*h)
    
    return grad


X = np.array([3., 4.])
gradientDescent(partialFunctionA, initX=X, learningRate=.01, repeatStep=100)
# Out []:
array([ 0.39742498,  0.53004453])
```

If ```learningRate``` is too big:
```python
X = np.array([3., 4.])
gradientDescent(partialFunctionA, X, learningRate=10., repeatStep=100)
# Out []:
array([ -4.73722866e+11,  -2.55129180e+12]) # -473722866000, -2551291800000
```

If ```learningRate``` is too small:
```python
X = np.array([3., 4.])
gradientDescent(partialFunctionA, X, learningRate=10., repeatStep=100)
# Out []:
array([ 2.99300693,  3.9910089 ])
```


```learningRate``` is called ```hyper parameter```, which means a parameter not be gained by an algorithm, but should be set by a man in manual.

#### Gradient in Neural Network
This gradient is the one of a ```Loss Function``` about the weights.  
When a 2x3 neural network with its weight __W__, its loss function *__L__*, the gradient is the ```partial difference``` of __W__.

```python
W = np.array([w11, w12, w13],
             [w21, w22, w23])
```
```python
pdW = np.array([pdw11, pdw12, pdw13],
               [pdw21, pdw22, pdw23])
```

#### Practice to get Gradients

Define a class for use:
```python
import numpy as np

class funcMod:
    def sigmoid(X):
        return 1 / (1 + np.exp(-X))


    def softmax(X):
        return np.exp(X) / np.exp(X).sum(axis=0)
 

    def crossEntropyError(yTrue, yPred):
        delta = 1e-7
        return -np.sum(yPred * np.log(yTrue + delta))


    def numericalGradient(function, X): 
    
        h = 1e-5
        grad = np.zeros_like(X)

        it = np.nditer(X, flags=['multi_index'], op_flags=['readwrite'])
        while not it.finished:
            idx = it.multi_index
            tmp_val = X[idx]
            X[idx] = float(tmp_val) + h
            fxh1 = f(X)

            X[idx] = tmp_val - h 
            fxh2 = f(X)
            grad[idx] = (fxh1 - fxh2) / (2*h)

            X[idx] = tmp_val
            it.iternext()   

        return grad
```

Define a Neural Network:
```python
class simpleNet:
    def __init__(self):
        self.W = np.random.randn(2,3)
        
    def predict(self, X):
        return np.dot(X, self.W)
    
    def loss(self, X, t):
        z = self.predict(X)
        y = funcMod.softmax(z)
        loss = funcMod.crossEntropyError(y, t)
        
        return loss
```

Test it:
```python
net = simpleNet()
net.W
Out[]: 
array([[-2.08127481, -0.47733376,  0.11947171],
       [ 0.44312669, -1.02350389,  0.29610165]])

X = np.array([.6, .9])
p = net.predict(X)
Out[]: 
array([-0.84995087, -1.20755376,  0.33817451])


t = np.array([0, 0, 1])
net.loss(x, t)
Out[]:
0.41735971668735855


def dummyFunc(W):
    return net.loss(X, t)

dW = funcMod.numericalGradient(dummyFunc, net.W)
dW
Out[]: 
array([[ 0.28330426,  0.19981389, -0.48311814],
       [ 0.42495638,  0.29972083, -0.72467721]])
```

Each ```dW``` is the gradient of __W__;a weight.  


### Implement
