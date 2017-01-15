# Training a Neural Network

In machine learning, We need to train a system with Training Datasets to minimize the loss.
It means the system get the values from ```Loss Function``` for each train-dataset and find out the parameter(weights & biases) make the summation of the values minimize.

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
----------------------
1e-50

np.float32(1e-50)
----------------------
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
----------------------------------------
0.19999999998909776

numericalDifferentiation(functionA, 10)
----------------------------------------
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
def patialFunctionA(X):
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
-----------------------------------------------
6.000000000128124

numericalDifferentiation(partialFunctionX1, 4.)
-----------------------------------------------
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
--------------------------------------------------------
array([ 6.,  8.])

numericalGradient(partialFunctionA, np.array([0., 2.]))
--------------------------------------------------------
array([ 0.,  4.])

numericalGradient(partialFunctionA, np.array([3., 0.]))
--------------------------------------------------------
array([ 6.,  0.])

```



