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
