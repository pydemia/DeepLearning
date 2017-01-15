# Basics on Deep Learning

## Perceptron

### Input Layer and Output Layer
```python

def ANDgate(input1, input2):

    weight1, weight2, threshold_theta = .5, .5, .7
    netvalue = weight1 * input1 + weight2 * input2
    activation_function = lambda x: 1 if x > threshold_theta else 0
    output = activation_function(netvalue)

    return output


andgate(0, 0)
andgate(1, 1)
andgate(1, 0)
```

### Weight & Bias
```python
import numpy as np

def ANDgate(*args):

    inputs = np.array(*args)
    weight = .5
    bias = -.6

    netvalue = np.sum(inputs * weight) + bias
    threshold = 0
    output = 1 if netvalue > threshold else 0
    return output
```

>Weight : A parameter leveraging the input signals.
> Bias  : A control parameter How easily the result can be activated.


### AND, NAND, OR gate Test

Definition:
```python

def ANDgate(*args):

    inputs = np.array(*args)
    weight = .5
    bias = -.6

    netvalue = np.sum(inputs * weight) + bias
    threshold = 0
    output = 1 if netvalue > threshold else 0
    return output

def NANDgate(*args):

    inputs = np.array(*args)
    weight = -.5
    bias = .6

    netvalue = np.sum(inputs * weight) + bias
    threshold = 0
    output = 1 if netvalue > threshold else 0
    return output


def ORgate(*args):

    inputs = np.array(*args)
    weight = .5
    bias = -.3

    netvalue = np.sum(inputs * weight) + bias
    threshold = 0
    output = 1 if netvalue > threshold else 0
    return output

```

Test with the truth table:
```python
ttbl = np.array([[0,0],
                [0,1],
                [1,0],
                [1,1]])

for _ in ttbl:
    print(_, ':\n', ANDgate(_), NANDgate(_), ORgate(_))

Out []:
[0 0] :
 0 1 0
[0 1] :
 0 1 1
[1 0] :
 0 1 1
[1 1] :
 1 0 1
```

### XOR gate with ```multi-layer perceptron```: ```non-linear```

Enter inputs to both of NAND and OR, and then each outputs are inputs of AND:

```python

def XORgate(*args):
    
    s1 = NANDgate(*args)
    s2 = ORgate(*args)
    output2 = ANDgate(np.array([s1, s2]))
    return output2


for _ in ttbl:
    print(_, ':\n', XORgate(_))

Out []:
[0 0] :
 0
[0 1] :
 1
[1 0] :
 1
[1 1] :
 0

```

## Neural Network

### Hidden Layer and Activation Fuction

Look at this:
```python
def ANDgate(input1, input2):

    weight1, weight2, threshold_theta = .5, .5, .7
    netvalue = weight1 * input1 + weight2 * input2
    activation_function = lambda x: 1 if x > threshold_theta else 0
    output = activation_function(netvalue)

    return output
```
In this perceptron(neuron), the activation function is a ```step function```.
Actually, It is the key that to apply other functions as a complement in Neural Network. ```sigmoid function``` is commonly used. as its name represents, sigmoid results can control the output in more detail, not **_0_** or **_1_**.  
Recently, ```ReLU(Rectified Linear Unit) function``` is an alternative since the problem of ```sigmoid function``` arises.  
```ReLU``` is a combination of ```step function``` and ```linear function```. It returns **_0_** if the input is below **_0_**, or return the input itself.

#### ```step function```

```python

def stepFunc(x):
    x = np.array(x).astype(np.int)
    res = 1 if x > 0 else 0
    return res

```

#### ```sigmoid function``` for Hidden Layer

```python

def sigmoid(x):
    res = 1 / (1 + np.exp(-x))
    return res

sigmoid = lambda x: 1 / (1 + np.exp(-x))

```

#### ```ReLU``` function for Hidden Layer

```python

def ReLU(x):
    res = np.maximum(0, x)
    return res

```


#### ```identity function``` for Regression
```python
def identityFunc(x):
    res = x
    return res

```

#### ```softmax``` for Classification
```python
def softmax(x):
    res = np.exp(x) / np.exp(x).sum(axis=0)
    return res

softmax = lambda x: np.exp(x) / np.exp(x).sum(axis=0)
    
```

To prevent overflow:
```python
def softmax(x):
    C = np.max(x)
    res = np.exp(x-C) / np.exp(x-C).sum(axis=0)
    return res

softmax = lambda x: np.exp(x) / np.exp(x).sum(axis=0)
    
```
That' because when computing softmax, the intermediate values may become very large. Dividing two large numbers can be numerically unstable.  
Calculating an array with ```softmax```, each output is a real number between *__0__* and *__1__* and the summation of it is *__1__*. This feature can help ```softmax``` operate the classification.  
For your information, ```softmax``` is normally used when training a model and skip it during the inference step. Because only the maximum output is selected as a result in classification and the ```softmax``` cannot affect the magnitude relationship among the output.

### Inner Product and Neural Network

Look at it first:
```python
Inputs = np.array([1, 2])
Out[]: array([1, 2])

Weight = np.array([[1, 3, 5], [2, 4, 6]])
Out[]: 
array([[1, 3, 5],
       [2, 4, 6]])

Netwrk = np.dot(inputs, weight)
Out[]: array([ 5, 11, 17])
```

In this case, the inputs are delivered each(3) ```hidden layer```s with the weights. It can be calcultated by ```inner product```.


```python

def neural_network(input1, input2):


```
### Hidden Layer

#### Define a ```neural network``` with 2 hidden layers
```python
def neural_network_two_hidden_layers(input1, input2):

    #--------------------------- INPUT LAYER ---------------------------------#
    # Two inputs and One bias
    inputs = np.array([input1, input2])
    bias1 = np.array([.4, .1, .2])


    #------------------------ 1st HIDDEN LAYER -------------------------------#
    # Weights directed to the 1st three hidden layers
    WEIGHT1 = np.array([[.1, .3, .5], [.2, .4, .6]])

    # Apply An Activation Function in three hidden layers
    hidden_layer1_input  = np.dot(inputs, WEIGHT1) + bias1
    activation_function1 = lambda x: 1 / (1 + np.exp(-x)) # sigmoid function
    hidden_layer1_output = activation_function1(hidden_layer1_input)


    #------------------------ 2nd HIDDEN LAYER -------------------------------#
    # Three inputs and One bias
    hidden_layer1_output
    bias2 = np.array([.3, .1])


    # Weights directed to the 2nd two hidden layers
    WEIGHT2 = np.array([[.2, .4], [.1, .6], [.3, .5]])

    # Apply An Activation Function in two hidden layers
    hidden_layer2_input  = np.dot(hidden_layer1_output, WEIGHT2) + bias2
    activation_function2 = lambda x: 1 / (1 + np.exp(-x)) # sigmoid function
    hidden_layer2_output = activation_function2(hidden_layer2_input)


    #-------------------------- OUTPUT LAYER ---------------------------------#
    # Two inputs and One bias
    hidden_layer2_output
    bias3 = np.array([.3, .1])
    
    # Weights directed to the output layers
    WEIGHT3 = np.array([[.2, .4], [.1, .6]])

    # Apply An Activation Function in two output layers
    output_layer_input  = np.dot(hidden_layer2_output, WEIGHT3) + bias3
    activation_function3 = lambda x: x # identity function for regression
    #activation_function3 = lambda x: np.exp(x) / np.exp(x).sum(axis=0) # softmax function for classification
    output_layer_output = activation_function3(output_layer_input)


    #----------------------------- OUTPUT ------------------------------------#
    return output_layer_output

```

Run it:
```python
neural_network_two_hidden_layers(.2, .3)

Out[]: array([ 0.50517432,  0.80259117])
```

### Output Layer

In general, the number of neurons on output layers are setted to the same number of the class to classify.

## Test with the MNIST Dataset(Only the inference step)

Definition
```python
### Load a Dataset
import sys
import numpy as np
import pickle
from PIL import Image

sys.path.append('/media/dawkiny/M/Git/DeepLearning/books/DeepLearningFromScratch')
from dataset.mnist import load_mnist

# Load Images: Flatten=True then it is loaded with 1D numpy
(x_train, t_train), (x_test, t_test) = load_mnist(flatten=True, normalize=False)

x_train.shape
x_test.shape

def showImage(image):
    pilImg = Image.fromarray(np.uint8(image))
    pilImg.show()


img = x_train[0]
label = t_train[0]

print(label)
print(img.shape)

# Change 1D numpy to an image
img = img.reshape(28, 28)
print(img.shape)

#%%
### Inference 
# Input neuron: 28x28=784
# Hidden layer: 2(1st=50, 2nd=100 heuristic)
# Output neuron: 0~9=10ea

def getData():
    # Load Images: Flatten=True then it is loaded with 1D numpy
    (x_train, t_train), (x_test, t_test) = load_mnist(flatten=True,
                                                      normalize=False,
                                                      one_hot_label=False)
    return x_test, t_test


def initNetwork():
    
    # Load the pre-learned weight parameters
    with open('sample_weight.pkl', 'rb') as f:
        network = pickle.load(f)
    
    return network


def predict(network, inputs):
    
    # Load the parameters
    WEIGHT1, WEIGHT2, WEIGHT3 = network['W1'], network['W2'], network['W3']
    bias1, bias2, bias3 = network['b1'], network['b2'], network['b3']
    
    #------------------------ 1st HIDDEN LAYER -------------------------------#
    hidden_layer1_input  = np.dot(inputs, WEIGHT1) + bias1
    activation_function1 = lambda x: 1 / (1 + np.exp(-x)) # sigmoid function
    hidden_layer1_output = activation_function1(hidden_layer1_input)
    
    #------------------------ 2nd HIDDEN LAYER -------------------------------#
    hidden_layer2_input  = np.dot(hidden_layer1_output, WEIGHT2) + bias2
    activation_function2 = lambda x: 1 / (1 + np.exp(-x)) # sigmoid function
    hidden_layer2_output = activation_function2(hidden_layer2_input)
    
    #-------------------------- OUTPUT LAYER ---------------------------------#
    output_layer_input  = np.dot(hidden_layer2_output, WEIGHT3) + bias3
    activation_function3 = lambda x: np.exp(x) / np.exp(x).sum(axis=0) # softmax function for classification
    output_layer_output = activation_function3(output_layer_input)
    
    return output_layer_output

```

Test it(one by one)
```python
x, t = getData()
network = initNetwork()

accuracyCount = 0
for _ in range(len(x)):
    y = predict(network, x[_])
    p = np.argmax(y)
    accuracyCount += 1 if p == t[_] else 0

print('Accuracy :' + str(float(accuracyCount) / len(x)))
Out []:
Accuracy :0.9207
```


Batch Test(x100 a time)
```python

x, t = getData()
network = initNetwork()

batchSize = 100
accuracyCount = 0

for _ in range(0, len(x), batchSize):
    batchX = x[_:(_ + batchSize)]
    batchY = predict(network, batchX)
    p = np.argmax(batchY, axis=1)
    accuracyCount += np.sum(p == t[_:(_ + batchSize)])

print('Accuracy :' + str(float(accuracyCount) / len(x)))
Out []:
Accuracy :0.9135
```
