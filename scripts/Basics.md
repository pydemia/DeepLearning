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
    weight1 = np.array([[.1, .3, .5], [.2, .4, .6]])

    # Apply An Activation Function in three hidden layers
    hidden_layer1_input  = np.dot(inputs, weight1) + bias1
    activation_function1 = lambda x: 1 / (1 + np.exp(-x)) # sigmoid function
    hidden_layer1_output = activation_function1(hidden_layer1_input)


    #------------------------ 2nd HIDDEN LAYER -------------------------------#
    # Three inputs and One bias
    hidden_layer1_output
    bias2 = np.array([.3, .1])

    # Weights directed to the 2nd two hidden layers
    weight2 = np.array([[.2, .4], [.1, .6], [.3, .5]])

    # Apply An Activation Function in two hidden layers
    hidden_layer2_input  = np.dot(hidden_layer1_output, weight2) + bias2
    activation_function2 = lambda x: 1 / (1 + np.exp(-x)) # sigmoid function
    hidden_layer2_output = activation_function2(hidden_layer2_input)


    #-------------------------- OUTPUT LAYER ---------------------------------#
    # Two inputs and One bias
    hidden_layer2_output
    bias3 = np.array([.3, .1])
    
    # Weights directed to the output layers
    weight3 = np.array([[.2, .4], [.1, .6]])

    # Apply An Activation Function in two output layers
    output_layer_input  = np.dot(hidden_layer2_output, weight3) + bias3
    activation_function3 = lambda x: x # identity function for regression
    #activation_function3 = lambda x: np.exp(x) / np.exp(x).sum(axis=0) # softmax function for classification
    output_layer_output = activation_function3(output_layer_input)


    #----------------------------- OUTPUT ------------------------------------#
    return output_layer_output

```
```python
neural_network_two_hidden_layers(.2, .3)

Out[]: array([ 0.50517432,  0.80259117])
```
