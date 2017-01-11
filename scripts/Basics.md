# Basics on Deep Learning

## Perceptron

```python
# AND GATE
def ANDgate(x1, x2):
    wght1, wght2, theta = .5, .5, .7
    y = wght1 * x1 + wght2 * x2
    res = 1 if y > theta else 0
    return res


andgate(0, 0)
andgate(1, 1)
andgate(1, 0)
```

## Weight & Bias
```python
import numpy as np

def ANDgate(*args):

    xArr = np.array(*args)
    wght = .5
    bias = -.6

    y = np.sum(xArr * wght) + bias
    res = 1 if y > 0 else 0
    return res
```

>Weight : A parameter leveraging the input signals.
> Bias  : A control parameter How easily the result can be activated.


#### AND, NAND, OR gate Test

Definition:
```python

def ANDgate(*args):

    xArr = np.array(*args)
    wght = .5
    bias = -.6

    y = np.sum(xArr * wght) + bias
    res = 1 if y > 0 else 0
    return res

def NANDgate(*args):

    xArr = np.array(*args)
    wght = -.5
    bias = .6

    y = np.sum(xArr * wght) + bias
    res = 1 if y > 0 else 0
    return res


def ORgate(*args):

    xArr = np.array(*args)
    wght = .5
    bias = -.3

    y = np.sum(xArr * wght) + bias
    res = 1 if y > 0 else 0
    return res
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

#### XOR gate with ```multi-layer perceptron```: ```non-linear```

Enter inputs to both of NAND and OR, and then each outputs are inputs of AND:

```python

def XORgate(*args):
    
    s1 = NANDgate(*args)
    s2 = ORgate(*args)
    res = ANDgate(np.array([s1, s2]))
    return res


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

