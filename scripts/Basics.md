# Basics on Deep Learning

## Perceptron

```python
# AND GATE
def andgate(x1, x2):
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

def andgate(x=[x1, x2], w=[.3, .7]):
   
    assert len(x) == len(w)
    assert np.sum(w) == 1
    
    xArr = np.array(x)
    wArr = np.array(w)
    bias = -.6
    
    y = np.sum(xArr * wArr) + bias
    res = 1 if y > 0 else 0
    return res
```

>Weight : A parameter leveraging the input signals.
> Bias  : A control parameter How easily the result can be activated.


#### AND, NAND, OR gate Test

Definition
```python

def ANDgate(x=None, w=[.5, .5]):

    xArr = np.array(x)
    wght = .5
    bias = -.6

    y = np.sum(xArr * wght) + bias
    res = 1 if y > 0 else 0
    return res

def NANDgate(x=None, w=[-.5, -.5]):

    xArr = np.array(x)
    wght = -.5
    bias = .6

    y = np.sum(xArr * wght) + bias
    res = 1 if y > 0 else 0
    return res


def ORgate(x=None, w=[.5, .5]):

    xArr = np.array(x)
    wght = .5
    bias = -.3

    y = np.sum(xArr * wght) + bias
    res = 1 if y > 0 else 0
    return res
```

Test with the truth table
```python
ttbl = np.array([[0,0],
                [0,1],
                [1,0],
                [1,1]])

for _ in ttbl:
    print(_, ':\n', ANDgate(x=_), NANDgate(x=_), ORgate(x=_))

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

