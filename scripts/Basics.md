# Basics on Deep Learning

## Perceptron

```python
#%% Perceptron

# AND GATE
def andgate(x1, x2):
    wght1, wght2, theta = .5, .5, .7
    y = wght1 * x1 + wght2 * x2
    res = 1 if y > theta else 0
    return res


andgate(0, 0)
andgate(1, 1)
andgate(1, 0)


#%% Weight & Bias
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

# Weight : A parameter leveraging the input signals.
#  Bias  : A control parameter How easily the result can be activated.
```
