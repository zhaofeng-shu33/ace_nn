# ace_nn

## Introduction

This repo contains experimental implementation of ace algorithm via neural network. It is shown by **xiangxiang-xu** that calculating optimal features by *Alternating Conditional Expectation* is equivalent to maximize *H-score*. 

## How to run

Three examples are provided ( one for continuous variable and the other twos are for discrete variable) and their results are the same as `ace`. The main function is `ace_nn` and its parameters are very similar to [`ace_cream`](https://github.com/zhaofeng-shu33/ace_cream). 

```python
import numpy as np
from ace_nn import ace_nn
# discrete case, binary symmetric channel with crossover probability 0.1
x = np.random.choice([0,1], size=N_SIZE)
n = np.random.choice([0,1], size=N_SIZE, p=[0.9, 0.1])
y = np.mod(x + n, 2)
# set both x(cat=0) and y(cat=-1) as categorical type
tx, ty = ace_nn(x, y, cat=[-1,0], epochs=100)

# continuous case
x = np.random.uniform(0, np.pi, 200)
y = np.exp(np.sin(x)+np.random.normal(size=200)/2)
tx, ty = ace_nn(x, y)
```

For more detail, run `help(ace_nn)` to see the parameters and returns of this function.

## Further discussion

Currently, the neural networks used to approximate optimal $f(x)$ and $g(y)$ are two-layer MLP with `tanh` as activation function. More turns of epochs are needed for large alphabet  $|\mathcal{X}|$ and  $|\mathcal{Y}|$ and the running time is not short.

Also, `batch_size` and `hidden_units_num` can be hypertuned, and there is no guarantee that current configuration of neural network is optimal for solving ace.