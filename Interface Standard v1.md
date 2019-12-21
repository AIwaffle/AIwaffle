# Interface Standard
### Abbreviations:
* W: weights
* b: bias
* A: activation values
* grad: gradient
* param: parameter
* forward/backward: forward/backward propagation
* prev, current: the size of the previous layer & the size of the current layer 
* layer_count: do not count the layers without weights (eg. ReLU)
* depth: count all layers

## Model -> Outside
| Name | Format |
|--|--|
| W | an array of length `layer_count`, each element is a (`current`, `prev`) matrix containing the W of current layer |
| b | an array of length `layer_count`, each element is a (`current`, ) matrix containing the bias |
| A | an array of length `depth`, each element is the a (`current`, ) matrix, containing the activation value of current layer. Especially, A[0] is the input, A[last] is the predicted output. |
| dW | same shape as W, containing gradients of every weight |
| db | same shape as b, containing gradients of every bias |
| loss | a scalar value, the loss on current input |

## Outside -> Model
This is just an abstract.
See [code annotation](https://github.com/IDl0T/AIwaffle/blob/master/MLsource/SimpleClassificationModel/SimpleClassificationNetwork.py) for more.

| operation | parameters |
| -- | -- |
| build | `size_list`, `learning_rate` |
| train | `step`, `data`: data includes input and ground truth |
| predict | `input` |

## Backend -> Frontend
|  |  |
|--|--|
|  |  |

## Frontend -> Backend
|  |  |
|--|--|
|  |  |


