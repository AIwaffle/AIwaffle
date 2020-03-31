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

### Front End Abbreviations：

- layers: a list of depths for each layer (including the input and output layers)
- model:

| Name            | format        |
| --------------- | ------------- |
| weights (W)     | `float[][][]` |
| activations (A) | `float[][]`   |
| cost            | `float`       |

- model_name: name of a type of model (eg. `"housing price predictor"`)

## Backend -> Frontend

|  |  |
|--|--|
|  |  |

## Frontend -> Backend

| Operation | Parameters             | Description                                 |
| --------- | ---------------------- | ------------------------------------------- |
| new_model | `model_name`-> `model` | Create a new model of the type `model_name` |
| iterate   | () -> `model`          | Do one iteration on the model               |

In the tutorial setting, I think all the training data should be stored in the backend and the frontend is used to select which type of model to display and calling next iteration on user's demand.

