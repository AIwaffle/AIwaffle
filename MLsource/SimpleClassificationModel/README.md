# The Simple Classification Model!
The documentation is contained in the code annotation.
Look at the code annotation to get started.

## Basic Usage in Pseudocode:
```
# in training
model = SimpleClassificationNetwork(size_list, learning_rate) # create model

for every epoch:
    all_activations = model.forward(X)
    loss = model.compute_loss(Y)
    model.backward()
    model.optimize()
    
# in prediction
just use Y_pred = model.forward(X)[-1],
Y_pred is a ndarray of shape (last_layer_size, )
you may want to apply softmax and use argmax to get the predicted
class.
```

## Notes (also contained in code annotation)
forward() will clear out the `grads` cached last time
compute_loss() must be called before backward()
backward() must be called before optimize()

get_params() and get_grads() returns List[`ndarray`] of the same shape,
Careful: weights and bias are seperated, so the list is of length 2 * linearLayerCount

The model auto inserted one ReLU layer between every 2 linear layers
Consequently the return value of forward() contains all activations including
ReLU layers.

You can use print(model) to check the overall structure of the model
