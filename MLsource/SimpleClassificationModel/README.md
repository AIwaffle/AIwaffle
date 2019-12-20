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

