# New Functional Version!
[Online Example](https://www.kaggle.com/idiott/logistic-regression-functional)

# Logistic Regression Documentation (for obsolete version)
## Basic usage pipeline
1. __init__()
2. forward(X)
3. compute_loss(Y)
4. backward()
5. optimize()
l
loop over 2, 3, 4, 5 over training data to train
## Things to display
### hyperparameters
* inputSize
* learningRate
### input
* forward(X) X is input
### output
* model.A or return value of forward()
### others (for fun)
* loss: model.loss / return value of compute_loss()
* gradients: model.dW, model.db
* parameters: model.W, model.b

## Data Generator
### prototype: generate_data(size, k, b, noise = 0.)
* k, b defines the decision boundary kx+b
* noise defines the proportion of the noise
