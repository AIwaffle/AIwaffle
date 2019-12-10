# New Functional Version!
[Online Example](https://www.kaggle.com/idiott/logistic-regression-functional)

No more bias! the bias is merged into W by a trick.

Only needs one single loop! This is the vectorized version, can go over all data at one time.

# Logistic Regression Documentation
## Shapes
**n = number of dimensions + 1  
m = number of samples**  
In the online example, n = 2, m = 100  
If you want to go though your data one by one, let m = 1

**X (input): shape = (n + 1, m)  
W & dW (weights): shape = (1, n + 1)**  
Why n + 1? If we stack a line of ones on the top of X, then the first element in W serves as a bias.

**Y (ground truth): shape = (1, m)  
A (predicted): shape = (1, m)**

## Basic Usage Pipeline In 1 Epoch
1. forward(X, W) -> A
2. backward(W, A, Y, learning_rate) -> W(updated), dW
3. (optional) compute_loss(A, Y) - returns loss
4. (optional) evaluate(X, W, Y) - returns accuracy

## Data Generator
### prototype: generate_data(size, k, b, noise = 0.)
* k, b defines the decision boundary kx+b
* noise defines the proportion of the noise

---

# Logistic Regression Documentation (for obsolete version)
## Basic usage pipeline
1. __init__()
2. forward(X)
3. compute_loss(Y)
4. backward()
5. optimize()

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
