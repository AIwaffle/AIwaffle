# LogicRegression Documentation
## Basic usage pipeline
1. __init__()
2. load_data()
3. forward()
4. backward()
5. optimize()

this is the operation within 1 epoch
## Things to display
### hyperparameters
* inputSize
* learningRate
### input
* use load_data()
### output
* model.A or return value of forward()
### others (for fun)
* loss: model.loss / return value of compute_loss()
* gradients: model.dW, model.db
* parameters: model.W, model.b
