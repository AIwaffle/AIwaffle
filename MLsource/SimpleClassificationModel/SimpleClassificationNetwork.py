import numpy as np
import torch
from torch import nn


class SimpleClassificationNetwork(nn.Sequential):

    def __init__(self, size_list, learning_rate=0.01):
        """Create a model
        Args:
            size_list: a list or tuple denotes the layer number and the size of each layer,
                size_list[0] equals to input size, size_list[-1] equals to output size
            
        Returns:
            a SimpleNetwork instance
        """
        l = []
        for i in range(len(size_list) - 1):
            if i > 0:
                l.append(nn.ReLU())
            l.append(nn.Linear(size_list[i], size_list[i + 1]))

        super().__init__(*l)

        self.loss_fn = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.SGD(self.parameters(), lr=learning_rate, momentum=0.9)

        self.Y_pred = None
        self.loss = None

    def __call__(self, X):
        return self.forward(X)

    def forward(self, X):
        """Forward propagation
        Args:
            X: an 2D input Tensor/ndarray with shape (`batchsize`, `inputsize`)
        Returns:
            out: a List[ndarray] with the same shape as `size_list`,
                storing the activation value for each layer, 
                including activation layers (in this model, ReLU)
        """

        if type(X) == np.ndarray:
            X = torch.tensor(X)

        self.optimizer.zero_grad()
        out = [X.numpy().copy()]
        for i, module in enumerate(self._modules.values()):
            if type(module) == nn.CrossEntropyLoss:  # special case, thanks to pytorch mechanics
                continue

            X = module(X)
            out.append(X.detach().numpy().copy())
        self.Y_pred = X
        return out

    def compute_loss(self, Y):
        """Compute the loss
        Args:
            Y: the Tensor/ndarray storing ground truth, same length as Y_pred (in dimension 1),
                aka the batch size
        Returns:
            loss: a scalar, the loss of the model
        """
        if type(Y) == np.ndarray:
            Y = torch.tensor(Y)
        self.loss = self.loss_fn(self.Y_pred, Y)
        return self.loss.item()

    def backward(self):
        """Compute the gradients, make sure to call this before getting gradients,
            and make sure to call compute_loss() before backward()
        """
        self.loss.backward()

    def optimize(self):
        self.optimizer.step()

    def get_params(self):
        """get the parameters
        Returns:
            parameters: a list[ndarray], containing the parameters of each layer.
                len(parameters): number of layer * 2, first two elements denotes the weight and bias
                    for the first linear layer, and so on. Use ndarray.shape to check
                    the shapes
        """
        parameters = []
        for param in self.parameters():
            parameters.append(param.data.detach().numpy().copy())
        return parameters

    def get_grads(self):
        """get the gradients
        Returns:
            gradients: a list[ndarray] containing the grads of each layer,
                shape is the same as get_parameters
        """
        gradients = []
        for param in self.parameters():
            gradients.append(param.grad.numpy().copy())
        return gradients


# demo
if __name__ == '__main__':
    model = SimpleClassificationNetwork([2, 4, 3])
    for i in range(100):
        model.forward(torch.tensor([[1.0, 2.0]]))
        loss = model.compute_loss(torch.tensor([1]))
        model.backward()
        model.optimize()

    print(*model.get_params(), sep='\n')
    print(*model.get_grads(), sep='\n')
