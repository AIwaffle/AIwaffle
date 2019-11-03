import numpy as np # linear algebra
import torch
import torch.nn.functional as F
from enum import Enum
import matplotlib.pyplot as plt


class SimpleClassificationModel():
    
    def __init__(self, layerNum, layerSizes, learningRate = 0.1):
        assert(len(layerSizes) == layerNum)
        
        self.loss = None
        self.learningRate = learningRate
        self.layerSizes = layerSizes
        
        self.activations = []
        self.weights = []
        for i in range(layerNum):
            self.activations.append(torch.zeros((1, layerSizes[i])))
        for i in range(layerNum - 1):
            self.weights.append(torch.rand((layerSizes[i], layerSizes[i + 1]), requires_grad=True))
            
    def forward(self, X):
        if isinstance(X, np.ndarray):
            X = torch.as_tensor(X, dtype=torch.float)
        X = X.reshape((1, self.layerSizes[0]))
        self.activations[0] = X
        for i in range(len(self.weights)):
            if i == len(self.weights) - 1:
                self.activations[i + 1] = F.sigmoid(torch.mm(self.activations[i], self.weights[i]))
            else:
                self.activations[i + 1] = F.relu(torch.mm(self.activations[i], self.weights[i]))
            
        return self.activations[-1]
    
    def compute_loss(self, Y):
        loss_fn = torch.nn.BCELoss(reduction='mean')
        self.loss = loss_fn(self.activations[-1], Y)
        return self.loss.item()
    
    def backward(self):
        for i in range(len(self.weights)):
            self.weights[i].grad.zero_()
        self.loss.backward()
        
    def optimize(self):
        with torch.no_grad():
            for i in range(len(self.weights)):
                self.weights[i] -= self.learningRate * self.weights[i].grad
                
    def get_grads(self):
        out = []
        for i in range(len(self.weights)):
            out.append(self.weights[i].grad)
        return out.numpy()
    
    def get_activations(self):
        return self.activations.numpy()
    
    def get_weights(self):
        return self.weights.numpy()
    

# demo        
model = SimpleClassificationModel(3, (2, 3, 1))
model.forward(np.array([1, 2]))
