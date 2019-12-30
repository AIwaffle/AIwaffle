import random

import numpy as np

import AIwaffle.MLsource.LogisticRegressionModel.DataGenerator as data_gen
import AIwaffle.MLsource.LogisticRegressionModel.functional as model


class LogisticRegressionModel:
    """
    Wrapped model
    """

    def __init__(self):
        self.k = (random.random() - 0.5) * 1e+9
        self.b = -0.5 * (self.k - 1)
        self.Y = None
        self.X = None
        self.data = None
        self.n = None
        self.m = None
        self.generate_data()
        self.W = np.random.randn(1, self.n + 1)
        self.dW = None
        self.A = None

    def generate_data(self) -> None:
        self.data = data_gen.generate_data(100, self.k, self.b, noise=0.2)
        self.X = self.data.T[0:2, :]
        self.Y = self.data.T[2, :]
        self.Y = self.Y.reshape((1, -1))
        self.n = self.X.shape[0]
        self.m = self.X.shape[1]
        self.X = np.vstack((np.ones((1, self.m)), self.X))

    def iterate(self, learning_rate=0.01) -> dict:
        self.generate_data()
        loss = list()
        eval_ = list()
        for epoch in range(1000):
            self.A = model.forward(self.X, self.W)
            if epoch % 50 == 0:
                loss.append(model.compute_loss(self.A, self.Y))
                eval_.append(model.evaluate(self.X, self.W, self.Y))
            self.W, self.dW = model.backward(self.W, self.A, self.Y, self.X, learning_rate)
        res = dict(loss=loss, eval=eval_)
        for attr in ["W", "dW", "A"]:
            v = self.__getattribute__(attr)
            if isinstance(v, np.ndarray):
                v = v.tolist()
            res.update({attr: v})
        return res


if __name__ == '__main__':
    m = LogisticRegressionModel()
    print(m.iterate())
