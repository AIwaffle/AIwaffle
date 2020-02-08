import random
import pprint

import numpy as np

import AIwaffle.MLsource.LogisticRegressionModel.DataGenerator as data_gen
import AIwaffle.MLsource.LogisticRegressionModel.functional as model


class LogisticRegressionModel:
    """
    Wrapped model
    """

    def __init__(self):
        # Random data seed
        self.k = (random.random() - 0.5) * 1e+9
        self.b = -0.5 * (self.k - 1)

        # Written according to functional.py
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

    def forward(self) -> np.ndarray:
        self.A = model.forward(self.X, self.W)
        return self.A

    def compute_loss(self) -> float:
        return model.compute_loss(self.A, self.Y)

    def evaluate(self) -> float:
        return model.evaluate(self.X, self.W, self.Y)

    def backward(self, learning_rate: float = 0.01) -> tuple:
        return model.backward(self.W, self.A, self.Y, self.X, learning_rate)

    def iterate(self, learning_rate: float = 0.01, epoch_num: int = 1) -> dict:
        # self.generate_data()
        a = list()
        loss = list()
        eval_ = list()
        for epoch in range(epoch_num):
            self.forward()
            a.append(self.A.tolist())
            loss.append(self.compute_loss())
            eval_.append(self.evaluate())
            self.W, self.dW = self.backward(learning_rate)
        res = dict(
            X=self.X.tolist(),
            Y=self.Y.tolist(),
            loss=loss,
            eval=eval_,
            avg_loss=sum(loss) / len(loss),
            A=a)
        for attr in ["W", "dW"]:  # Make W and dW 3-dimension
            v = self.__getattribute__(attr)
            if isinstance(v, np.ndarray):
                while not len(v.shape) == 3:
                    v = v[np.newaxis, :]
                v = v.tolist()
            res.update({attr: v})
        return res


if __name__ == '__main__':
    import json
    pp = pprint.PrettyPrinter(indent=4)
    m = LogisticRegressionModel()
    pp.pprint(m.iterate())
