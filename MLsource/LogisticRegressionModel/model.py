import random
import pprint

import numpy as np

import AIwaffle.MLsource.LogisticRegressionModel.DataGenerator as data_gen
import AIwaffle.MLsource.LogisticRegressionModel.functional as l_model


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
        self.A = l_model.forward(self.X, self.W)
        return self.A

    def compute_loss(self) -> float:
        return l_model.compute_loss(self.A, self.Y)

    def evaluate(self) -> float:
        return l_model.evaluate(self.X, self.W, self.Y)

    def backward(self, learning_rate: float = 0.01) -> tuple:
        return l_model.backward(self.W, self.A, self.Y, self.X, learning_rate)

    def iterate(self, learning_rate: float = 0.01, epoch_num: int = 1) -> dict:
        # self.generate_data()
        loss = list()
        eval_ = list()
        for epoch in range(epoch_num):
            self.forward()
            loss.append(self.compute_loss())
            eval_.append(self.evaluate())
            self.W, self.dW = self.backward(learning_rate)
        res = dict(
            X=self.X[1:].tolist(),
            Y=self.Y.tolist(),
            loss=loss,
            accuracy=eval_,
            avg_loss=sum(loss) / len(loss),
            A=self.A.tolist(),
        )
        for attr in ["W", "dW"]:  # Make W and dW 3-dimension
            v = self.__getattribute__(attr)
            if isinstance(v, np.ndarray):
                while not len(v.shape) == 3:
                    v = v[np.newaxis, :]
                v = v.tolist()
            res.update({attr: v})
        return res


def plot_data(model: LogisticRegressionModel):
    # TODO: Use numpy (?)
    x1 = list()
    y1 = list()
    x2 = list()
    y2 = list()
    for i in range(len(model.X[0])):
        if model.Y[0][i] == 1.0:
            x1.append(model.X[1][i])
            y1.append(model.X[2][i])
        else:
            x2.append(model.X[1][i])
            y2.append(model.X[2][i])
    import matplotlib.pyplot as plt
    plt.scatter(x1, y1, color="red")
    plt.scatter(x2, y2, color="blue")
    plt.show()


if __name__ == '__main__':
    pp = pprint.PrettyPrinter(indent=4)
    m = LogisticRegressionModel()
    for i in range(10):
        print(m.iterate()["avg_loss"])
