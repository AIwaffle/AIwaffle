import numpy as np
from matplotlib import pyplot as plt

def generate_data(size, k, b, noise = 0.):
    data = np.random.rand(size, 3)
    for i in range(size):
        if i <= size * (1 - noise):
            data[i, 2] = 1 if k * data[i, 0] + b > data[i, 1] else 0
        else:
            data[i, 2] = np.random.randint(0, 2)
    np.random.shuffle(data)
    return data

# demo
data = generate_data(100, 1, 0, noise=0.2)
plt.scatter(data[:, 0], data[:, 1], c = data[:, 2])
