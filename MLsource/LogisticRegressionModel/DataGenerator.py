import numpy as np


def generate_data(size, k, b, noise=0.):
    data = np.random.rand(size, 3)
    for i in range(size):
        if i <= size * (1 - noise):
            data[i, 2] = 1 if k * data[i, 0] + b > data[i, 1] else 0
        else:
            data[i, 2] = np.random.randint(0, 2)
    np.random.shuffle(data)
    return data


# demo
if __name__ == '__main__':
    data = generate_data(100, 1, 0, noise=0.2)
