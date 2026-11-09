import numpy as np

def construct():
    n = 12
    # Chebyshev nodes of the first kind on [0, 15]
    # x_k = 0.5 * (L + R) + 0.5 * (R - L) * cos(pi * (2k + 1) / (2n))
    L, R = 0.1, 15.0
    k = np.arange(n)
    nodes = 0.5 * (L + R) + 0.5 * (R - L) * np.cos(np.pi * (2 * k + 1) / (2 * n))
    # Ensure sorted and positive
    nodes = sorted([float(z) for z in nodes])
    return nodes
