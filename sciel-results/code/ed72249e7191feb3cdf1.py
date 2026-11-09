import numpy as np
from scipy.optimize import minimize

def kakeya_needle_2d_positions64() -> np.ndarray:
    n = 64
    a = np.arange(1, n + 1) / n
    y = (np.arange(20000) + 0.5) / 20000.0
    w = (1 - y) / n
    
    def area(x):
        # x is shape (n,)
        # L_i = x_i + a_i * y
        # L shape (n, len(y))
        L = x[:, None] + a[:, None] * y[None, :]
        L_sorted = np.sort(L, axis=0)
        diffs = np.diff(L_sorted, axis=0)
        # union length = w + sum(min(diffs, w))
        union_len = w[None, :] + np.sum(np.minimum(diffs, w[None, :]), axis=0)
        return np.mean(union_len)

    # Initial guess: Keich construction from the prompt
    k = int(np.log2(n))
    j = np.arange(1, k + 1)
    x_keich = []
    for i in range(n):
        eps = np.array(list(str(bin(i))[2:]), dtype=int)
        eps = np.pad(eps, (k - len(eps), 0), "constant")
        x_keich.append(np.sum((1 - j) * eps * (2.0 ** -j) / k))
    x0 = np.array(x_keich, dtype=np.float64)
    
    # Optimize
    # Powell is good for non-smooth functions
    res = minimize(area, x0, method='Powell', options={'maxiter': 1000, 'disp': False})
    
    return res.x
