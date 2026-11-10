# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize

def kakeya_needle_2d_positions64() -> np.ndarray:
    n = 64
    a = np.arange(1, n + 1) / n
    
    # Quadrature points
    m = np.arange(20000)
    y = (m + 0.5) / 20000.0
    w = (1.0 - y) / n

    def compute_area(x):
        # x shape (n,)
        # L shape (20000, n)
        L = x[None, :] + a[None, :] * y[:, None]
        L_sorted = np.sort(L, axis=1)
        diffs = np.diff(L_sorted, axis=1)
        # Union length for each y
        # w is (20000,), needs to be (20000, 1) for broadcasting in sum?
        # np.sum returns (20000,)
        # w[:, None] is (20000, 1)
        # minimum(diffs, w[:, None]) is (20000, n-1)
        # sum(axis=1) is (20000,)
        # w[:, None] + ... wait, w is (20000,), so w + sum(...) works if sum is (20000,)
        # But w in the formula is the width, which is (1-y)/n.
        # The formula is: w + sum(min(diffs, w))
        # So:
        union_lengths = w + np.sum(np.minimum(diffs, w[:, None]), axis=1)
        return np.mean(union_lengths)

    # Generate initial guess using Keich construction
    k = int(np.log2(n))
    j = np.arange(1, k + 1)
    x_keich = []
    for i in range(n):
        eps = np.array(list(str(bin(i))[2:]), dtype=int)
        eps = np.pad(eps, (k - len(eps), 0), "constant")
        x_keich.append(np.sum((1 - j) * eps * (2.0 ** -j) / k))
    x0 = np.array(x_keich, dtype=np.float64)

    # Optimization settings
    options = {
        'maxiter': 5000,
        'xatol': 1e-9,
        'fatol': 1e-11
    }

    # Run Nelder-Mead from Keich start
    res = minimize(compute_area, x0, method='Nelder-Mead', options=options)
    
    best_x = res.x
    best_area = res.fun

    # Try a few random restarts to ensure global optimality
    # Use a fixed seed for reproducibility in this context
    rng = np.random.default_rng(42)
    for _ in range(10):
        # Initialize in a reasonable range
        # The optimal x_i are likely within [-1, 1]
        x_rand = rng.uniform(-1.0, 1.0, n)
        res_rand = minimize(compute_area, x_rand, method='Nelder-Mead', options=options)
        if res_rand.fun < best_area:
            best_area = res_rand.fun
            best_x = res_rand.x

    return best_x

# EVOLVE-BLOCK-END
