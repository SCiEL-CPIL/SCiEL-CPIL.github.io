import numpy as np
from scipy.optimize import minimize

def compute_area(x, n=64, M=20000):
    # x: array of shape (n,)
    # a_i = (i+1)/n
    a = (np.arange(n) + 1) / n
    # y_m = (m + 0.5) / M
    y = (np.arange(M) + 0.5) / M
    # w(y) = (1-y)/n
    w = (1 - y) / n
    
    # L_i(y) = x_i + a_i * y
    # Shape: (n, M)
    L = x[:, np.newaxis] + a[:, np.newaxis] * y[np.newaxis, :]
    
    # Sort L at each y
    # L_sorted[m, i] is the i-th smallest L at height y_m
    L_sorted = np.sort(L, axis=0)
    
    # Diffs
    # diffs[m, i] = L_sorted[m, i+1] - L_sorted[m, i]
    diffs = np.diff(L_sorted, axis=0)
    
    # Union length at each y
    # U(y) = w(y) + sum( min(diffs, w(y)) )
    # min(diffs, w)
    mins = np.minimum(diffs, w[np.newaxis, :])
    union_lengths = w + np.sum(mins, axis=0)
    
    # Area is mean of union lengths
    return np.mean(union_lengths)

def kakeya_needle_2d_positions64() -> np.ndarray:
    n = 64
    # Baseline: Keich Construction
    k = int(np.log2(n))
    j = np.arange(1, k + 1)
    keich_pos = np.array([
        np.sum((1 - j) * np.array(list(str(bin(i))[2:]), dtype=int)[::-1][:k] * (2.0 ** -j) / k)
        for i in range(n)
    ])
    
    # Note: The bin string logic in the original code was:
    # eps = np.array(list(str(bin(i))[2:]), dtype=int)
    # eps = np.pad(eps, (k - len(eps), 0), "constant")
    # This pads on the LEFT. 
    # My quick check above used [::-1][:k] which is wrong if not careful.
    # Let's stick to the exact logic of the current program to ensure baseline is correct.
    
    positions_list = []
    for i in range(n):
        eps = np.array(list(str(bin(i))[2:]), dtype=int)
        eps = np.pad(eps, (k - len(eps), 0), "constant")
        positions_list.append(np.sum((1 - j) * eps * (2.0 ** -j) / k))
    keich_pos = np.array(positions_list, dtype=np.float64)
    
    best_x = keich_pos
    best_area = compute_area(keich_pos)
    
    # Try a few other initializations
    inits = [
        keich_pos,
        np.linspace(0, 1 - 1/n, n), # Linear
        np.random.uniform(0, 1, n), # Random
    ]
    
    # Optimize each initialization
    for x0 in inits:
        # Use Nelder-Mead for non-smooth function
        try:
            res = minimize(
                compute_area, 
                x0, 
                method='Nelder-Mead', 
                options={'maxiter': 2000, 'xatol': 1e-6, 'fatol': 1e-8}
            )
            if res.success and res.fun < best_area:
                best_area = res.fun
                best_x = res.x
        except Exception:
            pass
            
    return best_x
