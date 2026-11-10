# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def _compute_area(x):
    """
    Compute the area of the union of the 64 triangles for given base positions x.
    x: array of shape (64,)
    """
    n = 64
    a = (np.arange(1, n + 1) / n)
    
    # Quadrature points
    m = np.arange(20000)
    y = (m + 0.5) / 20000.0
    
    # L_i(y) = x_i + a_i * y
    # x: (64,), a: (64,), y: (20000,)
    # Result L: (20000, 64)
    L = x[None, :] + y[:, None] * a[None, :]
    
    # Sort intervals at each height
    L_sorted = np.sort(L, axis=1)
    
    # Calculate union length
    # Width of each interval at height y is w = (1-y)/n
    w = (1.0 - y) / n
    
    # Differences between consecutive sorted starts
    diffs = L_sorted[:, 1:] - L_sorted[:, :-1]
    
    # Union length = w + sum(min(diffs, w))
    # diffs: (20000, 63), w: (20000,)
    # We need to broadcast w to match diffs
    union_len = w + np.sum(np.minimum(diffs, w[:, None]), axis=1)
    
    # Area is the mean of union lengths (since dy = 1/20000, sum * dy = mean)
    return np.mean(union_len)

def _compute_area_constrained(x_tail):
    """
    Compute area with x_0 fixed to 0.
    x_tail: array of shape (63,) corresponding to x_1 ... x_63
    """
    x = np.zeros(64)
    x[1:] = x_tail
    return _compute_area(x)

def kakeya_needle_2d_positions64() -> np.ndarray:
    n = 64
    a = (np.arange(1, n + 1) / n)
    
    start_time = time.time()
    deadline = start_time + 580.0 # Safety margin
    
    # Generate initializations (shifted so x_0 = 0)
    inits = []
    
    # 1. Aligned initialization
    # x_i = -a_i + 0.5
    x_aligned = -a + 0.5
    # Shift so x_0 = 0
    x_aligned_shifted = x_aligned - x_aligned[0]
    inits.append(x_aligned_shifted[1:]) # x_1 ... x_63
    
    # 2. Keich construction
    k = int(np.log2(n))
    j = np.arange(1, k + 1)
    x_keich = []
    for i in range(n):
        eps = np.array(list(str(bin(i))[2:]), dtype=int)
        eps = np.pad(eps, (k - len(eps), 0), "constant")
        x_keich.append(np.sum((1 - j) * eps * (2.0 ** -j) / k))
    x_keich = np.array(x_keich, dtype=np.float64)
    # x_keich[0] is 0
    inits.append(x_keich[1:])
    
    # 3. Random perturbations around aligned
    rng = np.random.default_rng(42)
    for _ in range(10):
        x_rand = x_aligned_shifted + rng.normal(0, 0.05, 63)
        inits.append(x_rand)
        
    # 4. Random perturbations around Keich
    for _ in range(10):
        x_rand = x_keich[1:] + rng.normal(0, 0.05, 63)
        inits.append(x_rand)
        
    best_x_tail = inits[0]
    best_area = _compute_area_constrained(best_x_tail)
    
    for x0 in inits:
        if time.time() > deadline:
            break
            
        # Use Nelder-Mead for non-smooth optimization
        # 63 variables. maxfev 5000 should be enough for good convergence.
        res = minimize(_compute_area_constrained, x0, method='Nelder-Mead', 
                       options={'maxiter': 5000, 'maxfev': 5000, 'xatol': 1e-8, 'fatol': 1e-10})
        
        if res.fun < best_area:
            best_area = res.fun
            best_x_tail = res.x
            
        # Optional: Polish with a second run if time permits?
        # For now, just move to next init.
        
    # Construct final x
    best_x = np.zeros(64)
    best_x[1:] = best_x_tail
    
    return best_x

# EVOLVE-BLOCK-END
