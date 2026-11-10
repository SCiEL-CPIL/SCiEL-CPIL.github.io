# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time

def _compute_area(x, a, y, w, L_base):
    """
    Compute the area of the union of triangles.
    x: (64,) array of base positions
    a: (64,) array of slopes (a_i)
    y: (20000,) array of heights
    w: (20000,) array of widths at height y
    L_base: (64, 20000) array where L_base[i, m] = a[i] * y[m]
    """
    # L[i, m] = x[i] + a[i] * y[m]
    L = L_base + x[:, np.newaxis]
    
    # Sort L along axis 0 (for each height m, sort the 64 left edges)
    L_sorted = np.sort(L, axis=0)
    
    # Differences between consecutive sorted edges
    # d[i, m] = L_sorted[i+1, m] - L_sorted[i, m] for i=0..62
    d = L_sorted[1:, :] - L_sorted[:-1, :]
    
    # Union length at each height:
    # union_len[m] = w[m] + sum_i min(d[i, m], w[m])
    # min(d, w) can be computed as d - max(d - w, 0) or np.minimum(d, w)
    # Using np.minimum is direct.
    mins = np.minimum(d, w[np.newaxis, :])
    
    # Sum over i (axis 0)
    sum_mins = np.sum(mins, axis=0)
    
    # Total union length vector
    union_lens = w + sum_mins
    
    # Area is mean of union lengths
    return np.mean(union_lens)

def kakeya_needle_2d_positions64() -> np.ndarray:
    """
    Two-dimensional Kakeya-needle area-minimization problem.
    Returns 64 finite floats x_0..x_63.
    """
    n = 64
    k = int(np.log2(n))
    
    # Precompute static parts
    a = np.arange(1, n + 1, dtype=np.float64) / n
    num_heights = 20000
    y = (np.arange(num_heights, dtype=np.float64) + 0.5) / num_heights
    w = (1.0 - y) / n
    
    # L_base[i, m] = a[i] * y[m]
    L_base = np.outer(a, y)
    
    # Initial guess: Keich construction
    j = np.arange(1, k + 1)
    x_keich = np.zeros(n, dtype=np.float64)
    for i in range(n):
        eps = np.array(list(str(bin(i))[2:]), dtype=int)
        eps = np.pad(eps, (k - len(eps), 0), "constant")
        x_keich[i] = np.sum((1 - j) * eps * (2.0 ** -j) / k)
    
    # Optimization
    start_time = time.time()
    deadline = start_time + 600.0  # 10 minutes, leave 30s margin
    
    # Define objective function for optimizer
    def objective(x):
        return _compute_area(x, a, y, w, L_base)
    
    # We can try to improve upon the Keich solution
    # L-BFGS-B is a good choice for this size
    # We don't have explicit bounds, but x should be in a reasonable range.
    # The triangles are roughly in [x, x+1]. Overlaps happen.
    # Let's allow x to range from -1 to 2 to be safe, though unbounded is fine too.
    # Using bounds helps L-BFGS-B sometimes.
    bounds = [(-1.0, 2.0)] * n
    
    # Run optimization
    try:
        res = minimize(
            objective,
            x_keich,
            method='L-BFGS-B',
            bounds=bounds,
            options={
                'maxiter': 500,
                'ftol': 1e-12,
                'gtol': 1e-8,
                'maxls': 50,
                'maxcor': 10
            }
        )
        
        if res.success:
            x_opt = res.x
        else:
            # If it didn't converge perfectly, take the best found
            x_opt = res.x
        
        # Check time
        if time.time() > deadline:
            return x_keich # Fallback to safe initial if we ran out of time (unlikely)
            
        # Verify validity
        if not np.all(np.isfinite(x_opt)):
            return x_keich
            
        # Optional: A second pass with tighter tolerance if time permits?
        # For now, one good L-BFGS run should be strong.
        # Let's do a quick polish with Nelder-Mead or just return L-BFGS result.
        # L-BFGS-B is usually very good.
        
        return x_opt.astype(np.float64)

    except Exception as e:
        # If optimization fails, return the initial Keich solution
        return x_keich

# EVOLVE-BLOCK-END
