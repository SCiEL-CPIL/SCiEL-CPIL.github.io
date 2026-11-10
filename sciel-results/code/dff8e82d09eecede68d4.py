import numpy as np
import time
from scipy.optimize import minimize

def kakeya_needle_2d_positions64() -> np.ndarray:
    """
    Minimizes the area of the union of 64 Kakeya needle triangles.
    Uses Keich construction as baseline and refines with L-BFGS-B
    using a vectorized subgradient.
    """
    n = 64
    M = 20000
    
    # Precompute constants
    a = (np.arange(n) + 1) / n
    y = (np.arange(M) + 0.5) / M
    w = (1 - y) / n
    w_B = w[:, None]  # shape (M, 1)
    
    # Generate Keich baseline
    k = int(np.log2(n))
    x0 = np.zeros(n, dtype=np.float64)
    for i in range(n):
        bin_str = bin(i)[2:]
        eps = np.array(list(bin_str), dtype=int)
        eps = np.pad(eps, (k - len(eps), 0), "constant")
        j_idx = np.arange(1, k + 1)
        weights = (1 - j_idx) * (2.0 ** -j_idx) / k
        x0[i] = np.sum(eps * weights)
    
    def area_and_grad(x):
        # L[m, i] = x[i] + a[i] * y[m]
        L = x[None, :] + (a * y[:, None])
        
        # Sort L and get indices
        sorted_L, idx = np.sort(L, axis=1), np.argsort(L, axis=1)
        
        # Gaps between consecutive sorted intervals
        gaps = sorted_L[:, 1:] - sorted_L[:, :-1]  # shape (M, n-1)
        
        # Union length at each height
        min_gaps = np.minimum(gaps, w_B)
        union_lengths = w + np.sum(min_gaps, axis=1)
        area = np.mean(union_lengths)
        
        # Compute subgradient
        # For each y (row m), the gradient contribution from gap j is:
        # If gaps[m,j] < w[m]:
        #   +a[idx[m, j+1]] to component idx[m, j+1]
        #   -a[idx[m, j]] to component idx[m, j]
        
        mask = gaps < w_B  # shape (M, n-1)
        
        # Flatten mask and find true indices
        flat_mask = mask.ravel()
        true_indices = np.nonzero(flat_mask)[0]
        
        if len(true_indices) == 0:
            grad = np.zeros(n, dtype=np.float64)
        else:
            m_idx, j_idx = np.unravel_index(true_indices, (M, n - 1))
            
            # Plus part: add a[idx[m, j+1]] to grad[idx[m, j+1]]
            plus_cols = idx[m_idx, j_idx + 1]
            plus_vals = a[plus_cols]
            
            # Minus part: subtract a[idx[m, j]] from grad[idx[m, j]]
            minus_cols = idx[m_idx, j_idx]
            minus_vals = a[minus_cols]
            
            # Accumulate using np.add.at
            grad = np.zeros(n, dtype=np.float64)
            np.add.at(grad, plus_cols, plus_vals)
            np.add.at(grad, minus_cols, -minus_vals)
            grad /= M  # Average over M heights
        
        return area, grad
    
    # Optimize
    start_time = time.time()
    deadline = start_time + 550  # Safety margin
    
    try:
        res = minimize(
            area_and_grad,
            x0,
            method='L-BFGS-B',
            jac=True,
            options={'maxiter': 150, 'ftol': 1e-12, 'gtol': 1e-9, 'maxls': 20}
        )
        x_opt = res.x
    except Exception:
        x_opt = x0
    
    # Check time
    if time.time() > deadline:
        return x0
    
    # Validate
    if not np.all(np.isfinite(x_opt)):
        return x0
    
    return x_opt
