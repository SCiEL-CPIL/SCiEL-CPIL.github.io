# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time

def _smooth_area(x, n=64, M=20000, alpha=50.0):
    """
    Compute a smooth approximation of the area of the union of the n triangles.
    x: array of shape (n,)
    """
    y = (np.arange(M) + 0.5) / M
    a = (np.arange(n) + 1) / n
    L = x[None, :] + a[None, :] * y[:, None]
    L_sorted = np.sort(L, axis=1)
    w = (1.0 - y) / n
    d = np.diff(L_sorted, axis=1)
    
    # min(d, w) approx
    # log(exp(-alpha*d) + exp(-alpha*w))
    # Use logsumexp for stability
    # log(exp(a) + exp(b)) = max(a,b) + log(1 + exp(-|a-b|))
    # Here a = -alpha*d, b = -alpha*w
    # max(a,b) = -alpha * min(d,w)
    # diff = |a-b| = alpha * |d-w|
    # term = -alpha * min(d,w) + log(1 + exp(-alpha * |d-w|))
    # min(d,w) = -1/alpha * term
    
    # Let's just use np.logaddexp
    # np.logaddexp(x, y) = log(exp(x) + exp(y))
    # We want -1/alpha * log(exp(-alpha*d) + exp(-alpha*w))
    
    # To avoid underflow/overflow, we can use the identity:
    # log(exp(A) + exp(B)) = max(A,B) + log(1 + exp(-abs(A-B)))
    # A = -alpha*d, B = -alpha*w
    # max(A,B) = -alpha * min(d,w)
    # abs(A-B) = alpha * abs(d-w)
    
    # However, np.logaddexp is stable.
    # log_val = np.logaddexp(-alpha * d, -alpha * w)
    # min_approx = -log_val / alpha
    
    min_approx = -np.logaddexp(-alpha * d, -alpha * w) / alpha
    
    union_lengths = w + min_approx.sum(axis=1)
    area = np.mean(union_lengths)
    return area

def kakeya_needle_2d_positions64() -> np.ndarray:
    n = 64
    k = int(np.log2(n))
    
    # Keich construction initial guess
    j = np.arange(1, k + 1)
    positions = []
    for i in range(n):
        eps = np.array(list(str(bin(i))[2:]), dtype=int)
        eps = np.pad(eps, (k - len(eps), 0), "constant")
        positions.append(np.sum((1 - j) * eps * (2.0 ** -j) / k))
    x0 = np.array(positions, dtype=np.float64)
    
    # Define objective
    def objective(x):
        return _smooth_area(x, n, 20000, 50.0)
        
    start_time = time.time()
    deadline = start_time + 590.0
    
    try:
        # Use L-BFGS-B with numerical gradient
        # The smooth function allows for more reliable gradient estimation
        res = minimize(
            objective, 
            x0, 
            method='L-BFGS-B', 
            options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-10, 'disp': False}
        )
        
        if time.time() > deadline:
            return x0
            
        x_opt = res.x
        # Verify validity
        if not np.all(np.isfinite(x_opt)):
            return x0
            
        # Check improvement on the EXACT objective (non-smooth) to ensure the smooth approximation didn't lead to a worse true area
        exact_area_x0 = _compute_area_exact(x0, n, 20000)
        exact_area_xopt = _compute_area_exact(x_opt, n, 20000)
        
        if exact_area_xopt < exact_area_x0:
            return x_opt
        else:
            return x0
            
    except Exception:
        return x0

def _compute_area_exact(x, n=64, M=20000):
    y = (np.arange(M) + 0.5) / M
    a = (np.arange(n) + 1) / n
    L = x[None, :] + a[None, :] * y[:, None]
    L_sorted = np.sort(L, axis=1)
    w = (1.0 - y) / n
    d = np.diff(L_sorted, axis=1)
    contrib = np.minimum(d, w[:, None])
    union_lengths = w + contrib.sum(axis=1)
    area = np.mean(union_lengths)
    return area

# EVOLVE-BLOCK-END
