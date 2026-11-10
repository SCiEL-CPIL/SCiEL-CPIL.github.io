import numpy as np
from scipy.optimize import minimize, minimize_scalar
import time

def _compute_area(x):
    """
    Computes the area of the union of the 64 triangles.
    x: array of shape (64,)
    """
    n = 64
    # Quadrature points
    m = np.arange(20000)
    y = (m + 0.5) / 20000.0
    y = y.reshape(1, -1) # Shape (1, 20000)
    
    # a_i for each triangle
    a = (np.arange(n) + 1) / n # Shape (n,)
    a = a.reshape(-1, 1) # Shape (n, 1)
    
    # L_i(y) = x_i + a_i * y
    # x shape (n,), a shape (n,1), y shape (1, 20000)
    L = x.reshape(-1, 1) + a * y # Shape (n, 20000)
    
    # w(y) = (1 - y) / n
    w = (1.0 - y) / n # Shape (1, 20000)
    
    # Sort L along axis 0
    L_sorted = np.sort(L, axis=0) # Shape (n, 20000)
    
    # Differences between consecutive sorted L's
    D = L_sorted[1:, :] - L_sorted[:-1, :] # Shape (n-1, 20000)
    
    # Union length at each y: w + sum(min(D_j, w))
    # min(D, w) where D is (n-1, 20000) and w is (1, 20000)
    # Broadcasting works: w is broadcast to (n-1, 20000)
    min_diffs = np.minimum(D, w)
    
    # Sum over triangles (axis 0)
    union_lengths = w[0, :] + np.sum(min_diffs, axis=0) # Shape (20000,)
    
    # Area is mean of union lengths
    area = np.mean(union_lengths)
    return area

def kakeya_needle_2d_positions64() -> np.ndarray:
    n = 64
    k = int(np.log2(n))
    
    # 1. Initialize with Keich Construction
    j = np.arange(1, k + 1)
    positions = []
    for i in range(n):
        eps = np.array(list(str(bin(i))[2:]), dtype=int)
        eps = np.pad(eps, (k - len(eps), 0), "constant")
        positions.append(np.sum((1 - j) * eps * (2.0 ** -j) / k))
    x0 = np.array(positions, dtype=np.float64)
    
    start_time = time.time()
    deadline = start_time + 500  # 500s limit to be safe
    
    # 2. L-BFGS-B Optimization
    # We minimize the area
    res = minimize(_compute_area, x0, method='L-BFGS-B', options={'maxiter': 1000, 'ftol': 1e-12, 'gtol': 1e-8})
    x_best = res.x
    area_best = res.fun
    
    if time.time() > deadline:
        return x_best

    # 3. Coordinate Descent Polishing
    # Since the function is convex, coordinate descent converges to the global min.
    # We use minimize_scalar on each coordinate.
    # We define a window around the current value.
    # The optimal x_i is likely within [-1, 1] or so.
    # Let's use a window of +/- 0.5 initially, shrinking if needed.
    
    improved = True
    while improved and (time.time() < deadline):
        improved = False
        for i in range(n):
            # Define 1D function for coordinate i
            def f_1d(xi):
                x_temp = x_best.copy()
                x_temp[i] = xi
                return _compute_area(x_temp)
            
            # Search in a window around current x_best[i]
            # The kinks are at distance (a_j - a_i) * y.
            # Max distance is roughly 1.
            # Let's search in [x_best[i] - 1.0, x_best[i] + 1.0]
            # But to be fast, we can start with a smaller window if we are close to optimum.
            # Let's try +/- 0.5
            res_i = minimize_scalar(f_1d, bounds=(x_best[i] - 0.5, x_best[i] + 0.5), method='bounded', options={'xatol': 1e-8})
            
            if res_i.success:
                if abs(res_i.x - x_best[i]) > 1e-10:
                    x_best[i] = res_i.x
                    improved = True
                # Update best area if improved
                if res_i.fun < area_best:
                    area_best = res_i.fun
            if time.time() > deadline:
                return x_best

    return x_best
