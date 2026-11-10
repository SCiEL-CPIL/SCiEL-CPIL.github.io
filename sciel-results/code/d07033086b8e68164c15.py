# EVOLVE-BLOCK-START
import numpy as np
import time

def _compute_area(x, n=64, n_y=20000):
    """Compute the union area via quadrature."""
    y = (np.arange(n_y) + 0.5) / n_y
    w = (1.0 - y) / n
    a = (np.arange(n) + 1.0) / n
    # L_i = x_i + a_i * y_m, shape (n_y, n)
    L = x[None, :] + a[None, :] * y[:, None]
    # Sort L along axis 1
    L_sorted = np.sort(L, axis=1)
    # Union length = w + sum over consecutive pairs of min(L_{j+1} - L_j, w)
    gaps = np.diff(L_sorted, axis=1)  # shape (n_y, n-1)
    # union_len = w + sum(min(gap, w))
    # min(gap, w) = (gap - |gap - w|) / 2
    min_gaps = np.minimum(gaps, w[:, None])
    union_len = w + min_gaps.sum(axis=1)
    return union_len.mean()

def _compute_area_fast(x, n=64, n_y=20000):
    """Faster area computation using vectorized operations."""
    y = (np.arange(n_y) + 0.5) / n_y
    w = (1.0 - y) / n
    a = (np.arange(n) + 1.0) / n
    L = x[None, :] + a[None, :] * y[:, None]
    L_sorted = np.sort(L, axis=1)
    gaps = np.diff(L_sorted, axis=1)
    min_gaps = np.minimum(gaps, w[:, None])
    union_len = w + min_gaps.sum(axis=1)
    return union_len.mean()

def _keich_construction(n=64):
    """Classical Keich construction."""
    k = int(np.log2(n))
    j = np.arange(1, k + 1)
    positions = []
    for i in range(n):
        # Binary representation of i, padded to k bits
        bin_str = bin(i)[2:]
        eps = np.zeros(k, dtype=int)
        for pos, bit in enumerate(bin_str):
            eps[k - 1 - pos] = int(bit)
        positions.append(np.sum((1 - j) * eps * (2.0 ** -j) / k))
    return np.array(positions, dtype=np.float64)

def _gradient_descent(x0, n=64, n_y=20000, max_iter=500, lr=0.01, deadline=None):
    """Simple gradient descent to minimize area."""
    x = x0.copy()
    best_x = x.copy()
    best_area = _compute_area_fast(x, n, n_y)
    
    for it in range(max_iter):
        if deadline is not None and time.time() > deadline:
            break
        
        # Compute gradient via finite differences (central)
        grad = np.zeros(n)
        eps = 1e-6
        for i in range(n):
            xp = x.copy()
            xp[i] += eps
            xm = x.copy()
            xm[i] -= eps
            grad[i] = (_compute_area_fast(xp, n, n_y) - _compute_area_fast(xm, n, n_y)) / (2 * eps)
        
        # Update
        x = x - lr * grad
        
        area = _compute_area_fast(x, n, n_y)
        if area < best_area:
            best_area = area
            best_x = x.copy()
        
        # Adaptive learning rate
        if it % 50 == 0 and it > 0:
            lr *= 0.9
    
    return best_x, best_area

def _local_search(x0, n=64, n_y=20000, max_iter=200, step=0.01, deadline=None):
    """Coordinate descent / local search."""
    x = x0.copy()
    best_area = _compute_area_fast(x, n, n_y)
    
    for it in range(max_iter):
        if deadline is not None and time.time() > deadline:
            break
        
        improved = False
        for i in range(n):
            # Try moving x[i] by +/- step
            for delta in [-step, step]:
                xp = x.copy()
                xp[i] += delta
                area = _compute_area_fast(xp, n, n_y)
                if area < best_area:
                    x = xp
                    best_area = area
                    improved = True
        
        if not improved:
            # Reduce step size
            step *= 0.5
            if step < 1e-8:
                break
    
    return x, best_area

def kakeya_needle_2d_positions64() -> np.ndarray:
    """
    Two-dimensional Kakeya-needle area-minimization problem, n = 64.
    Returns base positions x = (x_0, ..., x_63) minimizing the union area.
    """
    start_time = time.time()
    deadline = start_time + 580  # Safety margin before 630s limit
    
    n = 64
    n_y = 20000  # Full resolution for final evaluation
    
    # Start with Keich construction
    x = _keich_construction(n)
    area = _compute_area_fast(x, n, n_y)
    
    # Try gradient-based refinement with lower resolution first
    n_y_coarse = 2000
    x_coarse = x.copy()
    
    # Coarse gradient descent
    for it in range(100):
        if time.time() > deadline:
            break
        grad = np.zeros(n)
        eps = 1e-5
        for i in range(n):
            xp = x_coarse.copy()
            xp[i] += eps
            xm = x_coarse.copy()
            xm[i] -= eps
            grad[i] = (_compute_area_fast(xp, n, n_y_coarse) - _compute_area_fast(xm, n, n_y_coarse)) / (2 * eps)
        x_coarse = x_coarse - 0.005 * grad
    
    # Check if coarse result is better
    area_coarse = _compute_area_fast(x_coarse, n, n_y_coarse)
    if area_coarse < area * (n_y / n_y_coarse) * 0.999:  # Rough comparison
        x = x_coarse
        area = _compute_area_fast(x, n, n_y)
    
    # Local search with full resolution
    x, area = _local_search(x, n, n_y, max_iter=50, step=0.005, deadline=deadline)
    
    # Final polish with smaller steps
    x, area = _local_search(x, n, n_y, max_iter=30, step=0.001, deadline=deadline)
    
    return x.astype(np.float64)

# EVOLVE-BLOCK-END
