# EVOLVE-BLOCK-START
import numpy as np
import time
import sys

def _compute_area(x, n=64, M=20000):
    """
    Compute the area of the union of n triangles using quadrature at M heights.
    x: array of shape (n,) with base positions.
    Returns: area (float).
    """
    a = (np.arange(n) + 1.0) / n  # a_i = (i+1)/n
    # y_m = (m + 0.5) / M for m = 0..M-1
    y = (np.arange(M) + 0.5) / M  # shape (M,)
    
    # L_i = x_i + a_i * y  -> shape (n, M)
    # w = (1 - y) / n     -> shape (M,)
    # We need to sort L_i along axis=0 for each y, then compute union length.
    
    # L[i, m] = x[i] + a[i] * y[m]
    L = x[:, None] + a[:, None] * y[None, :]  # shape (n, M)
    w = (1.0 - y) / n  # shape (M,)
    
    # Sort L along axis 0 (for each column m)
    L_sorted = np.sort(L, axis=0)  # shape (n, M)
    
    # Union length at each height m:
    # w + sum_{j=0}^{n-2} min(L_sorted[j+1, m] - L_sorted[j, m], w[m])
    diffs = L_sorted[1:, :] - L_sorted[:-1, :]  # shape (n-1, M)
    union_lengths = w[None, :] + np.sum(np.minimum(diffs, w[None, :]), axis=0)  # shape (M,)
    
    area = np.mean(union_lengths)
    return area


def _compute_area_fast(x, n=64, M=20000):
    """
    Optimized area computation.
    """
    a = (np.arange(n) + 1.0) / n
    y = (np.arange(M) + 0.5) / M
    
    L = x[:, None] + a[:, None] * y[None, :]
    w = (1.0 - y) / n
    
    L_sorted = np.sort(L, axis=0)
    
    diffs = L_sorted[1:, :] - L_sorted[:-1, :]
    union_lengths = w[None, :] + np.sum(np.minimum(diffs, w[None, :]), axis=0)
    
    area = np.mean(union_lengths)
    return area


def _keich_initial(n=64):
    """
    Keich construction for initial positions.
    """
    k = int(np.log2(n))
    j = np.arange(1, k + 1)
    positions = []
    for i in range(n):
        eps = np.array(list(str(bin(i))[2:]), dtype=int)
        eps = np.pad(eps, (k - len(eps), 0), "constant")
        positions.append(np.sum((1 - j) * eps * (2.0 ** -j) / k))
    return np.array(positions, dtype=np.float64)


def _coordinate_descent(x, n=64, M=20000, max_iters=500, tol=1e-12, deadline=None):
    """
    Coordinate descent with line search on each coordinate.
    """
    current_area = _compute_area_fast(x, n, M)
    best_x = x.copy()
    best_area = current_area
    
    for iteration in range(max_iters):
        if deadline is not None and time.time() > deadline:
            break
        
        improved = False
        for i in range(n):
            if deadline is not None and time.time() > deadline:
                break
            
            # Line search for x[i]
            # Try a range of values around current x[i]
            x_i = x[i]
            
            # Use golden section or ternary search in a reasonable range
            # The optimal x[i] is likely in [-1, 1] or so
            lo = x_i - 0.5
            hi = x_i + 0.5
            
            # Evaluate at several points
            # Use a grid search first to narrow down
            n_grid = 20
            grid = np.linspace(lo, hi, n_grid)
            areas = np.zeros(n_grid)
            for k, trial_xi in enumerate(grid):
                x_trial = x.copy()
                x_trial[i] = trial_xi
                areas[k] = _compute_area_fast(x_trial, n, M)
            
            best_k = np.argmin(areas)
            if best_k > 0 and best_k < n_grid - 1:
                # Narrow down
                lo = grid[max(0, best_k - 1)]
                hi = grid[min(n_grid - 1, best_k + 1)]
                grid2 = np.linspace(lo, hi, n_grid)
                areas2 = np.zeros(n_grid)
                for k, trial_xi in enumerate(grid2):
                    x_trial = x.copy()
                    x_trial[i] = trial_xi
                    areas2[k] = _compute_area_fast(x_trial, n, M)
                
                best_k2 = np.argmin(areas2)
                new_xi = grid2[best_k2]
                
                if areas2[best_k2] < current_area - tol:
                    x[i] = new_xi
                    current_area = areas2[best_k2]
                    improved = True
            else:
                # Best is at boundary, try expanding
                if best_k == 0:
                    hi = x_i + 0.5
                    lo = x_i - 1.0
                    grid = np.linspace(lo, hi, n_grid)
                    areas = np.zeros(n_grid)
                    for k, trial_xi in enumerate(grid):
                        x_trial = x.copy()
                        x_trial[i] = trial_xi
                        areas[k] = _compute_area_fast(x_trial, n, M)
                    best_k = np.argmin(areas)
                    new_xi = grid[best_k]
                    if areas[best_k] < current_area - tol:
                        x[i] = new_xi
                        current_area = areas[best_k]
                        improved = True
                elif best_k == n_grid - 1:
                    lo = x_i - 0.5
                    hi = x_i + 1.0
                    grid = np.linspace(lo, hi, n_grid)
                    areas = np.zeros(n_grid)
                    for k, trial_xi in enumerate(grid):
                        x_trial = x.copy()
                        x_trial[i] = trial_xi
                        areas[k] = _compute_area_fast(x_trial, n, M)
                    best_k = np.argmin(areas)
                    new_xi = grid[best_k]
                    if areas[best_k] < current_area - tol:
                        x[i] = new_xi
                        current_area = areas[best_k]
                        improved = True
        
        if not improved:
            break
        
        # Update best
        if current_area < best_area:
            best_area = current_area
            best_x = x.copy()
    
    return best_x, best_area


def _simulated_annealing(x, n=64, M=20000, deadline=None, max_steps=50000):
    """
    Simulated annealing to escape local minima.
    """
    current_area = _compute_area_fast(x, n, M)
    best_x = x.copy()
    best_area = current_area
    
    T = 0.01  # initial temperature
    cooling = 0.999
    
    rng = np.random.default_rng(42)
    
    step = 0
    while step < max_steps:
        if deadline is not None and time.time() > deadline:
            break
        
        # Perturb one or a few coordinates
        n_perturb = rng.integers(1, 4)
        indices = rng.choice(n, size=n_perturb, replace=False)
        x_trial = x.copy()
        x_trial[indices] += rng.normal(0, T, size=n_perturb)
        
        trial_area = _compute_area_fast(x_trial, n, M)
        
        delta = trial_area - current_area
        if delta < 0 or rng.random() < np.exp(-delta / max(T, 1e-10)):
            x = x_trial
            current_area = trial_area
            if current_area < best_area:
                best_area = current_area
                best_x = x.copy()
        
        T *= cooling
        step += 1
    
    return best_x, best_area


def kakeya_needle_2d_positions64() -> np.ndarray:
    """
    Two-dimensional Kakeya-needle area-minimization problem.
    """
    n = 64
    M = 20000
    
    start_time = time.time()
    deadline = start_time + 600.0  # 600s wall-clock limit
    
    # Start with Keich construction
    x = _keich_initial(n)
    current_area = _compute_area_fast(x, n, M)
    best_x = x.copy()
    best_area = current_area
    
    # Coordinate descent
    x_cd, area_cd = _coordinate_descent(x, n, M, max_iters=300, tol=1e-12, deadline=deadline - 10)
    if area_cd < best_area:
        best_area = area_cd
        best_x = x_cd.copy()
    
    # Simulated annealing from the best point
    if time.time() < deadline - 10:
        x_sa, area_sa = _simulated_annealing(best_x, n, M, deadline=deadline - 5, max_steps=20000)
        if area_sa < best_area:
            best_area = area_sa
            best_x = x_sa.copy()
    
    # Final coordinate descent refinement
    if time.time() < deadline - 5:
        x_final, area_final = _coordinate_descent(best_x, n, M, max_iters=200, tol=1e-12, deadline=deadline - 1)
        if area_final < best_area:
            best_area = area_final
            best_x = x_final.copy()
    
    # Ensure validity
    if not np.all(np.isfinite(best_x)) or best_area <= 0:
        return _keich_initial(n)
    
    return best_x


# EVOLVE-BLOCK-END
