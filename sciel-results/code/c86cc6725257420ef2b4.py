# EVOLVE-BLOCK-START
import numpy as np
import time
import scipy.optimize

def _compute_area_fast(x):
    """Fast vectorized area computation for 64 triangles."""
    n = 64
    a = (np.arange(n) + 1) / n
    w = 1.0 / n
    n_heights = 20000
    y = (np.arange(n_heights) + 0.5) / n_heights
    
    # L shape: (n_heights, n)
    L = x[None, :] + a[None, :] * y[:, None]
    L_sorted = np.sort(L, axis=1)
    diffs = np.diff(L_sorted, axis=1)
    union_lengths = w + np.sum(np.minimum(diffs, w), axis=1)
    return np.mean(union_lengths)

def _keich_positions():
    """Classical Keich construction for n a power of 2."""
    n = 64
    k = int(np.log2(n))
    j = np.arange(1, k + 1)
    positions = []
    for i in range(n):
        eps = np.array(list(str(bin(i))[2:]), dtype=int)
        eps = np.pad(eps, (k - len(eps), 0), "constant")
        positions.append(np.sum((1 - j) * eps * (2.0 ** -j) / k))
    return np.array(positions, dtype=np.float64)

def kakeya_needle_2d_positions64() -> np.ndarray:
    n = 64
    start_time = time.time()
    deadline = start_time + 600  # 600s wall-clock limit, leave 20s margin
    
    best_x = _keich_positions()
    best_area = _compute_area_fast(best_x)
    
    # Strategy: Differential Evolution (global) + L-BFGS-B (local polish)
    # DE is good at escaping local minima; L-BFGS-B refines to high precision.
    
    def objective(x):
        return _compute_area_fast(x)
    
    # Phase 1: Differential Evolution
    # Use a moderate population size and generation count
    pop_size = 30
    max_iter = 50
    tol = 1e-10
    
    def de_callback(xk, convergence):
        nonlocal best_x, best_area
        area = _compute_area_fast(xk)
        if area < best_area:
            best_area = area
            best_x = xk.copy()
        return time.time() > deadline - 10
    
    try:
        result = scipy.optimize.differential_evolution(
            objective, 
            bounds=[(-1, 2)] * n,
            popsize=pop_size,
            maxiter=max_iter,
            tol=tol,
            seed=42,
            callback=de_callback,
            polish=False,  # We'll do our own polish
            init='latinhypercube'
        )
        if result.fun < best_area:
            best_area = result.fun
            best_x = result.x
    except Exception:
        pass
    
    # Check time
    if time.time() > deadline - 30:
        return best_x
    
    # Phase 2: L-BFGS-B polish from best DE result
    try:
        result = scipy.optimize.minimize(
            objective, 
            best_x, 
            method='L-BFGS-B',
            options={'maxiter': 500, 'ftol': 1e-14, 'gtol': 1e-10}
        )
        if result.fun < best_area:
            best_area = result.fun
            best_x = result.x
    except Exception:
        pass
    
    # Phase 3: Coarse-to-fine local refinement
    # Try small perturbations around the best point
    rng = np.random.default_rng(123)
    step_sizes = [1e-2, 1e-3, 1e-4, 1e-5]
    
    for step_size in step_sizes:
        if time.time() > deadline - 10:
            break
        
        for _ in range(500):
            if time.time() > deadline - 10:
                break
            
            i = rng.integers(0, n)
            delta = rng.normal(0, step_size)
            
            x_new = best_x.copy()
            x_new[i] += delta
            
            new_area = _compute_area_fast(x_new)
            
            if new_area < best_area:
                best_x = x_new
                best_area = new_area
    
    # Phase 4: Final high-precision polish
    if time.time() < deadline - 5:
        try:
            result = scipy.optimize.minimize(
                objective,
                best_x,
                method='L-BFGS-B',
                options={'maxiter': 1000, 'ftol': 1e-16, 'gtol': 1e-12}
            )
            if result.fun < best_area:
                best_area = result.fun
                best_x = result.x
        except Exception:
            pass
    
    return best_x

# EVOLVE-BLOCK-END
