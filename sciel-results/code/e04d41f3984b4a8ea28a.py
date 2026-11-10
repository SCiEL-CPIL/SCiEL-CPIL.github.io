# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize, differential_evolution
import time


def _compute_area_fast(x, n=64, M=20000):
    """Compute only the area (no gradient) for evaluation."""
    a = (np.arange(n) + 1) / n
    w0 = 1.0 / n
    y = (np.arange(M) + 0.5) / M

    L = x[:, None] + a[:, None] * y[None, :]
    w = (1.0 - y[None, :]) * w0

    idx = np.argsort(L, axis=0)
    L_sorted = np.take_along_axis(L, idx, axis=0)
    gaps = np.diff(L_sorted, axis=0)

    union_lengths = w + np.minimum(gaps, w[None, :]).sum(axis=0)
    area = np.mean(union_lengths)
    return area

def _compute_area_and_grad(x, n=64, M=20000):
    """Compute area and gradient of union area using vectorized operations."""
    a = (np.arange(n) + 1) / n
    w0 = 1.0 / n
    y = (np.arange(M) + 0.5) / M

    L = x[:, None] + a[:, None] * y[None, :]
    w = (1.0 - y[None, :]) * w0

    idx = np.argsort(L, axis=0)
    L_sorted = np.take_along_axis(L, idx, axis=0)
    gaps = np.diff(L_sorted, axis=0)

    union_lengths = w + np.minimum(gaps, w[None, :]).sum(axis=0)
    area = np.mean(union_lengths)

    active = gaps < w[None, :]
    left_idx = idx[:-1, :].ravel()
    right_idx = idx[1:, :].ravel()
    active_flat = active.ravel()

    left_counts = np.bincount(left_idx, weights=active_flat, minlength=n)
    right_counts = np.bincount(right_idx, weights=active_flat, minlength=n)

    grad_x = (left_counts - right_counts) / M

    return area, grad_x


def _keich_initial(n=64):
    """Generate the Keich construction initial point."""
    k = int(np.log2(n))
    j = np.arange(1, k + 1)
    positions = []
    for i in range(n):
        eps = np.array(list(str(bin(i))[2:]), dtype=int)
        eps = np.pad(eps, (k - len(eps), 0), "constant")
        positions.append(np.sum((1 - j) * eps * (2.0 ** -j) / k))
    return np.array(positions, dtype=np.float64)

def kakeya_needle_2d_positions64() -> np.ndarray:
    """
    Two-dimensional Kakeya-needle area-minimization problem for n=64.
    
    Uses Differential Evolution for global search and L-BFGS-B for local polish.
    """
    start_time = time.time()
    deadline = start_time + 620  # 620 second wall-clock limit (10s safety margin)

    n = 64
    M = 20000

    x_keich = _keich_initial(n)
    best_x = x_keich.copy()
    best_area = _compute_area_fast(x_keich, n, M)

    # Bounds for Differential Evolution
    # x_i shifts the triangle. The triangle has base 1/64.
    # The union is roughly in [0, 1] if x_i are small.
    # Let's allow x_i to range from -1 to 1 to be safe.
    bounds = [(-1.0, 1.0)] * n

    # Phase 1: Differential Evolution (Global Search)
    # DE is expensive, so we limit maxiter and popsize.
    # popsize = 15 * n = 960 individuals. Each eval is ~0.1-0.2s? 
    # 20000 quadrature points, 64 triangles. Sorting 64x20000 is fast.
    # Let's estimate cost of one area calc: 
    # L is 64x20000. Sort is 20000 * 64 * log(64) ~ 20000 * 64 * 6 ~ 7.6M ops. 
    # Vectorized in numpy, this is very fast, maybe 5-10ms.
    # So 1000 evals is 10s. 
    # DE with popsize 15*64=960 and maxiter 50 is 48000 evals. That's ~500s. Too long.
    # Reduce popsize to 5*64 = 320. Maxiter 20. 320*20 = 6400 evals. ~60s.
    
    elapsed = time.time() - start_time
    if elapsed < deadline - 100: # Leave 100s for polish
        try:
            result_de = differential_evolution(
                _compute_area_fast,
                bounds=bounds,
                args=(n, M),
                maxiter=30,
                popsize=5, # 5 * 64 = 320 individuals
                tol=1e-5,
                seed=42,
                polish=False, # We will do our own polish
                update_interval=1
            )
            if result_de.success or result_de.fun < best_area:
                if result_de.fun < best_area:
                    best_area = result_de.fun
                    best_x = result_de.x.copy()
        except Exception:
            pass

    # Phase 2: Local Polish with L-BFGS-B from best DE result
    elapsed = time.time() - start_time
    if elapsed < deadline - 20:
        try:
            def fun_and_jac(x_eval):
                area, grad = _compute_area_and_grad(x_eval, n, M)
                return area, grad

            result_lbfgs = minimize(
                fun=fun_and_jac,
                x0=best_x,
                jac=True,
                method='L-BFGS-B',
                options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-8}
            )
            
            area_full = _compute_area_fast(result_lbfgs.x, n, M)
            if area_full < best_area:
                best_area = area_full
                best_x = result_lbfgs.x.copy()
        except Exception:
            pass

    # Phase 3: Try a few random restarts with L-BFGS-B if time permits
    rng = np.random.default_rng(12345)
    for trial in range(5):
        elapsed = time.time() - start_time
        if elapsed > deadline - 20:
            break
        
        # Random initialization within bounds
        x0 = rng.uniform(-0.5, 0.5, size=n)
        
        try:
            def fun_and_jac(x_eval):
                area, grad = _compute_area_and_grad(x_eval, n, M)
                return area, grad

            result_lbfgs = minimize(
                fun=fun_and_jac,
                x0=x0,
                jac=True,
                method='L-BFGS-B',
                options={'maxiter': 200, 'ftol': 1e-10, 'gtol': 1e-6}
            )
            
            area_full = _compute_area_fast(result_lbfgs.x, n, M)
            if area_full < best_area:
                best_area = area_full
                best_x = result_lbfgs.x.copy()
        except Exception:
            pass

    # Final check
    if not np.all(np.isfinite(best_x)):
        best_x = x_keich

    return best_x.astype(np.float64)


# EVOLVE-BLOCK-END
