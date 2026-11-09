# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time


def _compute_area_and_grad(x, n=64, M=20000):
    """Compute area and gradient of union area using vectorized operations."""
    a = (np.arange(n) + 1) / n  # a_i = (i+1)/n
    w0 = 1.0 / n
    y = (np.arange(M) + 0.5) / M  # M quadrature points
    
    # L_i(y) = x_i + a_i * y, shape (n, M)
    L = x[:, None] + a[:, None] * y[None, :]
    w = (1.0 - y[None, :]) * w0  # width at each y, shape (M,)
    
    # Sort L at each y
    idx = np.argsort(L, axis=0)  # (n, M)
    L_sorted = np.take_along_axis(L, idx, axis=0)  # (n, M)
    
    # Gaps between consecutive sorted L values
    gaps = np.diff(L_sorted, axis=0)  # (n-1, M)
    
    # Union length = w + sum(min(gap, w))
    union_lengths = w + np.minimum(gaps, w[None, :]).sum(axis=0)  # (M,)
    area = np.mean(union_lengths)
    
    # Gradient: d(area)/d(x_i) using bincount for speed
    active = gaps < w[None, :]  # (n-1, M), True where gap < w
    
    # Flatten for bincount
    left_idx = idx[:-1, :].ravel()   # (n-1)*M
    right_idx = idx[1:, :].ravel()   # (n-1)*M
    active_flat = active.ravel()     # (n-1)*M
    
    # Count contributions per original index
    left_counts = np.bincount(left_idx, weights=active_flat, minlength=n)
    right_counts = np.bincount(right_idx, weights=active_flat, minlength=n)
    
    grad_x = (left_counts - right_counts) / M  # (n,)
    
    return area, grad_x


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


def _coordinate_descent_polish(x, n=64, M=20000, max_rounds=50, deadline=None):
    """
    Coordinate descent polish: for each coordinate, try small perturbations
    to find local improvements. This handles the nonsmooth min() term well.
    """
    x = x.copy()
    best_area = _compute_area_fast(x, n, M)
    a = (np.arange(n) + 1) / n
    w0 = 1.0 / n
    y = (np.arange(M) + 0.5) / M
    
    # Precompute y-dependent terms
    L_base = a[:, None] * y[None, :]  # (n, M) without x contribution
    w = (1.0 - y[None, :]) * w0  # (M,)
    
    def _area_with_xi(xi_val, xi_idx):
        """Compute area with x[xi_idx] = xi_val, reusing precomputed data."""
        L = x[:, None] + L_base
        L[xi_idx, :] = xi_val + L_base[xi_idx, :]
        idx = np.argsort(L, axis=0)
        L_sorted = np.take_along_axis(L, idx, axis=0)
        gaps = np.diff(L_sorted, axis=0)
        union_lengths = w + np.minimum(gaps, w[None, :]).sum(axis=0)
        return np.mean(union_lengths)
    
    # Adaptive step sizes
    step_sizes = [0.05, 0.02, 0.01, 0.005, 0.002, 0.001, 0.0005, 0.0002]
    
    for round_idx in range(max_rounds):
        if deadline is not None and time.time() > deadline - 10:
            break
        
        improved = False
        for i in range(n):
            if deadline is not None and time.time() > deadline - 10:
                break
            
            current_xi = x[i]
            best_xi = current_xi
            best_local_area = best_area
            
            for step in step_sizes:
                for direction in [1, -1]:
                    new_xi = current_xi + direction * step
                    try:
                        new_area = _area_with_xi(new_xi, i)
                        if new_area < best_local_area - 1e-14:
                            best_local_area = new_area
                            best_xi = new_xi
                    except Exception:
                        pass
            
            if best_xi != current_xi:
                x[i] = best_xi
                best_area = best_local_area
                improved = True
        
        if not improved:
            break
    
    return x, best_area


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
    
    Uses multi-restart L-BFGS-B optimization with fast gradient computation
    plus coordinate descent polish to navigate the nonsmooth landscape.
    """
    start_time = time.time()
    deadline = start_time + 625  # 625 second wall-clock limit (5s safety margin)
    
    n = 64
    M = 20000  # Full quadrature resolution
    
    # Generate diverse initial points
    x_keich = _keich_initial(n)
    
    # Create multiple initializations
    inits = [x_keich.copy()]
    
    # Add perturbed versions of Keich with varied scales
    rng = np.random.default_rng(42)
    for trial in range(12):
        scale = 0.005 * (trial + 1)
        x_pert = x_keich + rng.normal(0, scale, size=n)
        inits.append(x_pert)
    
    # Add random initializations
    for trial in range(8):
        x_rand = rng.uniform(-0.5, 0.5, size=n)
        inits.append(x_rand)
    
    # Evaluate each initialization and run optimization
    best_x = x_keich.copy()
    best_area = _compute_area_fast(x_keich, n, M)
    
    for init_idx, x0 in enumerate(inits):
        elapsed = time.time() - start_time
        if elapsed > deadline - 45:  # Leave 45s margin for final polish
            break
        
        # Run L-BFGS-B with consistent fun/jac via single compute
        try:
            def fun_and_jac(x_eval):
                area, grad = _compute_area_and_grad(x_eval, n, M)
                return area, grad
            
            result = minimize(
                fun=fun_and_jac,
                x0=x0,
                jac=True,  # fun returns (f, g)
                method='L-BFGS-B',
                options={'maxiter': 400, 'ftol': 1e-12, 'gtol': 1e-8}
            )
            
            if result.fun < best_area:
                area_full = _compute_area_fast(result.x, n, M)
                if area_full < best_area:
                    best_area = area_full
                    best_x = result.x.copy()
        except Exception:
            continue
    
    # Final L-BFGS-B polish with tighter tolerance
    elapsed = time.time() - start_time
    if elapsed < deadline - 20:
        try:
            def fun_and_jac2(x_eval):
                area, grad = _compute_area_and_grad(x_eval, n, M)
                return area, grad
            
            result = minimize(
                fun=fun_and_jac2,
                x0=best_x,
                jac=True,
                method='L-BFGS-B',
                options={'maxiter': 600, 'ftol': 1e-14, 'gtol': 1e-10}
            )
            if result.fun < best_area:
                area_full = _compute_area_fast(result.x, n, M)
                if area_full < best_area:
                    best_area = area_full
                    best_x = result.x.copy()
        except Exception:
            pass
    
    # Coordinate descent polish to navigate nonsmooth landscape
    elapsed = time.time() - start_time
    if elapsed < deadline - 5:
        try:
            polished_x, polished_area = _coordinate_descent_polish(
                best_x, n, M, max_rounds=100, deadline=deadline
            )
            if polished_area < best_area:
                best_area = polished_area
                best_x = polished_x.copy()
        except Exception:
            pass
    
    # Ensure valid output
    if not np.all(np.isfinite(best_x)):
        best_x = x_keich
    
    return best_x.astype(np.float64)


# EVOLVE-BLOCK-END
