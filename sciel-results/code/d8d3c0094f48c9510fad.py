# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


def _union_area(x, M=20000):
    """
    Compute the union area of 64 triangles.
    """
    n = 64
    a = (np.arange(n) + 1) / n
    y = (np.arange(M) + 0.5) / M
    
    L = x[np.newaxis, :] + a[np.newaxis, :] * y[:, np.newaxis]
    w = (1.0 - y) / n
    
    order = np.argsort(L, axis=1)
    L_sorted = np.take_along_axis(L, order, axis=1)
    
    diffs = np.diff(L_sorted, axis=1)
    
    union_len = w + np.sum(np.minimum(diffs, w[:, np.newaxis]), axis=1)
    area = np.mean(union_len)
    
    return area


def _union_area_grad(x, M=20000):
    """
    Compute the union area and its gradient w.r.t. x.
    """
    n = 64
    a = (np.arange(n) + 1) / n
    y = (np.arange(M) + 0.5) / M
    
    L = x[np.newaxis, :] + a[np.newaxis, :] * y[:, np.newaxis]
    w = (1.0 - y) / n
    
    order = np.argsort(L, axis=1)
    L_sorted = np.take_along_axis(L, order, axis=1)
    
    diffs = np.diff(L_sorted, axis=1)
    mask = diffs < w[:, np.newaxis]
    
    union_len = w + np.sum(np.minimum(diffs, w[:, np.newaxis]), axis=1)
    area = np.mean(union_len)
    
    grad = np.zeros(n, dtype=np.float64)
    rows, cols = np.where(mask)
    
    if len(rows) > 0:
        idx_plus = order[rows, cols]
        idx_minus = order[rows, cols + 1]
        np.add.at(grad, idx_plus, 1.0)
        np.add.at(grad, idx_minus, -1.0)
    
    grad /= M
    return area, grad


def _keich_init(n=64):
    """
    Keich construction initial guess.
    """
    k = int(np.log2(n))
    j_arr = np.arange(1, k + 1)
    positions = np.zeros(n, dtype=np.float64)
    for i in range(n):
        eps = np.array(list(str(bin(i))[2:]), dtype=int)
        eps = np.pad(eps, (k - len(eps), 0), "constant")
        positions[i] = np.sum((1 - j_arr) * eps * (2.0 ** -j_arr) / k)
    return positions


def _random_init(n=64, seed=0):
    """
    Random initial guess.
    """
    rng = np.random.default_rng(seed)
    return rng.uniform(-1, 1, size=n)


def _soft_min(x, y, beta=100.0):
    """
    Smooth approximation of min(x, y) using log-sum-exp.
    """
    # min(x,y) = -max(-x,-y)
    # max(a,b) = (1/beta)*log(exp(beta*a) + exp(beta*b))
    a = -x
    b = -y
    m = np.maximum(a, b)
    return -m - np.log(np.exp(beta * (a - m)) + np.exp(beta * (b - m))) / beta


def _union_area_soft(x, M=20000, beta=100.0):
    """
    Compute the union area using a soft-min approximation for the min(diffs, w) term.
    This provides a smooth objective for gradient-based optimization.
    """
    n = 64
    a = (np.arange(n) + 1) / n
    y = (np.arange(M) + 0.5) / M
    
    L = x[np.newaxis, :] + a[np.newaxis, :] * y[:, np.newaxis]
    w = (1.0 - y) / n
    
    order = np.argsort(L, axis=1)
    L_sorted = np.take_along_axis(L, order, axis=1)
    
    diffs = np.diff(L_sorted, axis=1)
    
    # Use soft-min for smoothness
    soft_min_vals = _soft_min(diffs, w[:, np.newaxis], beta=beta)
    
    union_len = w + np.sum(soft_min_vals, axis=1)
    area = np.mean(union_len)
    
    return area


def _union_area_soft_grad(x, M=20000, beta=100.0):
    """
    Compute the union area using soft-min and its gradient.
    """
    n = 64
    a = (np.arange(n) + 1) / n
    y = (np.arange(M) + 0.5) / M
    
    L = x[np.newaxis, :] + a[np.newaxis, :] * y[:, np.newaxis]
    w = (1.0 - y) / n
    
    order = np.argsort(L, axis=1)
    L_sorted = np.take_along_axis(L, order, axis=1)
    
    diffs = np.diff(L_sorted, axis=1)
    
    # Soft-min: s = min(diffs, w) approximated smoothly
    # s = -max(-diffs, -w)
    # d(s)/d(diffs) = sigmoid(beta * (w - diffs))  (approximately, when w > diffs)
    # More precisely:
    # max(a,b) where a=-diffs, b=-w
    # d(max(a,b))/da = sigmoid(beta*(a-b)) = sigmoid(beta*(-diffs+w)) = sigmoid(beta*(w-diffs))
    # d(s)/d(diffs) = -d(max(-diffs,-w))/d(-diffs) = -sigmoid(beta*(w-diffs))
    # Wait, let me re-derive:
    # s = -max(-diffs, -w)
    # d(s)/d(diffs) = -d(max(-diffs,-w))/d(diffs)
    # Let u = -diffs, v = -w
    # max(u,v) = (1/beta)*log(exp(beta*u)+exp(beta*v))
    # d(max(u,v))/du = exp(beta*u)/(exp(beta*u)+exp(beta*v)) = 1/(1+exp(beta*(v-u))) = sigmoid(beta*(u-v))
    # d(max(u,v))/d(diffs) = d(max(u,v))/du * du/d(diffs) = sigmoid(beta*(u-v)) * (-1)
    # d(s)/d(diffs) = -d(max(u,v))/d(diffs) = sigmoid(beta*(u-v)) = sigmoid(beta*(-diffs+w)) = sigmoid(beta*(w-diffs))
    
    # So d(soft_min)/d(diffs) = sigmoid(beta * (w - diffs))
    
    sigmoid_vals = 1.0 / (1.0 + np.exp(-beta * (w[:, np.newaxis] - diffs)))
    
    soft_min_vals = _soft_min(diffs, w[:, np.newaxis], beta=beta)
    union_len = w + np.sum(soft_min_vals, axis=1)
    area = np.mean(union_len)
    
    # Gradient computation
    # d(union_len)/d(x_j) for each triangle j
    # For each y_m and each k in 0..n-2:
    #   soft_min_term = soft_min(diffs[m,k], w[m])
    #   d(soft_min_term)/d(x_{order[m,k]}) = -sigmoid_vals[m,k] * (a[order[m,k+1]] - a[order[m,k]]) * ... 
    #   Actually, diffs[m,k] = L_sorted[m,k+1] - L_sorted[m,k] = x[order[m,k+1]] + a[order[m,k+1]]*y_m - x[order[m,k]] - a[order[m,k]]*y_m
    #   d(diffs[m,k])/d(x_i) = 1 if i == order[m,k+1], -1 if i == order[m,k], 0 otherwise
    #   d(soft_min)/d(x_i) = sigmoid_vals[m,k] * d(diffs[m,k])/d(x_i)
    
    grad = np.zeros(n, dtype=np.float64)
    
    # For each (m, k):
    #   grad[order[m,k+1]] += sigmoid_vals[m,k]
    #   grad[order[m,k]] -= sigmoid_vals[m,k]
    
    # Vectorized:
    m_idx, k_idx = np.arange(M)[:, np.newaxis], np.arange(n-1)[np.newaxis, :]
    m_idx = np.broadcast_to(m_idx, (M, n-1)).flatten()
    k_idx = np.broadcast_to(k_idx, (M, n-1)).flatten()
    
    weights = sigmoid_vals.flatten()
    
    idx_plus = order[m_idx, k_idx + 1]
    idx_minus = order[m_idx, k_idx]
    
    np.add.at(grad, idx_plus, weights)
    np.add.at(grad, idx_minus, -weights)
    
    grad /= M
    return area, grad


def kakeya_needle_2d_positions64() -> np.ndarray:
    """
    Two-dimensional Kakeya-needle area-minimization problem for n=64.
    
    Uses a multi-phase approach:
    1. Coarse optimization with soft-min smooth surrogate (beta=50) to find good basins
    2. Fine optimization with exact nonsmooth objective at full resolution
    3. Multiple diverse initializations to escape local minima
    """
    n = 64
    
    start_time = time.time()
    deadline = start_time + 620.0  # Safety margin before 630s limit
    
    def time_left():
        return deadline - time.time()
    
    best_x = _keich_init(n)
    best_area = _union_area(best_x, M=2000)
    
    # Generate diverse initial points
    inits = []
    
    # Keich construction
    inits.append(_keich_init(n))
    
    # Perturbed Keich with different noise levels and seeds
    for seed in range(12):
        rng = np.random.default_rng(seed)
        noise_level = 0.01 + 0.05 * (seed % 6)
        inits.append(_keich_init(n) + rng.normal(0, noise_level, size=n))
    
    # Random initializations
    for seed in range(6):
        inits.append(_random_init(n, seed=200 + seed))
    
    # Structured initializations
    base = _keich_init(n)
    for shift in [-0.15, -0.08, 0.08, 0.15]:
        inits.append(base + shift)
    for scale in [0.7, 0.9, 1.1, 1.3]:
        inits.append(base * scale)
    
    # Sort inits by initial area (quick check)
    init_areas = [_union_area(x, M=800) for x in inits]
    init_order = np.argsort(init_areas)
    inits = [inits[i] for i in init_order]
    
    # Phase 1: Coarse optimization with soft-min surrogate
    # Use moderate M and moderate beta for speed
    coarse_M = 2000
    coarse_beta = 50.0
    
    for i, x0 in enumerate(inits):
        if time_left() < 30.0:
            break
        
        try:
            def objective_soft(x):
                return _union_area_soft(x, M=coarse_M, beta=coarse_beta)
            
            def objective_soft_grad(x):
                return _union_area_soft_grad(x, M=coarse_M, beta=coarse_beta)
            
            res = minimize(
                objective_soft_grad,
                x0,
                jac=True,
                method='L-BFGS-B',
                options={'maxiter': 100, 'gtol': 1e-6, 'ftol': 1e-8, 'maxls': 30}
            )
            
            x_opt = res.x
            # Evaluate with exact objective
            area_opt = _union_area(x_opt, M=coarse_M)
            
            if area_opt < best_area:
                best_x = x_opt.copy()
                best_area = area_opt
        except Exception:
            continue
    
    # Phase 2: Fine optimization with exact objective at higher resolution
    fine_M = 5000
    
    for i in range(min(5, len(inits))):
        if time_left() < 20.0:
            break
        
        x0 = best_x + np.random.default_rng(i * 137 + 42).normal(0, 0.02, size=n)
        
        try:
            def objective_fine(x):
                return _union_area(x, M=fine_M)
            
            def objective_fine_grad(x):
                return _union_area_grad(x, M=fine_M)
            
            res = minimize(
                objective_fine_grad,
                x0,
                jac=True,
                method='L-BFGS-B',
                options={'maxiter': 150, 'gtol': 1e-7, 'ftol': 1e-9, 'maxls': 40}
            )
            
            x_opt = res.x
            area_opt = _union_area(x_opt, M=fine_M)
            
            if area_opt < best_area:
                best_x = x_opt.copy()
                best_area = area_opt
        except Exception:
            continue
    
    # Phase 3: Full-resolution polish with tight tolerances
    if time_left() > 15.0:
        try:
            def objective_full(x):
                return _union_area(x, M=20000)
            
            def objective_full_grad(x):
                return _union_area_grad(x, M=20000)
            
            res = minimize(
                objective_full_grad,
                best_x,
                jac=True,
                method='L-BFGS-B',
                options={'maxiter': 100, 'gtol': 1e-8, 'ftol': 1e-10, 'maxls': 50}
            )
            
            x_final = res.x
            area_final = _union_area(x_final, M=20000)
            
            if area_final < best_area:
                best_x = x_final.copy()
                best_area = area_final
        except Exception:
            pass
    
    # Phase 4: Second polish if time remains
    if time_left() > 10.0:
        try:
            def objective_full2(x):
                return _union_area(x, M=20000)
            
            def objective_full2_grad(x):
                return _union_area_grad(x, M=20000)
            
            res2 = minimize(
                objective_full2_grad,
                best_x,
                jac=True,
                method='L-BFGS-B',
                options={'maxiter': 80, 'gtol': 1e-9, 'ftol': 1e-12, 'maxls': 50}
            )
            
            x_final2 = res2.x
            area_final2 = _union_area(x_final2, M=20000)
            
            if area_final2 < best_area:
                best_x = x_final2.copy()
                best_area = area_final2
        except Exception:
            pass
    
    # Ensure result is valid
    if not np.all(np.isfinite(best_x)):
        best_x = _keich_init(n)
    
    return best_x


# EVOLVE-BLOCK-END
