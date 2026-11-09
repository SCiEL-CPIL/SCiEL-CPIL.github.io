# EVOLVE-BLOCK-START
import numpy as np
import time


def _union_area(x, M=20000):
    """
    Compute the union area of 64 triangles (exact metric).
    
    x: shape (64,) - base positions
    M: number of quadrature points
    Returns: area (scalar)
    """
    n = 64
    a = (np.arange(n) + 1) / n
    y = (np.arange(M) + 0.5) / M
    
    # L_i(y) = x_i + a_i * y
    L = x[np.newaxis, :] + a[np.newaxis, :] * y[:, np.newaxis]
    w = (1.0 - y) / n
    
    # Sort L along axis=1
    order = np.argsort(L, axis=1)
    L_sorted = np.take_along_axis(L, order, axis=1)
    
    # Consecutive differences
    diffs = np.diff(L_sorted, axis=1)
    
    # union_len at each y
    union_len = w + np.sum(np.minimum(diffs, w[:, np.newaxis]), axis=1)
    area = np.mean(union_len)
    
    return area


def _union_area_smooth(x, M=20000, beta=50.0):
    """
    Compute a smooth surrogate of the union area using soft-min.
    
    Uses -log-sum-exp to approximate min(diffs, w) smoothly.
    This makes the objective differentiable everywhere, helping L-BFGS-B
    navigate through the non-smooth regions of the true objective.
    
    x: shape (64,) - base positions
    M: number of quadrature points
    beta: temperature parameter (higher = closer to true min)
    Returns: (area, grad) where grad has shape (64,)
    """
    n = 64
    a = (np.arange(n) + 1) / n
    y = (np.arange(M) + 0.5) / M
    
    # L_i(y) = x_i + a_i * y, shape (M, n)
    L = x[np.newaxis, :] + a[np.newaxis, :] * y[:, np.newaxis]
    w = (1.0 - y) / n  # shape (M,)
    
    # Sort L along axis=1 and track permutation
    order = np.argsort(L, axis=1)
    L_sorted = np.take_along_axis(L, order, axis=1)
    
    # Consecutive differences
    diffs = np.diff(L_sorted, axis=1)  # shape (M, n-1)
    
    # Smooth min: softmin(diffs, w) = -log(exp(beta*diffs) + exp(beta*w)) / beta
    # More stable: min(a,b) ≈ -(1/beta) * log(exp(beta*a) + exp(beta*b))
    # Using the log-sum-exp trick for numerical stability
    # softmin(d, w) = - (1/beta) * logsumexp(beta*d, beta*w)
    
    beta_d = beta * diffs
    beta_w = beta * w[:, np.newaxis]
    
    # logsumexp(beta_d, beta_w) = max + log(exp(beta_d - max) + exp(beta_w - max))
    max_val = np.maximum(beta_d, beta_w)
    log_sum_exp = max_val + np.log(np.exp(beta_d - max_val) + np.exp(beta_w - max_val))
    
    softmin = -log_sum_exp / beta  # shape (M, n-1)
    
    # union_len = w + sum(softmin)
    union_len = w + np.sum(softmin, axis=1)
    area = np.mean(union_len)
    
    # Gradient computation
    # d(softmin)/d(diffs) = -exp(beta*diffs) / (exp(beta*diffs) + exp(beta*w))
    # d(softmin)/d(w) = -exp(beta*w) / (exp(beta*diffs) + exp(beta*w))
    
    # For the gradient w.r.t. x:
    # d/dx_{order[k]} of softmin(diffs[k], w) = -exp(beta*diffs[k]) / (exp(beta*diffs[k]) + exp(beta*w)) * (1/beta) * (-1) [since diffs[k] = L[order[k+1]] - L[order[k]], d/dx_{order[k]} = -1]
    # Wait, let me be more careful.
    
    # diffs[k] = L_sorted[k+1] - L_sorted[k] = L[order[k+1]] - L[order[k]]
    # d(diffs[k])/d(x_i) = a_i if i = order[k+1], -a_i if i = order[k], 0 otherwise
    # (since L_i = x_i + a_i*y, dL_i/dx_i = 1)
    
    # d(softmin)/d(diffs[k]) = -exp(beta*diffs[k]) / (exp(beta*diffs[k]) + exp(beta*w))
    
    # So d(softmin)/d(x_i) = d(softmin)/d(diffs[k]) * d(diffs[k])/d(x_i)
    
    # Let's compute the weight for each diff:
    # weight_k = -exp(beta*diffs[k]) / (exp(beta*diffs[k]) + exp(beta*w))
    
    exp_bd = np.exp(beta_d - max_val)
    exp_bw = np.exp(beta_w - max_val)
    denom = exp_bd + exp_bw
    
    weight = -exp_bd / denom  # shape (M, n-1)
    
    # Now accumulate gradient
    # For each (m, k): 
    #   grad[order[m,k+1]] += weight[m,k]  (from d(diffs)/d(x_{order[k+1]}) = 1)
    #   grad[order[m,k]] -= weight[m,k]     (from d(diffs)/d(x_{order[k]}) = -1)
    
    # Using bincount for efficiency
    m_idx, k_idx = np.where(np.ones_like(weight, dtype=bool))  # all positions
    # Actually, we need all positions, so let's use a different approach
    
    # Reshape to use bincount
    weight_flat = weight.flatten()
    order_flat = order.flatten()
    
    # For position (m, k) in the (M, n-1) grid:
    # flat index = m * (n-1) + k
    # order[m, k] is at flat index m*n + k
    # order[m, k+1] is at flat index m*n + k + 1
    
    M_local, n_minus_1 = weight.shape
    total = M_local * n_minus_1
    
    # Create index arrays
    m_all = np.repeat(np.arange(M_local), n_minus_1)
    k_all = np.tile(np.arange(n_minus_1), M_local)
    
    # +1 contribution: order[m, k+1]
    idx_plus = order[m_all, k_all + 1]
    # -1 contribution: order[m, k]
    idx_minus = order[m_all, k_all]
    
    grad = np.zeros(n, dtype=np.float64)
    np.add.at(grad, idx_plus, weight_flat)
    np.add.at(grad, idx_minus, -weight_flat)
    
    grad /= M
    
    return area, grad


def _union_area_grad_exact(x, M=20000):
    """
    Compute the union area and its subgradient w.r.t. x (exact, non-smooth).
    
    x: shape (64,) - base positions
    M: number of quadrature points
    Returns: (area, grad) where grad has shape (64,)
    """
    n = 64
    a = (np.arange(n) + 1) / n
    y = (np.arange(M) + 0.5) / M
    
    # L_i(y) = x_i + a_i * y, shape (M, n)
    L = x[np.newaxis, :] + a[np.newaxis, :] * y[:, np.newaxis]
    w = (1.0 - y) / n
    
    # Sort L along axis=1 and track permutation
    order = np.argsort(L, axis=1)
    L_sorted = np.take_along_axis(L, order, axis=1)
    
    # Consecutive differences
    diffs = np.diff(L_sorted, axis=1)
    
    # mask: True where diffs < w (overlapping)
    mask = diffs < w[:, np.newaxis]
    
    # union_len at each y
    union_len = w + np.sum(np.minimum(diffs, w[:, np.newaxis]), axis=1)
    area = np.mean(union_len)
    
    # Gradient: where mask is True, the diff contributes
    # d/dx_{order[k]} of min(diffs[k], w) = -1 if overlapping
    # d/dx_{order[k+1]} of min(diffs[k], w) = +1 if overlapping
    
    m_idx, j_idx = np.where(mask)
    
    if len(m_idx) > 0:
        left_indices = order[m_idx, j_idx]
        right_indices = order[m_idx, j_idx + 1]
        
        grad_right = np.bincount(right_indices, minlength=n).astype(np.float64)
        grad_left = np.bincount(left_indices, minlength=n).astype(np.float64)
        grad = grad_right - grad_left
    else:
        grad = np.zeros(n, dtype=np.float64)
    
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


def kakeya_needle_2d_positions64() -> np.ndarray:
    """
    Two-dimensional Kakeya-needle area-minimization problem for n=64.
    
    Uses a smooth surrogate objective (soft-min) with L-BFGS-B for optimization,
    then verifies and polishes with the exact objective. Multiple diverse initial
    points help escape local minima.
    
    Returns:
        positions: np.ndarray of shape (64,) with the base x-coordinates x_i.
    """
    n = 64
    
    start_time = time.time()
    deadline = start_time + 630.0
    safety_margin = 15.0
    
    from scipy.optimize import minimize
    
    # Generate diverse initial points
    inits = []
    
    # Keich construction
    inits.append(_keich_init(n))
    
    # Perturbed Keich with different noise levels and seeds
    for seed in range(12):
        rng = np.random.default_rng(seed)
        noise_level = 0.01 + 0.04 * (seed % 5)
        inits.append(_keich_init(n) + rng.normal(0, noise_level, size=n))
    
    # Random initializations
    for seed in range(6):
        inits.append(_random_init(n, seed=100 + seed))
    
    # Sort inits by initial area (quick check with exact metric)
    init_areas = [_union_area(x, M=1000) for x in inits]
    init_order = np.argsort(init_areas)
    inits = [inits[i] for i in init_order]
    
    best_x = inits[0].copy()
    best_area = _union_area(best_x, M=2000)
    
    # Phase 1: Optimize using smooth surrogate with L-BFGS-B
    beta_start = 20.0  # Start with softer approximation
    
    for i, x0 in enumerate(inits):
        if time.time() > deadline - safety_margin:
            break
        
        try:
            # Use smooth objective for optimization
            def smooth_obj(x, beta=beta_start):
                area, grad = _union_area_smooth(x, M=2000, beta=beta)
                return area, grad
            
            res = minimize(
                smooth_obj,
                x0,
                jac=True,
                method='L-BFGS-B',
                options={'maxiter': 150, 'gtol': 1e-6, 'ftol': 1e-8}
            )
            
            x_opt = res.x
            # Evaluate with exact metric
            area_opt = _union_area(x_opt, M=2000)
            
            if area_opt < best_area:
                best_x = x_opt.copy()
                best_area = area_opt
                
        except Exception:
            continue
    
    # Phase 2: Polish with exact gradient-based optimization
    if time.time() < deadline - safety_margin:
        try:
            def exact_obj(x):
                area, grad = _union_area_grad_exact(x, M=5000)
                return area, grad
            
            res = minimize(
                exact_obj,
                best_x,
                jac=True,
                method='L-BFGS-B',
                options={'maxiter': 100, 'gtol': 1e-7, 'ftol': 1e-9}
            )
            
            x_opt = res.x
            area_opt = _union_area(x_opt, M=2000)
            
            if area_opt < best_area:
                best_x = x_opt.copy()
                best_area = area_opt
                
        except Exception:
            pass
    
    # Phase 3: Final high-precision polish at full resolution
    if time.time() < deadline - safety_margin:
        try:
            def full_obj(x):
                area, grad = _union_area_grad_exact(x, M=20000)
                return area, grad
            
            res = minimize(
                full_obj,
                best_x,
                jac=True,
                method='L-BFGS-B',
                options={'maxiter': 50, 'gtol': 1e-8, 'ftol': 1e-10}
            )
            
            x_opt = res.x
            area_opt = _union_area(x_opt, M=20000)
            
            if area_opt < best_area:
                best_x = x_opt.copy()
                best_area = area_opt
                
        except Exception:
            pass
    
    # Ensure valid output
    if not np.all(np.isfinite(best_x)):
        best_x = _keich_init(n)
    
    return best_x


# EVOLVE-BLOCK-END
