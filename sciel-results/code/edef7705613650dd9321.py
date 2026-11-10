# EVOLVE-BLOCK-START
import numpy as np
import time


def _union_area(x, M=20000):
    """
    Compute the union area of 64 triangles.
    
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


def _coordinate_descent_polish(x0, M=2000, max_rounds=5, timeout=20.0):
    """
    Coordinate descent polish: optimize each x_i individually while holding others fixed.
    Uses golden-section search for each coordinate.
    """
    n = 64
    start = time.time()
    x = x0.copy()
    best_area = _union_area(x, M=M)
    
    for rnd in range(max_rounds):
        if time.time() - start > timeout:
            break
        improved = False
        for i in range(n):
            if time.time() - start > timeout:
                break
            # Golden-section search for x[i]
            lo, hi = x[i] - 0.05, x[i] + 0.05
            gr = (np.sqrt(5.0) - 1.0) / 2.0
            c = hi - gr * (hi - lo)
            d = lo + gr * (hi - lo)
            x[i] = c
            fc = _union_area(x, M=M)
            x[i] = d
            fd = _union_area(x, M=M)
            
            for _ in range(20):
                if time.time() - start > timeout:
                    break
                if fc < fd:
                    hi = d
                    d = c
                    fd = fc
                    c = hi - gr * (hi - lo)
                    x[i] = c
                    fc = _union_area(x, M=M)
                else:
                    lo = c
                    c = d
                    fc = fd
                    d = lo + gr * (hi - lo)
                    x[i] = d
                    fd = _union_area(x, M=M)
            
            best_c = c if fc < fd else d
            best_f = min(fc, fd)
            if best_f < best_area - 1e-12:
                x[i] = best_c
                best_area = best_f
                improved = True
            else:
                x[i] = x0[i]  # revert if no improvement
        
        if not improved:
            break
    
    return x, best_area


def _coordinate_descent_polish(x0, M=2000, max_rounds=5, timeout=20.0):
    """
    Coordinate descent polish: optimize each x_i individually while holding others fixed.
    Uses golden-section search for each coordinate.
    """
    n = 64
    start = time.time()
    x = x0.copy()
    best_area = _union_area(x, M=M)
    
    for rnd in range(max_rounds):
        if time.time() - start > timeout:
            break
        improved = False
        for i in range(n):
            if time.time() - start > timeout:
                break
            # Golden-section search for x[i]
            lo, hi = x[i] - 0.1, x[i] + 0.1
            gr = (np.sqrt(5.0) - 1.0) / 2.0
            c = hi - gr * (hi - lo)
            d = lo + gr * (hi - lo)
            x[c_idx] = c
            fc = _union_area(x, M=M)
            x[d_idx] = d
            fd = _union_area(x, M=M)
            
            for _ in range(30):
                if time.time() - start > timeout:
                    break
                if fc < fd:
                    hi = d
                    d = c
                    fd = fc
                    c = hi - gr * (hi - lo)
                    x[c_idx] = c
                    fc = _union_area(x, M=M)
                else:
                    lo = c
                    c = d
                    fc = fd
                    d = lo + gr * (hi - lo)
                    x[d_idx] = d
                    fd = _union_area(x, M=M)
            
            best_c = c if fc < fd else d
            best_f = min(fc, fd)
            if best_f < best_area - 1e-12:
                x[i] = best_c
                best_area = best_f
                improved = True
            else:
                x[i] = x0[i]  # revert if no improvement
        
        if not improved:
            break
    
    return x, best_area


def _union_area_grad(x, M=20000):
    """
    Compute the union area and its gradient w.r.t. x using exact min.
    Uses bincount for fast gradient accumulation.
    
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
    
    # mask: (M, n-1) boolean, True where diffs < w (overlapping)
    mask = diffs < w[:, np.newaxis]
    
    # union_len = w + sum min(diffs, w)
    union_len = w + np.sum(np.minimum(diffs, w[:, np.newaxis]), axis=1)
    area = np.mean(union_len)
    
    # Gradient: where mask is True, +1 to left index, -1 to right index
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


def _soft_min_area(x, M=2000, beta=50.0):
    """
    Compute a smooth surrogate for the union area using soft-min.
    
    x: shape (64,) - base positions
    M: number of quadrature points
    beta: inverse temperature for soft-min (higher = sharper)
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
    diffs = np.diff(L_sorted, axis=1)  # shape (M, n-1)
    w_col = w[:, np.newaxis]  # shape (M, 1)
    
    # Soft-min of (diffs, w):
    # soft_min(a,b) = min(a,b) - (1/beta)*log(1 + exp(-beta*|a-b|))
    diff_abs = np.abs(diffs - w_col)
    s_min = np.minimum(diffs, w_col)
    
    # Use stable computation: log(1+exp(-beta*|d|)) is safe since -beta*|d| <= 0
    exp_term = np.exp(-beta * diff_abs)
    soft_correction = (1.0 / beta) * np.log1p(exp_term)
    soft_min_vals = s_min - soft_correction  # shape (M, n-1)
    
    # union_len = w + sum soft_min_vals
    union_len = w + np.sum(soft_min_vals, axis=1)
    area = np.mean(union_len)
    
    # Gradient: d(soft_min)/d(diffs) = sigmoid(-beta*(diffs-w)) for the diff term
    # and d(soft_min)/d(w) contributes to the w term (but w depends on y, not x)
    # So gradient w.r.t. x only comes through diffs
    d_diff = diffs - w_col
    # sigmoid(-beta * d_diff) = 1/(1+exp(beta*d_diff))
    # Use stable sigmoid
    g_plus = np.empty_like(d_diff)
    mask_pos = d_diff >= 0
    g_plus[mask_pos] = 1.0 / (1.0 + np.exp(beta * d_diff[mask_pos]))
    exp_neg = np.exp(-beta * d_diff[~mask_pos])
    g_plus[~mask_pos] = exp_neg / (1.0 + exp_neg)
    
    # Accumulate gradient using bincount for speed
    # g_plus has shape (M, n-1)
    # For each (m,k): grad[order[m,k+1]] += g_plus[m,k], grad[order[m,k]] -= g_plus[m,k]
    
    # Flatten
    g_flat = g_plus.flatten()
    order_plus_flat = order[:, 1:].flatten()
    order_minus_flat = order[:, :-1].flatten()
    
    # Use bincount with weights for fast accumulation
    grad_plus = np.bincount(order_plus_flat, weights=g_flat, minlength=n)
    grad_minus = np.bincount(order_minus_flat, weights=g_flat, minlength=n)
    grad = grad_plus - grad_minus
    
    # Scale by 1/M (mean)
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


def _nm_optimize(x0, M=2000, maxiter=200, timeout=30.0):
    """
    Nelder-Mead optimization on the exact objective.
    Robust to non-smoothness.
    """
    from scipy.optimize import minimize
    
    start = time.time()
    
    def objective(x):
        if time.time() - start > timeout:
            raise TimeoutError("NM timeout")
        return _union_area(x, M=M)
    
    try:
        res = minimize(
            objective,
            x0,
            method='Nelder-Mead',
            options={'maxiter': maxiter, 'xatol': 1e-8, 'fatol': 1e-10, 'adaptive': True}
        )
        return res.x
    except (TimeoutError, Exception):
        return x0


def _lbfgs_optimize_soft(x0, maxiter=200, M=2000, beta=50.0, timeout=30.0):
    """
    Run L-BFGS-B optimization using the soft-min surrogate.
    Returns (x_opt, exact_area_opt).
    """
    from scipy.optimize import minimize
    
    start = time.time()
    
    def objective_soft(x):
        if time.time() - start > timeout:
            raise TimeoutError("L-BFGS timeout")
        area, grad = _soft_min_area(x, M=M, beta=beta)
        return area, grad
    
    try:
        res = minimize(
            objective_soft,
            x0,
            jac=True,
            method='L-BFGS-B',
            options={'maxiter': maxiter, 'gtol': 1e-8, 'ftol': 1e-10, 'maxls': 50}
        )
        x_opt = res.x
        exact_area_opt = _union_area(x_opt, M=M)
        return x_opt, exact_area_opt
    except (TimeoutError, Exception):
        return x0, _union_area(x0, M=M)


def _lbfgs_optimize_exact(x0, maxiter=50, M=20000, timeout=30.0, grad_M=None):
    """
    Run L-BFGS-B optimization using the exact objective.
    Uses grad_M for gradient computation if provided (faster), M for area eval.
    Returns (x_opt, area_opt).
    """
    from scipy.optimize import minimize
    
    if grad_M is None:
        grad_M = M
    
    start = time.time()
    
    def objective_exact(x):
        if time.time() - start > timeout:
            raise TimeoutError("L-BFGS exact timeout")
        area, grad = _union_area_grad(x, M=grad_M)
        return area, grad
    
    try:
        res = minimize(
            objective_exact,
            x0,
            jac=True,
            method='L-BFGS-B',
            options={'maxiter': maxiter, 'gtol': 1e-10, 'ftol': 1e-12, 'maxls': 30}
        )
        x_opt = res.x
        area_opt = _union_area(x_opt, M=M)
        return x_opt, area_opt
    except (TimeoutError, Exception):
        return x0, _union_area(x0, M=M)


def kakeya_needle_2d_positions64() -> np.ndarray:
    """
    Two-dimensional Kakeya-needle area-minimization problem for n=64.
    
    Strategy:
    1. Generate diverse initial points (Keich + perturbations + random)
    2. Optimize each with adaptive-beta soft-min L-BFGS (smooth, fast)
    3. Basin hopping with structured and random perturbations
    4. Multi-stage polish: NM at moderate M, L-BFGS with reduced-grad M, final NM
    5. Final validation at full resolution
    
    Returns:
        positions: np.ndarray of shape (64,) with the base x-coordinates x_i.
    """
    n = 64
    
    start_time = time.time()
    deadline = start_time + 620.0  # 620s limit (10s safety margin for 630s kill)
    
    def time_left():
        return deadline - time.time()
    
    # Initialize with Keich
    best_x = _keich_init(n)
    best_area = _union_area(best_x, M=2000)
    
    # Phase 1: Generate diverse initial points
    inits = []
    
    # Keich construction
    inits.append(_keich_init(n))
    
    # Perturbed Keich with various noise levels
    for seed in range(12):
        rng = np.random.default_rng(seed)
        noise_level = 0.01 + 0.06 * (seed % 5)
        inits.append(_keich_init(n) + rng.normal(0, noise_level, size=n))
    
    # Random initializations
    for seed in range(6):
        inits.append(_random_init(n, seed=200 + seed))
    
    # Structured initializations: shift subsets
    rng_struct = np.random.default_rng(999)
    for seed in range(4):
        x_struct = _keich_init(n).copy()
        # Randomly shift a random subset of positions
        mask = rng_struct.random(n) < 0.3
        x_struct[mask] += rng_struct.normal(0, 0.05, size=mask.sum())
        inits.append(x_struct)
    
    # Sort inits by quick area estimate
    init_areas = [_union_area(x, M=500) for x in inits]
    init_order = np.argsort(init_areas)
    inits = [inits[i] for i in init_order]
    
    # Phase 2: Soft-min L-BFGS optimization from each initial point
    # Use adaptive beta: start smoother, get sharper
    for i, x0 in enumerate(inits):
        if time_left() < 60.0:
            break
        
        # Use lower beta for early iterations (smoother landscape)
        beta_start = 20.0 + 10.0 * (i % 3)
        try:
            x_opt, exact_area = _lbfgs_optimize_soft(x0, maxiter=150, M=2000, beta=beta_start, timeout=40.0)
            
            if exact_area < best_area:
                best_x = x_opt.copy()
                best_area = exact_area
        except Exception:
            continue
    
    # Phase 3: Basin hopping - perturb best solution and re-optimize
    rng = np.random.default_rng(42)
    max_hops = 60
    hop_size = 0.04
    
    for hop in range(max_hops):
        if time_left() < 80.0:
            break
        if time.time() - start_time > 420.0:  # Reserve time for final polish
            break
        
        # Mix of random and structured perturbations
        hop_type = hop % 5
        if hop_type == 0:
            # Pure Gaussian
            perturbation = rng.normal(0, hop_size, size=n)
        elif hop_type == 1:
            # Shift random subset more
            perturbation = np.zeros(n)
            mask = rng.random(n) < 0.4
            perturbation[mask] = rng.normal(0, hop_size * 1.5, size=mask.sum())
        elif hop_type == 2:
            # Pairwise swap-like perturbation
            perturbation = np.zeros(n)
            for _ in range(8):
                i1, i2 = rng.choice(n, size=2, replace=False)
                delta = rng.normal(0, hop_size)
                perturbation[i1] += delta
                perturbation[i2] -= delta
        elif hop_type == 3:
            # Block shift: shift a contiguous block of indices
            perturbation = np.zeros(n)
            block_size = rng.integers(4, 16)
            start_idx = rng.integers(0, n - block_size)
            delta = rng.normal(0, hop_size * 1.2)
            perturbation[start_idx:start_idx + block_size] = delta
        else:
            # Sparse large perturbation on few coordinates
            perturbation = np.zeros(n)
            num_sparse = rng.integers(3, 8)
            sparse_idx = rng.choice(n, size=num_sparse, replace=False)
            perturbation[sparse_idx] = rng.normal(0, hop_size * 2.0, size=num_sparse)
        
        x_perturbed = best_x + perturbation
        
        # Use slightly higher beta for basin hopping (sharper)
        beta_hop = 60.0 + 20.0 * (hop % 2)
        try:
            # Optimize from perturbed point using soft-min
            x_opt, exact_area = _lbfgs_optimize_soft(x_perturbed, maxiter=100, M=2000, beta=beta_hop, timeout=30.0)
            
            if exact_area < best_area:
                best_x = x_opt.copy()
                best_area = exact_area
                hop_size = max(0.01, hop_size * 0.85)  # Reduce hop size on success
            else:
                hop_size = min(0.15, hop_size * 1.1)  # Increase hop size on failure
        except Exception:
            continue
    
    # Phase 4: Coordinate descent polish at moderate resolution
    if time_left() > 50.0:
        try:
            x_cd, area_cd = _coordinate_descent_polish(best_x, M=2000, max_rounds=3, timeout=30.0)
            if area_cd < best_area:
                best_x = x_cd.copy()
                best_area = area_cd
        except Exception:
            pass
    
    # Phase 4b: Nelder-Mead polish at moderate resolution (fast, no gradient needed)
    if time_left() > 40.0:
        try:
            x_nm = _nm_optimize(best_x, M=2000, maxiter=400, timeout=30.0)
            area_nm = _union_area(x_nm, M=2000)
            if area_nm < best_area:
                best_x = x_nm.copy()
                best_area = area_nm
        except Exception:
            pass
    
    # Phase 5: L-BFGS with exact area but reduced-resolution gradient
    # This is much faster than full-res gradient while still using exact area for step acceptance
    if time_left() > 40.0:
        try:
            x_final, area_final = _lbfgs_optimize_exact(best_x, maxiter=80, M=20000, grad_M=4000, timeout=30.0)
            if area_final < best_area:
                best_x = x_final.copy()
                best_area = area_final
        except Exception:
            pass
    
    # Phase 6: Final tight Nelder-Mead at full resolution if time allows
    if time_left() > 20.0:
        try:
            x_final = _nm_optimize(best_x, M=20000, maxiter=100, timeout=15.0)
            area_final = _union_area(x_final, M=20000)
            if area_final < best_area:
                best_x = x_final.copy()
                best_area = area_final
        except Exception:
            pass
    
    # Phase 7: One more L-BFGS pass with full-res gradient if time allows
    if time_left() > 25.0:
        try:
            x_final, area_final = _lbfgs_optimize_exact(best_x, maxiter=30, M=20000, grad_M=20000, timeout=15.0)
            if area_final < best_area:
                best_x = x_final.copy()
                best_area = area_final
        except Exception:
            pass
    
    # Phase 8: Final coordinate descent polish at high resolution
    if time_left() > 15.0:
        try:
            x_cd_final, area_cd_final = _coordinate_descent_polish(best_x, M=5000, max_rounds=2, timeout=10.0)
            if area_cd_final < best_area:
                best_x = x_cd_final.copy()
                best_area = area_cd_final
        except Exception:
            pass
    
    # Final validation
    final_area = _union_area(best_x, M=20000)
    if not np.isfinite(final_area) or final_area <= 0:
        # Fallback to Keich
        best_x = _keich_init(n)
    
    return best_x


# EVOLVE-BLOCK-END
