# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


def _union_area_and_grad(x, M=4000):
    """
    Compute the union area of 64 triangles and its gradient w.r.t. x.
    x: shape (64,) - base positions
    M: number of quadrature points
    Returns: (area, grad) where grad has shape (64,)
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


def _softmin_area_and_grad(x, beta=50.0, M=4000):
    """
    Smooth surrogate for the union area using soft-min approximation.
    Returns (area, grad) where grad has shape (64,).
    """
    n = 64
    a = (np.arange(n) + 1) / n
    y = (np.arange(M) + 0.5) / M
    
    L = x[np.newaxis, :] + a[np.newaxis, :] * y[:, np.newaxis]
    w = (1.0 - y) / n
    
    order = np.argsort(L, axis=1)
    L_sorted = np.take_along_axis(L, order, axis=1)
    diffs = np.diff(L_sorted, axis=1)
    
    neg_beta_diffs = -beta * diffs
    neg_beta_w = -beta * w[:, np.newaxis]
    
    max_vals = np.maximum(neg_beta_diffs, neg_beta_w)
    exp_diffs = np.exp(neg_beta_diffs - max_vals)
    exp_w = np.exp(neg_beta_w - max_vals)
    log_sum_exp = max_vals + np.log(exp_diffs + exp_w)
    
    soft_min = -log_sum_exp / beta
    union_len = w + np.sum(soft_min, axis=1)
    area = np.mean(union_len)
    
    denom = exp_diffs + exp_w
    weight_right = -exp_diffs / denom
    weight_left = exp_diffs / denom
    
    m_idx, j_idx = np.where(np.ones_like(diffs, dtype=bool))
    
    if len(m_idx) > 0:
        left_indices = order[m_idx, j_idx]
        right_indices = order[m_idx, j_idx + 1]
        
        grad_left = np.bincount(left_indices, weights=weight_left[m_idx, j_idx], minlength=n)
        grad_right = np.bincount(right_indices, weights=weight_right[m_idx, j_idx], minlength=n)
        grad = grad_right + grad_left
    else:
        grad = np.zeros(n, dtype=np.float64)
    
    grad /= M
    
    return area, grad


def _compute_area_exact(x, n=64, n_heights=20000):
    """Compute the area of the union of triangles for given base positions x (full precision)."""
    a = (np.arange(n) + 1) / n
    y_vals = (np.arange(n_heights) + 0.5) / n_heights
    w_vals = (1.0 - y_vals) / n
    L = x[None, :] + a[None, :] * y_vals[:, None]
    L_sorted = np.sort(L, axis=1)
    gaps = np.diff(L_sorted, axis=1)
    union_lens = w_vals + np.sum(np.minimum(gaps, w_vals[:, None]), axis=1)
    return np.mean(union_lens)


def _keich_initial(n=64):
    """Generate the Keich construction starting point."""
    k = int(np.log2(n))
    j = np.arange(1, k + 1)
    x0 = np.zeros(n)
    for i in range(n):
        eps = np.array(list(str(bin(i))[2:]), dtype=int)
        eps = np.pad(eps, (k - len(eps), 0), "constant")
        x0[i] = np.sum((1 - j) * eps * (2.0 ** -j) / k)
    return x0


def _structured_perturbation(x, rng, scale=0.1):
    """
    Apply structured perturbations: subset shifts, pairwise swaps.
    Returns a perturbed copy.
    """
    n = len(x)
    x_new = x.copy()
    
    subset_size = rng.integers(4, 16)
    subset = rng.choice(n, size=subset_size, replace=False)
    x_new[subset] += rng.normal(0, scale, size=subset_size)
    
    if rng.random() < 0.5:
        i, j = rng.choice(n, size=2, replace=False)
        swap_amount = rng.normal(0, scale * 0.5)
        x_new[i] += swap_amount
        x_new[j] -= swap_amount
    
    return x_new


def _subgradient_polish(x, n=64, M=20000, max_iters=100, deadline=None):
    """
    Subgradient method polish with adaptive step sizes for the nonsmooth objective.
    """
    x = x.copy()
    best_area = _compute_area_exact(x, n, M)
    step = 0.01
    patience = 0
    max_patience = 10
    
    for it in range(max_iters):
        if deadline is not None and time.time() > deadline - 5:
            break
        
        area, grad = _union_area_and_grad(x, M=M)
        
        if np.linalg.norm(grad, ord=np.inf) < 1e-8:
            break
        
        x_new = x - step * grad
        
        if not np.all(np.isfinite(x_new)):
            step *= 0.5
            continue
        
        area_new = _compute_area_exact(x_new, n, M)
        
        if area_new < best_area:
            best_area = area_new
            x = x_new
            patience = 0
            step = min(step * 1.1, 0.05)
        else:
            patience += 1
            step *= 0.7
            if patience > max_patience:
                break
    
    return x, best_area


def _coordinate_descent_polish(x, n=64, M=20000, max_rounds=30, deadline=None):
    """
    Coordinate descent polish: for each coordinate, use ternary search to find
    the optimal position that minimizes area.
    """
    x = x.copy()
    best_area = _compute_area_exact(x, n, M)
    
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
            
            # Ternary search over a window around current value
            lo = current_xi - 0.02
            hi = current_xi + 0.02
            
            for _ in range(15):
                m1 = lo + (hi - lo) / 3
                m2 = hi - (hi - lo) / 3
                
                x[i] = m1
                area_m1 = _compute_area_exact(x, n, M)
                x[i] = m2
                area_m2 = _compute_area_exact(x, n, M)
                
                if area_m1 < area_m2:
                    hi = m2
                else:
                    lo = m1
                
                x[i] = current_xi
            
            mid = (lo + hi) / 2
            x[i] = mid
            area_mid = _compute_area_exact(x, n, M)
            
            if area_mid < best_local_area - 1e-14:
                best_local_area = area_mid
                best_xi = mid
                improved = True
            else:
                x[i] = current_xi
        
        if not improved:
            break
        best_area = min(best_area, best_local_area)
    
    return x, best_area


def kakeya_needle_2d_positions64() -> np.ndarray:
    """
    Two-dimensional Kakeya-needle area-minimization problem for n=64.
    
    Uses the Keich construction as initialization, then refines with:
    1. L-BFGS-B with analytical gradient at reduced resolution (M=4000)
    2. Soft-min surrogate optimization with adaptive beta for smooth navigation
    3. Subgradient polish with adaptive step sizes
    4. Coordinate descent polish at full resolution (M=20000)
    Multiple restarts with structured perturbations to escape local minima.
    
    Returns:
        positions: np.ndarray of shape (64,) with the base x-coordinates x_i.
    """
    n = 64
    start_time = time.time()
    deadline = start_time + 625.0
    safety_margin = 25.0
    
    x0 = _keich_initial(n)
    best_x = x0.copy()
    best_area = _compute_area_exact(best_x)
    
    rng = np.random.default_rng(42)
    
    initial_guesses = [x0.copy()]
    
    for attempt in range(10):
        scale = 0.3 / (attempt + 1)
        perturbed = _structured_perturbation(x0, rng, scale)
        initial_guesses.append(perturbed)
    
    for attempt in range(5):
        perturbation_scale = 0.5 / (attempt + 1)
        perturbed = x0 + rng.normal(0, perturbation_scale, size=n)
        initial_guesses.append(perturbed)
    
    for attempt in range(5):
        random_start = rng.uniform(-1, 1, size=n)
        initial_guesses.append(random_start)
    
    # Phase 1: L-BFGS-B with analytical gradient (fast M=4000)
    for idx, x_start in enumerate(initial_guesses):
        if time.time() > deadline - safety_margin - 120.0:
            break
        
        try:
            result = minimize(
                lambda x: _union_area_and_grad(x, M=4000)[0],
                x_start,
                method='L-BFGS-B',
                jac=lambda x: _union_area_and_grad(x, M=4000)[1],
                options={
                    'maxiter': 150,
                    'maxfun': 400,
                    'ftol': 1e-12,
                    'gtol': 1e-8,
                    'maxls': 20,
                }
            )
            
            if np.all(np.isfinite(result.x)):
                area_candidate = _compute_area_exact(result.x)
                if np.isfinite(area_candidate) and area_candidate > 0 and area_candidate < best_area:
                    best_x = result.x.copy()
                    best_area = area_candidate
        except Exception:
            pass
    
    # Phase 2: Soft-min surrogate with adaptive beta
    if time.time() < deadline - safety_margin - 60.0:
        beta_schedule = [50.0, 100.0, 200.0, 400.0]
        for beta in beta_schedule:
            if time.time() > deadline - safety_margin - 30.0:
                break
            
            try:
                result_soft = minimize(
                    lambda x: _softmin_area_and_grad(x, beta=beta, M=4000)[0],
                    best_x,
                    method='L-BFGS-B',
                    jac=lambda x: _softmin_area_and_grad(x, beta=beta, M=4000)[1],
                    options={
                        'maxiter': 200,
                        'ftol': 1e-12,
                        'gtol': 1e-8,
                        'maxls': 20,
                    }
                )
                
                if np.all(np.isfinite(result_soft.x)):
                    area_candidate = _compute_area_exact(result_soft.x)
                    if np.isfinite(area_candidate) and area_candidate > 0 and area_candidate < best_area:
                        best_x = result_soft.x.copy()
                        best_area = area_candidate
            except Exception:
                pass
    
    # Phase 3: Subgradient polish with adaptive step sizes
    if time.time() < deadline - safety_margin - 20.0:
        try:
            polished_x, polished_area = _subgradient_polish(
                best_x, n, M=20000, max_iters=50, deadline=deadline
            )
            if polished_area < best_area:
                best_x = polished_x.copy()
                best_area = polished_area
        except Exception:
            pass
    
    # Phase 4: Coordinate descent polish at full resolution
    if time.time() < deadline - 5.0:
        try:
            polished_x, polished_area = _coordinate_descent_polish(
                best_x, n, M=20000, max_rounds=30, deadline=deadline
            )
            if polished_area < best_area:
                best_x = polished_x.copy()
                best_area = polished_area
        except Exception:
            pass
    
    # Ensure valid output
    if not np.all(np.isfinite(best_x)):
        best_x = x0
    
    return best_x.astype(np.float64)

# EVOLVE-BLOCK-END
