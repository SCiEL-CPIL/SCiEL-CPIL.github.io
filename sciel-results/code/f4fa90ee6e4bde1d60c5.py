# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


def _compute_area_and_grad(x, n=64, M=20000):
    """Compute area and gradient of union area using vectorized operations with bincount."""
    a = (np.arange(n) + 1) / n
    w0 = 1.0 / n
    y = (np.arange(M) + 0.5) / M

    # L_i(y) = x_i + a_i * y, shape (n, M)
    L = x[:, None] + a[:, None] * y[None, :]
    w = (1.0 - y[None, :]) * w0

    # Sort L at each y
    idx = np.argsort(L, axis=0)
    L_sorted = np.take_along_axis(L, idx, axis=0)

    # Gaps between consecutive sorted L values
    gaps = np.diff(L_sorted, axis=0)  # (n-1, M)

    # Union length = w + sum(min(gap, w))
    union_lengths = w + np.minimum(gaps, w[None, :]).sum(axis=0)
    area = np.mean(union_lengths)

    # Gradient computation using bincount
    active = gaps < w[None, :]  # (n-1, M)
    
    # Get indices of active gaps
    m_idx, j_idx = np.where(active)
    
    if len(m_idx) > 0:
        left_indices = idx[:-1, :][m_idx, j_idx]
        right_indices = idx[1:, :][m_idx, j_idx]
        
        # Use bincount for efficient accumulation
        grad_right = np.bincount(right_indices, minlength=n).astype(np.float64)
        grad_left = np.bincount(left_indices, minlength=n).astype(np.float64)
        
        grad_x = (grad_right - grad_left) / M
    else:
        grad_x = np.zeros(n, dtype=np.float64)

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


def _soft_min_surrogate(x, n=64, M=20000, beta=100.0):
    """
    Compute a differentiable surrogate for the union area using soft-min.
    This allows gradient-based optimization to escape local minima.
    """
    a = (np.arange(n) + 1) / n
    w0 = 1.0 / n
    y = (np.arange(M) + 0.5) / M

    L = x[:, None] + a[:, None] * y[None, :]
    w = (1.0 - y[None, :]) * w0

    idx = np.argsort(L, axis=0)
    L_sorted = np.take_along_axis(L, idx, axis=0)
    gaps = np.diff(L_sorted, axis=0)  # (n-1, M)

    # Soft-min: log-sum-exp based approximation to min(gaps, w)
    # min(a, b) ≈ -1/beta * log(exp(-beta*a) + exp(-beta*b))
    # But we want min(gaps, w), so:
    # soft_min = -1/beta * log(exp(-beta*gaps) + exp(-beta*w))
    
    # For numerical stability, use the log-sum-exp trick
    # soft_min(gaps, w) = min(gaps, w) - 1/beta * log(1 + exp(-beta*|gaps - w|))
    # Actually, simpler: soft_min(a, b) = -1/beta * log(exp(-beta*a) + exp(-beta*b))
    
    # Let's use a different approach: soft-min of gaps and w
    # soft_min = min(gaps, w) - 1/beta * log(1 + exp(-beta * (gaps - w)))
    # This is numerically stable
    
    diff = gaps - w[None, :]
    # soft_min = min(gaps, w) - 1/beta * log(1 + exp(-beta * abs(diff)))
    # But we need the gradient, so let's use the log-sum-exp form
    
    # Actually, let's use: soft_min(a, b) = -1/beta * log(exp(-beta*a) + exp(-beta*b))
    # Gradient w.r.t. a: exp(-beta*a) / (exp(-beta*a) + exp(-beta*b))
    # Gradient w.r.t. b: exp(-beta*b) / (exp(-beta*a) + exp(-beta*b))
    
    # For numerical stability:
    # soft_min = -1/beta * log(exp(-beta*gaps) + exp(-beta*w))
    # = -1/beta * log(exp(-beta*max(gaps,w)) * (exp(-beta*(gaps-max)) + exp(-beta*(w-max))))
    # = max(gaps,w) - 1/beta * log(exp(-beta*(gaps-max)) + exp(-beta*(w-max)))
    
    max_val = np.maximum(gaps, w[None, :])
    exp_gaps = np.exp(-beta * (gaps - max_val))
    exp_w = np.exp(-beta * (w[None, :] - max_val))
    
    log_sum = np.log(exp_gaps + exp_w)
    soft_min = max_val - log_sum / beta
    
    union_lengths = w + soft_min.sum(axis=0)
    area = np.mean(union_lengths)
    
    # Gradient
    # d(soft_min)/d(gaps) = exp_gaps / (exp_gaps + exp_w)
    # d(soft_min)/d(w) = exp_w / (exp_gaps + exp_w)
    # But w doesn't depend on x, so we only need d/d(gaps)
    
    weight = exp_gaps / (exp_gaps + exp_w)  # (n-1, M)
    
    # d(area)/d(x_i) = (1/M) * sum_m d(union_length_m)/d(x_i)
    # d(union_length_m)/d(x_i) = sum_j weight[j,m] * d(gaps[j,m])/d(x_i)
    # d(gaps[j,m])/d(x_i) = delta_{idx[j+1,m], i} - delta_{idx[j,m], i}
    
    # Use bincount
    m_idx, j_idx = np.where(np.ones_like(weight, dtype=bool))
    
    if len(m_idx) > 0:
        left_indices = idx[:-1, :][m_idx, j_idx]
        right_indices = idx[1:, :][m_idx, j_idx]
        weights_flat = weight[m_idx, j_idx]
        
        grad_right = np.bincount(right_indices, weights=weights_flat, minlength=n).astype(np.float64)
        grad_left = np.bincount(left_indices, weights=weights_flat, minlength=n).astype(np.float64)
        
        grad_x = (grad_right - grad_left) / M
    else:
        grad_x = np.zeros(n, dtype=np.float64)

    return area, grad_x


def kakeya_needle_2d_positions64() -> np.ndarray:
    """
    Two-dimensional Kakeya-needle area-minimization problem for n=64.
    
    Uses a combination of:
    1. Soft-min surrogate with increasing beta for basin escape
    2. Subgradient polish with adaptive step sizes for nonsmoothness
    3. L-BFGS-B with bincount-based gradient for final refinement
    """
    start_time = time.time()
    deadline = start_time + 570  # 570 second limit (30s safety margin)

    n = 64
    M = 20000

    # Generate initial points
    x_keich = _keich_initial(n)
    
    # State for caching
    _state = {'x': None, 'area': None, 'grad': None}
    
    def _fun_jac_combined(x):
        """Combined function and gradient evaluation with caching."""
        if _state['x'] is not None and np.array_equal(x, _state['x']):
            return _state['area'], _state['grad']
        
        area, grad = _compute_area_and_grad(x, n, M)
        _state['x'] = x.copy()
        _state['area'] = area
        _state['grad'] = grad
        return area, grad
    
    def _fun_only(x):
        area, _ = _fun_jac_combined(x)
        return area
    
    def _jac_only(x):
        _, grad = _fun_jac_combined(x)
        return grad
    
    best_x = x_keich.copy()
    best_area = _compute_area_fast(x_keich, n, M)
    
    # Phase 1: Soft-min surrogate with increasing beta
    # This helps escape local minima by providing smooth gradients
    beta_schedule = [50, 100, 200, 400, 800]
    
    for beta in beta_schedule:
        elapsed = time.time() - start_time
        if elapsed > deadline - 120:  # Leave 120s for later phases
            break
        
        # Reset cache for new beta
        _state['x'] = None
        _state['area'] = None
        _state['grad'] = None
        
        def _soft_fun_jac(x):
            return _soft_min_surrogate(x, n, M, beta)
        
        def _soft_fun(x):
            area, _ = _soft_fun_jac(x)
            return area
        
        def _soft_jac(x):
            _, grad = _soft_fun_jac(x)
            return grad
        
        try:
            result = minimize(
                fun=_soft_fun,
                x0=best_x,
                jac=_soft_jac,
                method='L-BFGS-B',
                options={'maxiter': 200, 'ftol': 1e-10, 'gtol': 1e-6}
            )
            
            if result.fun < 1e6:  # Sanity check
                # Evaluate at full resolution
                area_full = _compute_area_fast(result.x, n, M)
                if area_full < best_area:
                    best_area = area_full
                    best_x = result.x.copy()
        except Exception:
            continue
    
    # Phase 2: Subgradient polish with adaptive step sizes
    # Handles nonsmoothness where L-BFGS struggles
    elapsed = time.time() - start_time
    if elapsed < deadline - 60:
        # Subgradient method with adaptive step size
        x_curr = best_x.copy()
        step_size = 0.01
        best_x_sub = x_curr.copy()
        best_area_sub = best_area
        
        for iteration in range(100):
            if time.time() - start_time > deadline - 60:
                break
            
            area, grad = _compute_area_and_grad(x_curr, n, M)
            
            if area < best_area_sub:
                best_area_sub = area
                best_x_sub = x_curr.copy()
            
            # Adaptive step size: decrease if area increases
            x_next = x_curr - step_size * grad
            
            # Check if step was good
            area_next = _compute_area_fast(x_next, n, M)
            
            if area_next < area:
                # Good step, keep or increase step size
                x_curr = x_next
                step_size = min(step_size * 1.1, 0.05)
            else:
                # Bad step, decrease step size
                step_size = step_size * 0.5
            
            # Early termination if step size is too small
            if step_size < 1e-6:
                break
        
        if best_area_sub < best_area:
            best_area = best_area_sub
            best_x = best_x_sub.copy()
    
    # Phase 3: Final L-BFGS-B refinement with exact gradient
    elapsed = time.time() - start_time
    if elapsed < deadline - 30:
        # Reset cache
        _state['x'] = None
        _state['area'] = None
        _state['grad'] = None
        
        try:
            result = minimize(
                fun=_fun_only,
                x0=best_x,
                jac=_jac_only,
                method='L-BFGS-B',
                options={'maxiter': 500, 'ftol': 1e-14, 'gtol': 1e-10}
            )
            
            if result.fun < best_area:
                area_full = _compute_area_fast(result.x, n, M)
                if area_full < best_area:
                    best_area = area_full
                    best_x = result.x.copy()
        except Exception:
            pass
    
    # Ensure valid output
    if not np.all(np.isfinite(best_x)):
        best_x = x_keich
    
    return best_x.astype(np.float64)


# EVOLVE-BLOCK-END
