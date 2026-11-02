import numpy as np
from scipy.optimize import minimize
import time

N = 50
_TIME_LIMIT = 135.0
_start_time = time.time()

def _score(h):
    """
    Score function: (sum h_i^2) / ( (sum h_i) * max_k conv_k )
    where conv is the linear autoconvolution.
    """
    h = np.asarray(h, dtype=float)
    s2 = np.dot(h, h)
    s1 = np.sum(h)
    if s1 < 1e-12:
        return 0.0
    # Linear autoconvolution
    conv = np.convolve(h, h, mode='full')
    max_conv = np.max(conv)
    if max_conv < 1e-12:
        return 0.0
    return s2 / (s1 * max_conv)

def _soft_max_score(h, temperature=1.0):
    """
    Score function with soft-max surrogate for the autoconvolution max.
    This makes the objective smooth and differentiable.
    """
    h = np.asarray(h, dtype=float)
    s2 = np.dot(h, h)
    s1 = np.sum(h)
    if s1 < 1e-12:
        return 0.0
    # Linear autoconvolution
    conv = np.convolve(h, h, mode='full')
    # Soft max approximation
    if temperature <= 0:
        max_conv = np.max(conv)
    else:
        # Log-sum-exp trick for numerical stability
        c_max = np.max(conv)
        log_sum_exp = c_max + np.log(np.sum(np.exp((conv - c_max) / temperature)))
        max_conv = temperature * log_sum_exp
    if max_conv < 1e-12:
        return 0.0
    return s2 / (s1 * max_conv)

def _soft_max_score_neg(h, temperature=1.0):
    """Negative soft-max score for minimization."""
    return -_soft_max_score(h, temperature)

def _soft_max_grad(h, temperature=1.0):
    """
    Gradient of the soft-max score with respect to h.
    Uses finite differences or analytical computation.
    """
    h = np.asarray(h, dtype=float)
    s2 = np.dot(h, h)
    s1 = np.sum(h)
    if s1 < 1e-12:
        return np.zeros_like(h)
    
    conv = np.convolve(h, h, mode='full')
    
    # Soft max weights
    if temperature <= 0:
        max_idx = np.argmax(conv)
        weights = np.zeros_like(conv)
        weights[max_idx] = 1.0
    else:
        c_max = np.max(conv)
        exp_vals = np.exp((conv - c_max) / temperature)
        sum_exp = np.sum(exp_vals)
        weights = exp_vals / sum_exp
    
    max_conv = temperature * (c_max + np.log(np.sum(exp_vals))) if temperature > 0 else np.max(conv)
    if max_conv < 1e-12:
        return np.zeros_like(h)
    
    # Score = s2 / (s1 * max_conv)
    # d(score)/d(h_i) = [2*h_i * s1 * max_conv - s2 * (d(s1)/d(h_i) * max_conv + s1 * d(max_conv)/d(h_i))] / (s1 * max_conv)^2
    # d(s1)/d(h_i) = 1
    # d(max_conv)/d(h_i) = sum_k weights_k * d(conv_k)/d(h_i)
    # conv_k = sum_j h_j * h_{k-j} (for valid indices)
    # d(conv_k)/d(h_i) = h_{k-i} + h_{i-k} (with appropriate bounds)
    
    # Compute d(max_conv)/d(h_i) for each i
    d_max_conv_dh = np.zeros(N)
    for i in range(N):
        for k in range(2*N - 1):
            # conv_k = sum_j h_j * h_{k-j}
            # d(conv_k)/d(h_i) = h_{k-i} + h_{i-k} if indices valid
            term1 = h[k-i] if 0 <= k-i < N else 0.0
            term2 = h[i-k] if 0 <= i-k < N else 0.0
            d_conv_k_dh_i = term1 + term2
            d_max_conv_dh[i] += weights[k] * d_conv_k_dh_i
    
    # Now compute gradient
    score = s2 / (s1 * max_conv)
    grad = np.zeros(N)
    for i in range(N):
        ds2_dh_i = 2 * h[i]
        ds1_dh_i = 1.0
        dmaxconv_dh_i = d_max_conv_dh[i]
        
        # d(score)/d(h_i) = [ds2_dh_i * s1 * max_conv - s2 * (ds1_dh_i * max_conv + s1 * dmaxconv_dh_i)] / (s1 * max_conv)^2
        numerator = ds2_dh_i * s1 * max_conv - s2 * (ds1_dh_i * max_conv + s1 * dmaxconv_dh_i)
        denominator = (s1 * max_conv) ** 2
        grad[i] = numerator / denominator
    
    return grad

def _optimize_smooth(h0, max_iter=200, temperature_start=5.0, temperature_end=0.1):
    """
    Optimize using smooth soft-max surrogate with annealing temperature.
    """
    h = np.array(h0, dtype=float)
    h = np.clip(h, 0, None)
    
    best_h = h.copy()
    best_score = _score(h)
    
    # Anneal temperature from high to low
    n_anneal = 5
    for stage in range(n_anneal):
        if time.time() - _start_time > _TIME_LIMIT - 10:
            break
        
        t_frac = stage / max(1, n_anneal - 1)
        temperature = temperature_start * (temperature_end / temperature_start) ** t_frac
        
        # Run optimization with current temperature
        for _ in range(max_iter // n_anneal):
            if time.time() - _start_time > _TIME_LIMIT - 10:
                break
            
            # Compute gradient
            grad = _soft_max_grad(h, temperature)
            
            # Gradient ascent with step size
            step_size = 0.01
            h_new = h + step_size * grad
            h_new = np.clip(h_new, 0, None)
            
            # Check improvement
            s_new = _soft_max_score(h_new, temperature)
            s_old = _soft_max_score(h, temperature)
            
            if s_new > s_old:
                h = h_new
            else:
                # Reduce step size
                step_size *= 0.5
            
            # Update best
            s_actual = _score(h)
            if s_actual > best_score:
                best_score = s_actual
                best_h = h.copy()
    
    return best_h, best_score

def _local_polish(h0, max_iter=100):
    """
    High-precision local polish using scipy.optimize with exact score.
    """
    h = np.array(h0, dtype=float)
    h = np.clip(h, 0, None)
    
    best_h = h.copy()
    best_score = _score(h)
    
    # Use Nelder-Mead for local optimization
    try:
        result = minimize(
            _score_neg,
            h,
            method='Nelder-Mead',
            options={'maxiter': max_iter, 'xatol': 1e-8, 'fatol': 1e-10}
        )
        if result.success or result.fun < -best_score:
            h_opt = np.clip(result.x, 0, None)
            s_opt = _score(h_opt)
            if s_opt > best_score:
                best_score = s_opt
                best_h = h_opt
    except Exception:
        pass
    
    return best_h, best_score

def construct():
    # Known best vector
    h0 = [0.0] * N
    h0[0] = 2.0 / 3.0
    h0[1] = 1.0 / 3.0
    
    best_h = np.array(h0)
    best_score = _score(best_h)
    
    # Try a few different initializations
    inits = [
        h0,
        [1.0/2, 1.0/2] + [0.0]*(N-2),
        [1.0/3]*3 + [0.0]*(N-3),
        [0.4, 0.3, 0.3] + [0.0]*(N-3),
        [0.5, 0.25, 0.25] + [0.0]*(N-3),
        [0.6, 0.2, 0.2] + [0.0]*(N-3),
        [0.7, 0.15, 0.15] + [0.0]*(N-3),
    ]
    
    for init in inits:
        if time.time() - _start_time > _TIME_LIMIT - 20:
            break
        h_opt, s_opt = _optimize_smooth(init, max_iter=150, temperature_start=3.0, temperature_end=0.2)
        if s_opt > best_score:
            best_score = s_opt
            best_h = h_opt
    
    # Random search around known best
    rng = np.random.default_rng(42)
    n_random = 30
    for _ in range(n_random):
        if time.time() - _start_time > _TIME_LIMIT - 30:
            break
        h_pert = np.array(h0, dtype=float)
        noise = rng.normal(0, 0.05, N)
        h_pert = h_pert + noise
        h_pert = np.clip(h_pert, 0, None)
        s_pert = _score(h_pert)
        if s_pert > best_score:
            best_score = s_pert
            best_h = h_pert
    
    # Final high-precision polish
    if time.time() - _start_time < _TIME_LIMIT - 15:
        h_polished, s_polished = _local_polish(best_h, max_iter=50)
        if s_polished > best_score:
            best_score = s_polished
            best_h = h_polished
    
    return [float(x) for x in best_h]
