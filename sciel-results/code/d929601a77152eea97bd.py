import numpy as np
import time
import scipy.optimize

N = 400

def _autoconv_lin(x):
    """Compute linear autoconvolution using FFT."""
    n = len(x)
    size = 1
    while size < 2 * n - 1:
        size *= 2
    fx = np.fft.rfft(x, size)
    return np.fft.irfft(fx * fx, size)[:2 * n - 1]

def _metric_std(x):
    """Standard metric: max |c_k| / S^2 including k=0."""
    s = np.sum(x)
    if abs(s) < 1e-12:
        return float('inf')
    c = _autoconv_lin(x)
    return float(np.max(np.abs(c))) / (s * s)

def _metric_offdiag(x):
    """Off-diagonal metric: max |c_k| for k!=0 / S^2."""
    s = np.sum(x)
    if abs(s) < 1e-12:
        return float('inf')
    c = _autoconv_lin(x)
    if len(c) > 1:
        return float(np.max(np.abs(c[1:]))) / (s * s)
    return 0.0

def _objective_combined(x, beta=50.0, weight_offdiag=0.5):
    """
    Combined objective: minimize max|c_k|/S^2 with emphasis on off-diagonal peaks.
    Uses soft-max surrogate for differentiability.
    """
    s = np.sum(x)
    if abs(s) < 1e-12:
        return 1e10
    
    c = _autoconv_lin(x)
    abs_c = np.abs(c)
    
    # Soft max of all |c_k|
    m_all = np.max(abs_c)
    if m_all < 1e-15:
        return 0.0
    soft_max_all = m_all + np.log(np.sum(np.exp(beta * (abs_c - m_all)))) / beta
    
    # Soft max of off-diagonal |c_k| (k != 0)
    if len(c) > 1:
        abs_c_off = np.abs(c[1:])
        m_off = np.max(abs_c_off)
        if m_off < 1e-15:
            soft_max_off = 0.0
        else:
            soft_max_off = m_off + np.log(np.sum(np.exp(beta * (abs_c_off - m_off)))) / beta
    else:
        soft_max_off = 0.0
    
    # Combined objective: primarily minimize overall max, but also penalize off-diagonal peaks
    obj = (1.0 - weight_offdiag) * soft_max_all / (s * s) + weight_offdiag * soft_max_off / (s * s)
    return obj

def _spectral_flatness_objective(x, beta=50.0):
    """
    Objective based on spectral flatness: minimize concentration of autoconvolution energy.
    This directly addresses the spectral concentration problem.
    """
    s = np.sum(x)
    if abs(s) < 1e-12:
        return 1e10
    
    c = _autoconv_lin(x)
    abs_c = np.abs(c)
    
    # Normalize by S^2
    norm_c = abs_c / (s * s)
    
    # Soft max to get a smooth surrogate for the peak
    m = np.max(norm_c)
    if m < 1e-15:
        return 0.0
    soft_max = m + np.log(np.sum(np.exp(beta * (norm_c - m)))) / beta
    
    # Also penalize variance in the spectrum to encourage flatness
    # The variance of the autoconvolution profile indicates concentration
    var_c = np.var(norm_c)
    
    # Combined: minimize peak and variance
    return soft_max + 0.1 * var_c

def _subgradient_refine(x, max_iters=50, tol=1e-8, beta=50.0):
    """
    Subgradient refinement for the non-smooth max|c_k| objective.
    Uses a projected gradient approach with adaptive step size.
    """
    x = x.copy()
    s = np.sum(x)
    if abs(s) < 1e-12:
        return x
    
    c = _autoconv_lin(x)
    abs_c = np.abs(c)
    m = np.max(abs_c)
    
    if m < 1e-15:
        return x
    
    # Find which lags are at the maximum
    threshold = m * 0.999
    active_mask = abs_c >= threshold
    
    # Compute subgradient w.r.t. x
    # c_k = sum_i x_i x_{i-k}
    # dc_k/dx_i = x_{i-k} + x_{i+k} (with boundary handling)
    n = len(x)
    grad = np.zeros(n)
    
    for k in range(len(c)):
        if not active_mask[k]:
            continue
        
        # c_k = sum_{i=k}^{n-1} x_i x_{i-k}
        # dc_k/dx_i = x_{i-k} for i >= k, and x_{i+k} for i < k (when k > 0)
        # More carefully: c_k = sum_j x_j x_{j-k} where j-k is in [0, n-1]
        
        # For a given lag k, the contribution to x_i is:
        # - x_{i-k} if i-k >= 0 (i.e., i >= k)
        # - x_{i+k} if i+k < n (i.e., i < n-k) for k > 0
        
        for i in range(n):
            contrib = 0.0
            if i >= k and i - k < n:
                contrib += x[i - k]
            if k > 0 and i + k < n:
                contrib += x[i + k]
            grad[i] += contrib
    
    # Normalize gradient
    grad_norm = np.linalg.norm(grad)
    if grad_norm < 1e-15:
        return x
    
    # Adaptive step size
    step_size = 0.1 / grad_norm
    
    # Update
    x_new = x - step_size * grad
    
    # Project: ensure sum is preserved (or close to it)
    s_new = np.sum(x_new)
    if abs(s_new) > 1e-12:
        x_new = x_new * (s / s_new)
    
    # Check convergence
    c_new = _autoconv_lin(x_new)
    m_new = np.max(np.abs(c_new))
    
    if abs(m_new - m) < tol:
        return x_new
    
    return x_new

def _polish_pass(x, max_iters=100, tol=1e-10):
    """
    High-precision polish using L-BFGS-B with tighter tolerances.
    """
    try:
        res = scipy.optimize.minimize(
            _objective_combined,
            x,
            method='L-BFGS-B',
            options={'maxiter': max_iters, 'ftol': tol, 'gtol': 1e-8}
        )
        return res.x
    except Exception:
        return x

def _generate_random_signs(rng, N):
    """Generate random sign sequence with sum normalized to 1."""
    x = rng.choice([-1.0, 1.0], size=N).astype(float)
    s = np.sum(x)
    if abs(s) > 1e-6:
        x = x / s
    return x

def _generate_m_sequence(N):
    """
    Generate an m-sequence (maximum length sequence) using a primitive polynomial.
    This gives good autocorrelation properties.
    """
    # Use a simple LFSR with a known primitive polynomial for N=400
    # 400 is not 2^m - 1, so we'll use a truncated version
    # For simplicity, use a random-looking sequence with good properties
    
    # Actually, let's use a different approach: generate a sequence with 
    # approximately flat spectrum by using a windowed approach
    
    # Start with random signs
    rng = np.random.default_rng(12345)
    x = rng.choice([-1.0, 1.0], size=N).astype(float)
    
    # Apply a few iterations of spectral flattening
    for _ in range(10):
        # Compute FFT
        fx = np.fft.fft(x)
        # Flatten magnitude
        mag = np.abs(fx)
        phase = np.angle(fx)
        target_mag = np.ones_like(mag)
        # Blend
        fx_new = target_mag * np.exp(1j * phase)
        x = np.real(np.fft.ifft(fx_new))
        # Project to signs
        x = np.sign(x)
        # Fix zeros
        x[x == 0] = 1.0
    
    s = np.sum(x)
    if abs(s) > 1e-6:
        x = x / s
    
    return x

def construct():
    deadline = time.time() + 135.0
    rng = np.random.default_rng(42)
    
    best_x = np.ones(N)
    best_val_std = _metric_std(best_x)
    
    # Candidate 1: Constant sequence
    cand_const = np.ones(N)
    v_const = _metric_std(cand_const)
    
    # Candidate 2: M-sequence
    try:
        cand_m = _generate_m_sequence(N)
        v_m = _metric_std(cand_m)
        if v_m < best_val_std:
            best_val_std = v_m
            best_x = cand_m
    except Exception:
        pass
    
    # Candidate 3: Random signs
    for trial in range(5):
        if time.time() > deadline - 30:
            break
        try:
            cand_rand = _generate_random_signs(rng, N)
            v_rand = _metric_std(cand_rand)
            if v_rand < best_val_std:
                best_val_std = v_rand
                best_x = cand_rand
        except Exception:
            pass
    
    # Candidate 4: Optimize from best candidate using combined objective
    if time.time() < deadline - 20:
        try:
            x_opt = _polish_pass(best_x, max_iters=100, tol=1e-10)
            v_opt = _metric_std(x_opt)
            if v_opt < best_val_std:
                best_val_std = v_opt
                best_x = x_opt
        except Exception:
            pass
    
    # Candidate 5: Subgradient refinement
    if time.time() < deadline - 10:
        try:
            x_sub = _subgradient_refine(best_x, max_iters=50, tol=1e-8)
            v_sub = _metric_std(x_sub)
            if v_sub < best_val_std:
                best_val_std = v_sub
                best_x = x_sub
        except Exception:
            pass
    
    # Final polish
    if time.time() < deadline - 5:
        try:
            x_final = _polish_pass(best_x, max_iters=50, tol=1e-12)
            v_final = _metric_std(x_final)
            if v_final < best_val_std:
                best_val_std = v_final
                best_x = x_final
        except Exception:
            pass
    
    return best_x
