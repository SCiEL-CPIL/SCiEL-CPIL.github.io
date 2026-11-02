"""Spectral design with vectorized peak-targeted polish for signed step functions."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

START_TIME = time.time()
TIME_LIMIT = 115
SAFETY_MARGIN = 5

def _check_deadline():
    return (time.time() - START_TIME) > (TIME_LIMIT - SAFETY_MARGIN)

def _autoconv_fft(x):
    """Compute linear autoconvolution using FFT."""
    L = 2 * N - 1
    L_fft = 1
    while L_fft < L:
        L_fft *= 2
    X = np.fft.rfft(x, n=L_fft)
    C = np.fft.irfft(X * X, n=L_fft)[:L]
    return C

def _objective(x):
    """Compute max(|autoconv|) / (sum x)^2."""
    s = np.sum(x)
    if abs(s) < 1e-10:
        return 1e10
    ac = _autoconv_fft(x)
    return np.max(np.abs(ac)) / (s * s)

def _psd_objective_and_grad(params, beta=100.0):
    """
    Objective and gradient for optimizing the Power Spectral Density (PSD).
    params: log-PSD values for k=0..N/2.
    P[k] = exp(params[k]).
    P is extended to full N with symmetry P[N-k] = P[k].
    c[n] = IFFT(P_full)[n].
    Objective: SoftMax(|c|) / P[0].
    """
    half = N // 2
    P_half = np.exp(params)
    
    # Construct full P (length N)
    P_full = np.zeros(N)
    P_full[:half+1] = P_half
    P_full[half+1:] = P_half[:half-1][::-1]
    
    # Calculate c = IFFT(P_full)
    c = np.fft.ifft(P_full).real
    
    s_sq = P_full[0]
    if s_sq < 1e-10:
        return 1e10, np.zeros_like(params)
    
    abs_c = np.abs(c)
    max_c = np.max(abs_c)
    
    if max_c < 1e-15:
        return 1e10, np.zeros_like(params)
    
    # Soft-max weights
    shifted = beta * (abs_c - max_c)
    exp_shifted = np.exp(shifted)
    sum_exp = np.sum(exp_shifted)
    weights = exp_shifted / sum_exp
    soft_max = max_c + (1.0 / beta) * np.log(sum_exp)
    
    val = soft_max / s_sq
    
    # Gradient with respect to params
    # d(c)/d(P[k]) = (1/N) * exp(-2j*pi*k*n/N) for all n
    # d(soft_max)/d(P[k]) = sum_n weights[n] * sign(c[n]) * (1/N) * exp(-2j*pi*k*n/N)
    # d(val)/d(params[k]) = d(val)/d(P[k]) * P[k]
    
    # Vectorized gradient computation
    n = np.arange(N)
    k = np.arange(half + 1)
    
    # exp(-2j*pi*k*n/N) for all k, n
    phase = np.exp(-2j * np.pi * np.outer(k, n) / N)
    
    # d(soft_max)/d(P[k]) = sum_n weights[n] * sign(c[n]) * phase[k,n] / N
    sign_c = np.sign(c)
    weighted_sign = weights * sign_c / N
    d_soft_dP = np.real(phase @ weighted_sign)
    
    # d(val)/d(P[k]) = d_soft_dP[k] / s_sq - 2 * soft_max * (P_full[0] if k==0 else 0) / s_sq^2
    # Actually s_sq = P_full[0] = P[0], so d(s_sq)/d(P[0]) = 1, d(s_sq)/d(P[k]) = 0 for k>0
    d_val_dP = d_soft_dP / s_sq
    d_val_dP[0] -= 2.0 * soft_max / (s_sq * s_sq)
    
    # Chain rule: d(val)/d(params[k]) = d_val_dP[k] * P_half[k]
    grad = d_val_dP[:half+1] * P_half
    
    return val, grad

def _recover_x_from_psd(params):
    """Recover real signal x from log-PSD params."""
    half = N // 2
    P_half = np.exp(params)
    
    P_full = np.zeros(N)
    P_full[:half+1] = P_half
    P_full[half+1:] = P_half[:half-1][::-1]
    
    c = np.fft.ifft(P_full).real
    return c

def _peak_targeted_polish(x, max_iters=100, beta=200.0):
    """
    Vectorized peak-targeted polish: identify peaks in |autoconv| and 
    apply targeted corrections to reduce them.
    """
    best_x = x.copy()
    best_val = _objective(best_x)
    
    for it in range(max_iters):
        if _check_deadline():
            break
        
        ac = _autoconv_fft(best_x)
        abs_ac = np.abs(ac)
        max_ac = np.max(abs_ac)
        
        if max_ac < 1e-15:
            break
        
        # Find top peaks
        top_k = min(10, len(ac))
        peak_indices = np.argsort(abs_ac)[-top_k:]
        
        # For each peak, compute the direction to reduce it
        # ac[k] = sum_j x[j]*x[k-j]
        # To reduce |ac[k]|, we want to change x in a direction that reduces ac[k]
        
        # Gradient of ac[k] w.r.t. x: d(ac[k])/dx[i] = 2*x[k-i] for valid i
        # Direction to minimize |ac[k]|: -sign(ac[k]) * grad_ac_k
        
        s = np.sum(best_x)
        s_sq = s * s
        
        # Compute combined gradient from top peaks
        grad = np.zeros(N)
        for k_idx in peak_indices:
            k = k_idx
            sign_ac = np.sign(ac[k])
            valid_i = np.arange(max(0, k - N + 1), min(k + 1, N))
            if len(valid_i) > 0:
                grad[valid_i] -= sign_ac * 2.0 * best_x[k - valid_i]
        
        grad_norm = np.linalg.norm(grad)
        if grad_norm < 1e-10:
            break
        
        # Normalize and scale
        direction = -grad / grad_norm
        
        # Adaptive step size
        for lr in [0.05, 0.02, 0.01, 0.005]:
            trial = best_x + lr * direction
            trial_val = _objective(trial)
            if trial_val < best_val:
                best_x = trial.copy()
                best_val = trial_val
                break
        
        if best_val >= _objective(best_x):
            break
    
    return best_x, best_val

def _spectral_design(rng, n_candidates=20):
    """
    Generate spectral designs by optimizing PSD in log-space.
    """
    half = N // 2
    best_x = None
    best_val = 1e10
    
    for i in range(n_candidates):
        if _check_deadline():
            break
        
        # Initialize log-PSD
        if i == 0:
            # Flat spectrum (white noise)
            log_P = np.zeros(half + 1)
        elif i == 1:
            # Low-pass
            freqs = np.arange(half + 1)
            log_P = -freqs * 0.05
        elif i == 2:
            # Band-pass
            freqs = np.arange(half + 1)
            center = half // 3
            log_P = -np.abs(freqs - center) * 0.03
        else:
            # Random smooth spectrum
            log_P = np.cumsum(rng.randn(half + 1)) * 0.1
            log_P -= np.mean(log_P)
        
        # Optimize using L-BFGS
        res = minimize(
            _psd_objective_and_grad,
            log_P,
            jac=True,
            method='L-BFGS-B',
            options={'maxiter': 50, 'ftol': 1e-8, 'gtol': 1e-6}
        )
        
        if res.success:
            x = _recover_x_from_psd(res.x)
            val = _objective(x)
            if val < best_val:
                best_val = val
                best_x = x.copy()
    
    return best_x, best_val

def construct():
    rng = np.random.RandomState(42)
    
    # Phase 1: Spectral design
    x_spec, val_spec = _spectral_design(rng, n_candidates=10)
    
    if x_spec is None:
        x_spec = np.ones(N)
        val_spec = _objective(x_spec)
    
    # Phase 2: Peak-targeted polish from spectral result
    x_polished, val_polished = _peak_targeted_polish(x_spec, max_iters=50)
    
    # Phase 3: Try additional starting points
    candidates = [x_polished]
    
    # Add some structured initializations
    for sparsity in [0.3, 0.5]:
        n_sparse = int(N * sparsity)
        idx = rng.choice(N, size=n_sparse, replace=False)
        x = np.zeros(N)
        x[idx] = 1.0
        x += 0.05 * rng.randn(N)
        candidates.append(x)
    
    # Block patterns
    for block_size in [10, 20]:
        n_blocks = N // block_size
        x = np.zeros(N)
        for b in range(n_blocks):
            x[b*block_size:(b+1)*block_size] = 1.0 if b % 2 == 0 else -0.5
        candidates.append(x)
    
    # Evaluate and polish each candidate
    best_x = x_polished.copy()
    best_val = val_polished
    
    for x0 in candidates[1:]:
        if _check_deadline():
            break
        
        val0 = _objective(x0)
        if val0 < best_val:
            best_val = val0
            best_x = x0.copy()
        
        # Polish
        x_p, val_p = _peak_targeted_polish(x0, max_iters=30)
        if val_p < best_val:
            best_val = val_p
            best_x = x_p.copy()
    
    # Phase 4: Final refinement with higher beta
    if not _check_deadline():
        x_final, val_final = _peak_targeted_polish(best_x, max_iters=50, beta=500.0)
        if val_final < best_val:
            best_x = x_final.copy()
            best_val = val_final
    
    return best_x
# EVOLVE-BLOCK-END
