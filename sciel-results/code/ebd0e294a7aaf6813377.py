"""
Spectral Phase Optimization for Signed Autocorrelation Minimization.

Strategy:
1. Represent the signal x in the frequency domain: X = FFT(x).
2. Fix the magnitude spectrum |X(k)| to be constant (or a smooth taper) across the
   available frequencies. This maximizes the "spread" of energy in frequency, which
   generally helps reduce time-domain peaks in the autoconvolution (by the uncertainty
   principle / spreading effect).
3. Optimize the phase spectrum phi(k) using gradient-based methods (L-BFGS-B) to
   minimize a smooth approximation of the metric:
       metric = max_k |c_k| / (sum x)^2
   where c = x * x (autoconvolution).
   We use a soft-max function to make the max differentiable.
4. Since the metric is scale-invariant (max|c| scales as A^2, (sum x)^2 scales as A^2),
   we can normalize the signal to have unit sum during optimization, or handle the
   ratio directly.
5. After finding a good phase, convert back to time domain, and perform a final
   local search in the time domain using the exact metric to polish the result.
"""
import numpy as np
import time
from scipy.optimize import minimize

N = 400

# Next power of 2 >= 2*N - 1 for efficient FFT
_FFT_SIZE = 1
while _FFT_SIZE < 2 * N - 1:
    _FFT_SIZE *= 2

# Number of frequency bins for rfft
_N_FREQ = _FFT_SIZE // 2 + 1

# Precompute indices
_k_indices = np.arange(_N_FREQ)

def _autoconv_max_abs(x):
    """Compute max |autoconv(x)|."""
    fx = np.fft.rfft(x, _FFT_SIZE)
    conv = np.fft.irfft(fx * fx, _FFT_SIZE)
    return float(np.max(np.abs(conv[:2 * N - 1])))

def _integral(x):
    return float(np.sum(x))

def _metric(x):
    s = _integral(x)
    if abs(s) < 1e-12:
        return float('inf')
    return _autoconv_max_abs(x) / (s * s)

def _soft_max_abs(conv, beta=50.0):
    """
    Smooth approximation of max |conv|.
    Uses log-sum-exp trick on |conv|^2 to avoid issues with max.
    soft_max |y| = (1/beta) * log(sum(exp(beta * |y|)))
    But for numerical stability, we use:
    soft_max |y| = (1/beta) * log(sum(exp(beta * |y| - max|y|))) + max|y|
    Actually, simpler: use (sum(|y|^p))^(1/p) as p -> inf.
    Let's use p = beta.
    """
    abs_conv = np.abs(conv)
    # To avoid overflow, subtract max
    max_val = np.max(abs_conv)
    if max_val < 1e-12:
        return 0.0
    # Use log-sum-exp for stability
    # log(sum(exp(beta * x))) = beta * max(x) + log(sum(exp(beta * (x - max(x)))))
    # But we want (sum(x^p))^(1/p)
    # Let's stick to log-sum-exp of abs_conv directly for max approximation.
    # max(x) ~ (1/beta) * log(sum(exp(beta * x)))
    scaled = beta * abs_conv
    max_scaled = np.max(scaled)
    if max_scaled > 700: # overflow protection
        scaled = scaled - max_scaled
    log_sum_exp = max_scaled + np.log(np.sum(np.exp(scaled)))
    return log_sum_exp / beta

def _smooth_metric(x, beta=50.0):
    """
    Smooth version of the metric for gradient optimization.
    metric = max|c| / (sum x)^2
    We approximate max|c| with soft_max.
    """
    fx = np.fft.rfft(x, _FFT_SIZE)
    conv = np.fft.irfft(fx * fx, _FFT_SIZE)
    conv = conv[:2 * N - 1]
    max_abs = _soft_max_abs(conv, beta)
    s = np.sum(x)
    if abs(s) < 1e-12:
        return 1e10
    return max_abs / (s * s)

def _phase_to_signal(phase, mag_template):
    """
    Reconstruct signal from phase and fixed magnitude template.
    phase: array of shape (_N_FREQ,)
    mag_template: array of shape (_N_FREQ,)
    Returns x (real, length N).
    """
    fx = mag_template * np.exp(1j * phase)
    # Ensure Hermitian symmetry is handled by irfft
    x = np.fft.irfft(fx, _FFT_SIZE)[:N]
    return x

def _phase_objective(params, mag_template, beta):
    """
    Objective function for phase optimization.
    params: phase values
    """
    x = _phase_to_signal(params, mag_template)
    return _smooth_metric(x, beta)

def _try_phase_optimization(mag_template, deadline, rng):
    """
    Optimize phase for a given magnitude template.
    """
    best_x = np.zeros(N)
    best_val = float('inf')
    
    # Initial phase: random or zero
    # Random phase often works better to escape local minima
    phase0 = rng.uniform(-np.pi, np.pi, size=_N_FREQ)
    
    # Use L-BFGS-B
    # Bounds for phase: [-pi, pi]
    bounds = [(-np.pi, np.pi)] * _N_FREQ
    
    try:
        res = minimize(
            _phase_objective,
            phase0,
            args=(mag_template, 50.0),
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': 200, 'ftol': 1e-8, 'gtol': 1e-5}
        )
        if res.success or res.fun < best_val:
            phase_opt = res.x
            x_opt = _phase_to_signal(phase_opt, mag_template)
            val = _metric(x_opt)
            if val < best_val:
                best_x = x_opt
                best_val = val
    except Exception:
        pass
    
    # Try a few random restarts if time permits
    start_time = time.time()
    n_restarts = 5
    for i in range(n_restarts):
        if time.time() > deadline - 10:
            break
        phase0 = rng.uniform(-np.pi, np.pi, size=_N_FREQ)
        try:
            res = minimize(
                _phase_objective,
                phase0,
                args=(mag_template, 100.0), # Higher beta for sharper max
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 100, 'ftol': 1e-8}
            )
            if res.fun < best_val:
                phase_opt = res.x
                x_opt = _phase_to_signal(phase_opt, mag_template)
                val = _metric(x_opt)
                if val < best_val:
                    best_x = x_opt
                    best_val = val
        except Exception:
            pass
            
    return best_x, best_val

def _local_search_polish(x, deadline, rng):
    """
    Fast local search in time domain using exact metric.
    """
    best_x = x.copy()
    best_val = _metric(best_x)
    no_improve = 0
    step_scale = 0.1
    
    for round_idx in range(20000):
        if time.time() > deadline:
            break
        
        if no_improve > 50:
            step_scale *= 0.9
            no_improve = 0
            if step_scale < 0.001:
                break
        
        # Modify 1-3 random indices
        k = int(rng.integers(1, 4))
        indices = rng.choice(N, size=k, replace=False)
        
        best_delta = 0
        best_x_new = None
        for trial in range(3):
            x_test = best_x.copy()
            for idx in indices:
                x_test[idx] += rng.normal(0, step_scale)
            v = _metric(x_test)
            if v < best_val + best_delta:
                best_delta = v - best_val
                best_x_new = x_test
        
        if best_x_new is not None and best_delta < 0:
            best_val += best_delta
            best_x = best_x_new
            no_improve = 0
        else:
            no_improve += 1
            
    return best_x, best_val

def construct():
    deadline = time.time() + 145.0
    rng = np.random.default_rng(2024)
    
    # Initialize best result
    best_x = np.ones(N)
    best_val = _metric(best_x)
    
    # Try different magnitude templates
    # 1. Constant magnitude (flat spectrum)
    mag_flat = np.ones(_N_FREQ)
    mag_flat[0] = 0.0 # Remove DC to avoid bias? Or keep it? 
    # Actually, if we remove DC, sum(x) might be small. 
    # The metric is max|c| / sum(x)^2. 
    # If sum(x) is small, the metric is large. 
    # So we want sum(x) to be non-zero. 
    # Let's keep DC but maybe reduce it.
    mag_flat[0] = 0.1
    
    # 2. Tapered magnitude (Hamming window in freq)
    k_norm = _k_indices / (_N_FREQ - 1)
    mag_taper = 0.54 - 0.46 * np.cos(2 * np.pi * k_norm)
    mag_taper[0] = 0.1
    
    # 3. Random smooth magnitude
    mag_rand = np.exp(rng.normal(0, 0.5, size=_N_FREQ))
    mag_rand[0] = 0.1
    
    templates = [mag_flat, mag_taper, mag_rand]
    
    for mag in templates:
        if time.time() > deadline - 20:
            break
        x_candidate, val_candidate = _try_phase_optimization(mag, deadline - 20, rng)
        if val_candidate < best_val:
            best_x = x_candidate
            best_val = val_candidate
            
    # Final local search polish
    if time.time() < deadline - 2:
        best_x, best_val = _local_search_polish(best_x, deadline - 2, rng)
        
    return best_x
