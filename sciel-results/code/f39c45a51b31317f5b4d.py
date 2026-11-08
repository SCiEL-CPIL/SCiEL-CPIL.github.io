"""
Optimized approach: Use FFT-based overlap computation with multi-start L-BFGS-B
to find non-uniform step heights that minimize maximum overlap below 0.5.
"""
import numpy as np
from scipy.optimize import minimize
import time

N = 100
DEADLINE = 120.0  # seconds, with safety margin from 150s kill
START_TIME = None

def _compute_overlap_fft(h):
    """Compute overlap O(k) = sum_i h[i] * (1 - h[(i+k) % N]) for all shifts k."""
    n = len(h)
    H = np.fft.fft(h)
    corr = np.real(np.fft.ifft(np.conj(H) * H))
    return 1.0 - corr

def _max_overlap(h):
    """Compute maximum overlap value."""
    overlaps = _compute_overlap_fft(h)
    return np.max(overlaps)

def _objective(params, tau):
    """Objective: soft-max approximation of max overlap."""
    h = params
    overlaps = _compute_overlap_fft(h)
    max_val = np.max(overlaps)
    shifted = tau * (overlaps - max_val)
    # Numerically stable log-sum-exp
    log_sum = max_val + (1.0/tau) * np.log(np.sum(np.exp(shifted)))
    return log_sum

def _gradient(params, tau):
    """Gradient of the soft-max objective using FFT for efficiency."""
    h = params
    overlaps = _compute_overlap_fft(h)
    max_val = np.max(overlaps)
    shifted = tau * (overlaps - max_val)
    weights = np.exp(shifted) / np.sum(np.exp(shifted))
    
    # Gradient of O(k) w.r.t. h[i]:
    # O(k) = 1 - sum_j h[j] * h[(j+k)%n]
    # dO(k)/dh[i] = -(h[(i+k)%n] + h[(i-k)%n])
    # grad_obj[i] = sum_k weights[k] * dO(k)/dh[i]
    # = -sum_k weights[k] * (h[(i+k)%n] + h[(i-k)%n])
    
    # Use FFT: sum_k weights[k] * h[(i+k)%n] is circular cross-correlation
    # Let W = fft(weights), H = fft(h)
    # sum_k weights[k] * h[(i+k)%n] = ifft(conj(W) * H)[i]
    # sum_k weights[k] * h[(i-k)%n] = ifft(W * conj(H))[i]
    
    W = np.fft.fft(weights)
    H = np.fft.fft(h)
    term1 = np.real(np.fft.ifft(np.conj(W) * H))  # sum_k weights[k] * h[(i+k)%n]
    term2 = np.real(np.fft.ifft(W * np.conj(H)))   # sum_k weights[k] * h[(i-k)%n]
    
    grad = -(term1 + term2)
    return grad

def _objective_and_grad(params, tau):
    """Combined objective and gradient."""
    val = _objective(params, tau)
    grad = _gradient(params, tau)
    return val, grad

def _normalize(h):
    """Normalize h to have sum = N/2 and values in [0,1]."""
    h = np.clip(h, 0.0, 1.0)
    s = np.sum(h)
    if s > 0:
        h = h / s * (N / 2.0)
    h = np.clip(h, 0.0, 1.0)
    return h

def construct():
    global START_TIME
    if START_TIME is None:
        START_TIME = time.time()
    
    # Start with uniform solution as baseline
    best_h = np.full(N, 0.5)
    best_score = _max_overlap(best_h)
    
    # Multi-start optimization
    rng = np.random.RandomState(42)
    n_starts = 15
    max_iter = 150
    
    for start_idx in range(n_starts):
        # Check deadline
        if time.time() - START_TIME > DEADLINE - 10:
            break
        
        # Generate initial guess: uniform + random Fourier modes
        h_init = np.full(N, 0.5)
        freqs = np.fft.fftfreq(N)
        perturbation = np.zeros(N, dtype=complex)
        for f_idx in range(1, N//2 + 1):
            if freqs[f_idx] != 0:
                perturbation[f_idx] = rng.randn() * 0.1
                perturbation[N - f_idx] = np.conj(perturbation[f_idx])
        h_init += np.real(np.fft.ifft(perturbation))
        h_init = _normalize(h_init)
        
        # Optimize with soft-max
        tau = 10.0
        try:
            result = minimize(
                lambda p: _objective(p, tau),
                h_init,
                method='L-BFGS-B',
                jac=lambda p: _gradient(p, tau),
                bounds=[(0.0, 1.0)] * N,
                options={'maxiter': max_iter, 'ftol': 1e-12, 'gtol': 1e-8}
            )
            
            h_opt = np.clip(result.x, 0.0, 1.0)
            h_opt = _normalize(h_opt)
            score = _max_overlap(h_opt)
            if score < best_score:
                best_score = score
                best_h = h_opt
        except Exception:
            continue
    
    # Final polish: use higher temperature soft-max for finer optimization
    if time.time() - START_TIME < DEADLINE - 15:
        tau_polish = 50.0
        try:
            result = minimize(
                lambda p: _objective(p, tau_polish),
                best_h,
                method='L-BFGS-B',
                jac=lambda p: _gradient(p, tau_polish),
                bounds=[(0.0, 1.0)] * N,
                options={'maxiter': 300, 'ftol': 1e-14, 'gtol': 1e-10}
            )
            h_opt = np.clip(result.x, 0.0, 1.0)
            h_opt = _normalize(h_opt)
            score = _max_overlap(h_opt)
            if score < best_score:
                best_score = score
                best_h = h_opt
        except Exception:
            pass
    
    # Second polish with even higher tau if time allows
    if time.time() - START_TIME < DEADLINE - 5:
        tau_polish2 = 100.0
        try:
            result = minimize(
                lambda p: _objective(p, tau_polish2),
                best_h,
                method='L-BFGS-B',
                jac=lambda p: _gradient(p, tau_polish2),
                bounds=[(0.0, 1.0)] * N,
                options={'maxiter': 200, 'ftol': 1e-15, 'gtol': 1e-11}
            )
            h_opt = np.clip(result.x, 0.0, 1.0)
            h_opt = _normalize(h_opt)
            score = _max_overlap(h_opt)
            if score < best_score:
                best_score = score
                best_h = h_opt
        except Exception:
            pass
    
    return best_h
