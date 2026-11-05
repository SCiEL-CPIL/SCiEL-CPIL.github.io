"""Non-harmonic band-limited signal optimization for autocorrelation minimization.

Strategy:
- Represent h as a sum of complex sinusoids with arbitrary (non-integer) frequencies.
- Use smooth-max surrogate (log-sum-exp) for the max of the autocorrelation.
- Optimize frequencies, amplitudes, phases, and DC offset via L-BFGS-B.
- Multiple restarts with different random initializations.
- Final high-precision polish with tighter alpha.
"""
import time
import numpy as np
from scipy.optimize import minimize

N = 400
_t0 = time.time()
_DEADLINE = 135.0  # safety margin under 150s wall-clock

def _value(h):
    """Compute the task value (lower is better)."""
    h = np.asarray(h, dtype=float)
    s = float(np.sum(h))
    if s == 0.0 or not np.all(np.isfinite(h)):
        return np.inf
    conv = np.convolve(h, h)
    m = float(np.max(conv))
    return abs(2.0 * N * m / (s * s))

def _smooth_max_value(h, alpha=50.0):
    """Smooth surrogate for the max of convolution using log-sum-exp with FFT."""
    h = np.asarray(h, dtype=float)
    s = float(np.sum(h))
    if s == 0.0 or not np.all(np.isfinite(h)):
        return np.inf
    n_fft = 2 * N
    H = np.fft.rfft(h, n=n_fft)
    conv = np.fft.irfft(H * H, n=n_fft)[:2 * N - 1]
    scaled = conv * alpha
    max_scaled = np.max(scaled)
    log_sum_exp = max_scaled + np.log(np.sum(np.exp(scaled - max_scaled)))
    soft_max = log_sum_exp / alpha
    return abs(2.0 * N * soft_max / (s * s))

# Precompute time grid
_n_idx = np.arange(N, dtype=float)

def _nonharmonic_h(params):
    """
    Generate h from non-harmonic sinusoid parameters.
    params: [freq1, freq2, ..., freqK, amp1, amp2, ..., ampK, phase1, phase2, ..., phaseK, dc]
    Total: 4*K + 1 parameters
    """
    K = len(params) // 4
    freqs = params[0:K]
    amps = params[K:2*K]
    phases = params[2*K:3*K]
    dc = params[3*K]
    
    # h[n] = dc + sum_k amps[k] * cos(2*pi*freqs[k]*n/N + phases[k])
    h = np.full(N, dc, dtype=float)
    for k in range(K):
        h += amps[k] * np.cos(2.0 * np.pi * freqs[k] * _n_idx / N + phases[k])
    return h

def _nonharmonic_smooth_obj(params, alpha=50.0):
    h = _nonharmonic_h(params)
    return _smooth_max_value(h, alpha=alpha)

def _nonharmonic_obj(params):
    h = _nonharmonic_h(params)
    return _value(h)

def construct():
    global _t0
    _t0 = time.time()
    
    best_h = np.ones(N)
    best_v = _value(best_h)
    
    def deadline_ok():
        return (time.time() - _t0) < _DEADLINE
    
    # Number of sinusoidal components
    K = 12
    
    # Random seeds for reproducibility
    rng = np.random.RandomState(42)
    
    # Multiple restarts with different initializations
    n_restarts = 8
    
    for restart in range(n_restarts):
        if not deadline_ok():
            break
        
        # Initialize parameters
        # Frequencies: spread across [0.5, 20] (non-integer)
        freqs_init = rng.uniform(0.5, 20.0, size=K)
        # Amplitudes: random positive
        amps_init = rng.uniform(0.1, 1.0, size=K)
        # Phases: random
        phases_init = rng.uniform(0, 2*np.pi, size=K)
        # DC: small
        dc_init = rng.uniform(0.5, 2.0)
        
        x0 = np.concatenate([freqs_init, amps_init, phases_init, [dc_init]])
        
        # Use smooth-max surrogate for optimization
        try:
            res = minimize(
                _nonharmonic_smooth_obj,
                x0,
                args=(50.0,),
                method='L-BFGS-B',
                options={'maxiter': 300, 'ftol': 1e-12, 'gtol': 1e-10},
            )
            if res.x is not None:
                h = _nonharmonic_h(res.x)
                v = _value(h)
                if v < best_v:
                    best_v = v
                    best_h = h.copy()
        except Exception:
            pass
        
        if not deadline_ok():
            break
    
    # Final high-precision polish with tighter alpha
    if deadline_ok():
        try:
            # Re-initialize from best_h using a simple projection
            # For simplicity, just run another L-BFGS-B with tighter settings
            h_best = best_h
            dc_init = float(np.mean(h_best))
            
            # Estimate frequencies from FFT of best_h
            H_fft = np.fft.rfft(h_best - dc_init)
            # Find dominant frequency components
            magnitudes = np.abs(H_fft)
            # Get top K frequency indices
            top_indices = np.argsort(magnitudes)[-K:][::-1]
            freqs_init = top_indices.astype(float)
            
            # Estimate amplitudes and phases
            amps_init = np.zeros(K)
            phases_init = np.zeros(K)
            for i, idx in enumerate(top_indices):
                if idx == 0:
                    continue
                H_k = H_fft[idx]
                amps_init[i] = 2.0 * np.abs(H_k) / N
                phases_init[i] = -np.angle(H_k)
            
            x0 = np.concatenate([freqs_init, amps_init, phases_init, [dc_init]])
            
            res = minimize(
                _nonharmonic_smooth_obj,
                x0,
                args=(200.0,),
                method='L-BFGS-B',
                options={'maxiter': 500, 'ftol': 1e-14, 'gtol': 1e-12},
            )
            if res.x is not None:
                h = _nonharmonic_h(res.x)
                v = _value(h)
                if v < best_v:
                    best_v = v
                    best_h = h.copy()
        except Exception:
            pass
    
    # Final exact-value Nelder-Mead polish
    if deadline_ok():
        try:
            h_best = best_h
            dc_init = float(np.mean(h_best))
            
            H_fft = np.fft.rfft(h_best - dc_init)
            magnitudes = np.abs(H_fft)
            top_indices = np.argsort(magnitudes)[-K:][::-1]
            freqs_init = top_indices.astype(float)
            
            amps_init = np.zeros(K)
            phases_init = np.zeros(K)
            for i, idx in enumerate(top_indices):
                if idx == 0:
                    continue
                H_k = H_fft[idx]
                amps_init[i] = 2.0 * np.abs(H_k) / N
                phases_init[i] = -np.angle(H_k)
            
            x0 = np.concatenate([freqs_init, amps_init, phases_init, [dc_init]])
            
            res = minimize(
                _nonharmonic_obj,
                x0,
                method='Nelder-Mead',
                options={'xatol': 1e-12, 'fatol': 1e-14, 'maxiter': 500, 'maxfev': 1000},
            )
            if res.x is not None:
                h = _nonharmonic_h(res.x)
                v = _value(h)
                if v < best_v:
                    best_v = v
                    best_h = h.copy()
        except Exception:
            pass
    
    # Final validation
    s = np.sum(best_h)
    if s == 0 or not np.all(np.isfinite(best_h)):
        return np.ones(N)
    if abs(s) < 1e-10:
        return np.ones(N)
    return best_h
