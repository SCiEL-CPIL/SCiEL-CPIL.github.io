"""Minimize the normalized maximum of the autoconvolution of a nonnegative
step function on 600 equal intervals over [-1/4, 1/4].

Strategy: Combine FFT-based water-filling with smooth epigraph optimization.
  1. FFT autoconvolution for O(N log N) evaluation
  2. Soft-max surrogate (log-sum-exp) for smooth gradient of the max
  3. Projected gradient descent with epigraph variable t
  4. Parametric Fourier restarts for diversity
  5. Final high-precision polish pass
"""

import time
import math

import numpy as np

N = 600
DELTA = 0.5 / N
DEADLINE = 140.0  # safety margin before 150s kill

_t0 = time.time()


def _time_left():
    return DEADLINE - (time.time() - _t0)


def _midpoints():
    x = np.linspace(-0.25, 0.25, N, endpoint=False) + DELTA / 2.0
    return x


def _autoconv_fft(h):
    """Compute autoconvolution of step heights using FFT. O(N log N)."""
    n_fft = 2 * N
    H = np.fft.rfft(h, n=n_fft)
    C = np.fft.irfft(H * H, n=n_fft)[:2*N-1]
    return C * DELTA


def _evaluate(h):
    """Compute the normalized max autoconvolution score (lower is better).
    
    We minimize: max(autoconv) / (sum(h) * delta)^2
    """
    h = np.asarray(h, dtype=float)
    h = np.clip(h, 0.0, None)
    s = float(np.sum(h))
    if s < 1e-12:
        return float('inf')
    
    c = _autoconv_fft(h)
    peak = float(np.max(c))
    
    integral = s * DELTA
    return peak / (integral * integral)


def _softmax(c, beta):
    """Smooth approximation of max(c) using log-sum-exp."""
    c_max = np.max(c)
    # Shift for numerical stability
    shifted = c - c_max
    # Clip to avoid overflow
    shifted = np.clip(shifted, -50, 50)
    return c_max + (1.0 / beta) * np.log(np.sum(np.exp(beta * shifted)))


def _softmax_grad(c, beta):
    """Gradient of soft-max w.r.t. c."""
    c_max = np.max(c)
    shifted = c - c_max
    shifted = np.clip(shifted, -50, 50)
    weights = np.exp(beta * shifted)
    weights /= np.sum(weights)
    return weights


def _grad_autoconv_peak(h, c, beta):
    """Compute gradient of soft-max autoconvolution w.r.t. h.
    
    Uses the chain rule: d(softmax(c))/dh_i = sum_k w_k * dc_k/dh_i
    where w_k are softmax weights over lags.
    """
    w = _softmax_grad(c, beta)
    
    # dc_k/dh_i = h_{i+k} * delta (if i+k < N) + h_{i-k} * delta (if i-k >= 0)
    # For efficiency, compute via convolution-like operations
    grad = np.zeros(N)
    
    # For each lag k, the contribution to grad_i is w_k * (h_{i+k} + h_{i-k}) * delta
    # We can compute this efficiently using the fact that c_k = sum_i h_i * h_{i+k} * delta
    
    # Actually, let's use a simpler approach: the subgradient of max at the peak lag
    # But for smooth version, we need weighted sum over all lags.
    
    # Efficient computation: grad_i = delta * sum_k w_k * (h_{i+k} + h_{i-k})
    # where h_j = 0 if j outside [0, N-1]
    
    # Use FFT-based approach or direct summation for small effective support
    # For simplicity and correctness, do direct computation with vectorization
    
    for k in range(len(c)):
        wk = w[k]
        if wk < 1e-10:
            continue
        if k == 0:
            grad += 2.0 * wk * h * DELTA
        else:
            if k < N:
                grad[:N-k] += wk * h[k:] * DELTA
                grad[k:] += wk * h[:N-k] * DELTA
    
    return grad


def _projected_gradient_step(h, grad, step_size, target_sum):
    """One projected gradient step with projection to h >= 0 and fixed sum."""
    h_new = h - step_size * grad
    h_new = np.clip(h_new, 0.0, None)
    
    s_new = float(np.sum(h_new))
    if s_new > 1e-12:
        h_new = h_new * (target_sum / s_new)
    else:
        h_new = np.ones(N) * (target_sum / N)
    
    return h_new


def _fourier_to_heights(params, x_mid, K=8):
    """Convert Fourier coefficients to step heights."""
    base = params[0]
    h = np.full(N, base)
    for k in range(1, K+1):
        a = params[2*k-1]
        b = params[2*k]
        h += a * np.cos(2*np.pi*k*x_mid/0.5) + b * np.sin(2*np.pi*k*x_mid/0.5)
    return np.clip(h, 0.0, None)


def construct():
    global _t0
    _t0 = time.time()
    
    x_mid = _midpoints()
    
    # Start with uniform
    best_h = np.ones(N)
    best_score = _evaluate(best_h)
    
    # Phase 1: Quick parametric family scan
    for freq in [1, 2, 3, 4, 5]:
        for amp in np.linspace(0.0, 0.6, 13):
            if _time_left() < 10.0:
                break
            h = 1.0 + amp * np.cos(2 * np.pi * freq * x_mid / 0.5)
            h = np.clip(h, 0.0, None)
            score = _evaluate(h)
            if score < best_score:
                best_score = score
                best_h = h.copy()
    
    # Phase 2: Smooth gradient optimization with soft-max surrogate
    if _time_left() > 40.0:
        h = best_h.copy()
        target_sum = float(np.sum(h))
        
        # Start with moderate beta, anneal to larger beta
        beta = 50.0
        step = 0.05
        
        for it in range(500):
            if _time_left() < 20.0:
                break
            
            c = _autoconv_fft(h)
            grad = _grad_autoconv_peak(h, c, beta)
            
            # Normalize gradient for stable step size
            grad_norm = np.linalg.norm(grad)
            if grad_norm > 1e-12:
                grad = grad / grad_norm
            
            h_new = _projected_gradient_step(h, grad, step, target_sum)
            score_new = _evaluate(h_new)
            
            if score_new < best_score:
                best_score = score_new
                best_h = h_new.copy()
                # Anneal: increase beta for sharper max, decrease step
                beta *= 1.02
                step = max(step * 0.995, 0.001)
            else:
                step = max(step * 0.95, 0.0005)
                # Occasionally restart from best
                if it % 50 == 0:
                    h = best_h.copy()
                    target_sum = float(np.sum(h))
        
        # Phase 3: Fine-tune with larger beta (sharper max approximation)
        if _time_left() > 20.0:
            h = best_h.copy()
            target_sum = float(np.sum(h))
            beta = 200.0
            step = 0.01
            
            for it in range(300):
                if _time_left() < 10.0:
                    break
                
                c = _autoconv_fft(h)
                grad = _grad_autoconv_peak(h, c, beta)
                
                grad_norm = np.linalg.norm(grad)
                if grad_norm > 1e-12:
                    grad = grad / grad_norm
                
                h_new = _projected_gradient_step(h, grad, step, target_sum)
                score_new = _evaluate(h_new)
                
                if score_new < best_score:
                    best_score = score_new
                    best_h = h_new.copy()
                    step = max(step * 0.99, 0.0001)
                else:
                    step = max(step * 0.9, 0.00005)
    
    # Phase 4: Fourier parameterization with random restarts
    if _time_left() > 30.0:
        K = 6
        np.random.seed(42)
        
        for restart in range(5):
            if _time_left() < 10.0:
                break
            
            # Random initial Fourier coefficients
            params = np.zeros(1 + 2*K)
            params[0] = 1.0
            params[1:] = np.random.uniform(-0.3, 0.3, 2*K)
            
            h = _fourier_to_heights(params, x_mid, K)
            target_sum = float(np.sum(h))
            
            # Local optimization in Fourier space
            beta = 100.0
            step = 0.02
            h_curr = h.copy()
            
            for it in range(100):
                if _time_left() < 5.0:
                    break
                
                c = _autoconv_fft(h_curr)
                grad = _grad_autoconv_peak(h_curr, c, beta)
                
                # Project gradient back to Fourier basis (simplified: use direct h-space)
                grad_norm = np.linalg.norm(grad)
                if grad_norm > 1e-12:
                    grad = grad / grad_norm
                
                h_new = _projected_gradient_step(h_curr, grad, step, target_sum)
                score_new = _evaluate(h_new)
                
                if score_new < best_score:
                    best_score = score_new
                    best_h = h_new.copy()
                    h_curr = h_new.copy()
                    step = max(step * 0.98, 0.001)
                else:
                    step = max(step * 0.9, 0.0005)
    
    # Phase 5: Final high-precision polish
    if _time_left() > 5.0:
        h = best_h.copy()
        target_sum = float(np.sum(h))
        beta = 500.0
        step = 0.001
        
        for it in range(100):
            if _time_left() < 2.0:
                break
            
            c = _autoconv_fft(h)
            grad = _grad_autoconv_peak(h, c, beta)
            
            grad_norm = np.linalg.norm(grad)
            if grad_norm > 1e-12:
                grad = grad / grad_norm
            else:
                break
            
            h_new = _projected_gradient_step(h, grad, step, target_sum)
            score_new = _evaluate(h_new)
            
            if score_new < best_score:
                best_score = score_new
                best_h = h_new.copy()
                step = max(step * 0.95, 0.00001)
            else:
                step = max(step * 0.8, 0.000005)
    
    best_h = np.clip(best_h, 0.0, None)
    
    return [float(v) for v in best_h]


_t0 = time.time()
