"""
FFT-based overlap computation with multi-start L-BFGS-B optimization.
Uses smooth surrogate (log-sum-exp) for the max-overlap objective.
Falls back to constant h=0.5 if optimization fails or times out.
"""
import numpy as np
import time
from scipy.optimize import minimize

N = 100
DEADLINE = 140.0  # safety margin before 150s kill

def _overlap_fft(h):
    """Compute overlap O(k) = sum_i h[i] * (1 - h[(i+k)%N]) for all shifts k."""
    h1 = 1.0 - h
    # Cross-correlation via FFT: O(k) = sum_i h[i] * h1[(i+k)%N]
    # This is the circular cross-correlation of h and h1
    H = np.fft.fft(h)
    H1 = np.fft.fft(h1)
    # Cross-correlation: ifft(fft(h) * conj(fft(h1)))
    corr = np.real(np.fft.ifft(H * np.conj(H1)))
    return corr

def _max_overlap(h):
    """Maximum overlap value."""
    return np.max(_overlap_fft(h))

def _objective_and_grad(x, beta=50.0):
    """
    Smooth surrogate: log-sum-exp of overlaps / beta + penalty for constraint violation.
    x is the raw parameter vector; h = sigmoid(x) to ensure 0 < h < 1.
    """
    h = 1.0 / (1.0 + np.exp(-x))
    # Clip to avoid numerical issues
    h = np.clip(h, 1e-10, 1.0 - 1e-10)
    
    # Compute overlaps
    H = np.fft.fft(h)
    H1 = np.fft.fft(1.0 - h)
    corr = np.real(np.fft.ifft(H * np.conj(H1)))
    
    # Log-sum-exp for smooth max
    max_corr = np.max(corr)
    if beta > 0:
        # Soft max: (1/beta) * log(sum(exp(beta * corr)))
        shifted = beta * (corr - max_corr)
        log_sum_exp = max_corr + np.log(np.sum(np.exp(shifted))) / beta
    else:
        log_sum_exp = max_corr
    
    # Constraint: sum(h) should be N/2
    sum_h = np.sum(h)
    target = N / 2.0
    constraint_viol = (sum_h - target) ** 2
    penalty = 100.0 * constraint_viol
    
    val = log_sum_exp + penalty
    
    # Gradient via chain rule
    # d(val)/d(h) = d(log_sum_exp)/d(corr) * d(corr)/d(h) + 2*100*(sum_h - target)
    # d(log_sum_exp)/d(corr_k) = exp(beta*corr_k) / sum(exp(beta*corr))
    if beta > 0:
        weights = np.exp(shifted)
        weights = weights / np.sum(weights)
    else:
        # One-hot at argmax
        weights = np.zeros(N)
        weights[np.argmax(corr)] = 1.0
    
    # d(corr)/d(h) via FFT: corr = ifft(fft(h) * conj(fft(1-h)))
    # d(corr)/d(h_i) is the circular cross-correlation of delta_i and (1-h)
    # More simply: corr_k = sum_j h[j] * (1-h[(j+k)%N])
    # d(corr_k)/d(h[i]) = (1-h[(i+k)%N]) + sum_j h[j] * d(1-h[(j+k)%N])/d(h[i])
    # = (1-h[(i+k)%N]) - h[(i-k)%N]  (for the second term, only j such that (j+k)%N = i, i.e., j=(i-k)%N)
    
    # Actually let's use a cleaner formulation:
    # corr = ifft(fft(h) * conj(fft(1-h)))
    # d(corr)/d(h) = ifft(1 * conj(fft(1-h))) + ifft(fft(h) * conj(-fft(1)))
    # Wait, this gets complicated. Let me just use finite differences for the gradient
    # since N=100 is small.
    
    # Actually, let's compute gradient more carefully.
    # corr_k = sum_j h[j] * (1 - h[(j+k) % N])
    # d(corr_k)/d(h[i]) = (1 - h[(i+k) % N]) + h[(i-k) % N] * (-1)
    #                    = (1 - h[(i+k) % N]) - h[(i-k) % N]
    
    # So d(val)/d(h[i]) = sum_k weights[k] * [(1 - h[(i+k)%N]) - h[(i-k)%N]] + 2*100*(sum_h - target)
    # Using FFT: sum_k weights[k] * (1-h[(i+k)%N]) is the circular cross-correlation of weights and (1-h)
    # And sum_k weights[k] * h[(i-k)%N] is the circular cross-correlation of weights and h
    
    W = np.fft.fft(weights)
    d_corr_dh = np.real(np.fft.ifft(W * np.conj(np.fft.fft(1.0 - h)))) - np.real(np.fft.ifft(W * np.conj(np.fft.fft(h))))
    
    grad_h = d_corr_dh + 2.0 * 100.0 * (sum_h - target)
    
    # Chain rule: d(val)/d(x) = d(val)/d(h) * d(h)/d(x) = grad_h * h * (1-h)
    grad_x = grad_h * h * (1.0 - h)
    
    return val, grad_x

def _normalize(h):
    """Ensure sum(h) = N/2 by affine adjustment, keeping h in [0,1]."""
    current_sum = np.sum(h)
    target = N / 2.0
    if abs(current_sum - target) < 1e-12:
        return h
    # Linear adjustment
    h_adj = h + (target - current_sum) / N
    # If any values go out of [0,1], clip and redistribute
    if np.any(h_adj < 0) or np.any(h_adj > 1):
        h_adj = np.clip(h_adj, 0.0, 1.0)
        # Iteratively adjust
        for _ in range(100):
            diff = target - np.sum(h_adj)
            if abs(diff) < 1e-12:
                break
            mask = (h_adj > 0) & (h_adj < 1)
            if not np.any(mask):
                break
            h_adj[mask] += diff / np.sum(mask)
            h_adj = np.clip(h_adj, 0.0, 1.0)
    return h_adj

def construct():
    start_time = time.time()
    
    # Baseline: constant 0.5
    h_best = np.full(N, 0.5)
    best_score = _max_overlap(h_best)  # Should be 0.5
    
    # Multi-start L-BFGS-B optimization
    n_starts = 5
    rng = np.random.default_rng(42)
    
    for s in range(n_starts):
        if time.time() > DEADLINE:
            break
        
        # Initialize x such that sigmoid(x) gives various starting points
        if s == 0:
            # Start near uniform with small perturbation
            x0 = np.zeros(N) + rng.normal(0, 0.1, N)
        elif s == 1:
            # Linear ramp perturbation
            x0 = np.linspace(-1, 1, N) + rng.normal(0, 0.05, N)
        elif s == 2:
            # Sine perturbation
            x0 = 0.5 * np.sin(2 * np.pi * np.arange(N) / N) + rng.normal(0, 0.1, N)
        elif s == 3:
            # Higher frequency
            x0 = 0.3 * np.sin(4 * np.pi * np.arange(N) / N) + rng.normal(0, 0.1, N)
        else:
            # Random
            x0 = rng.normal(0, 0.5, N)
        
        try:
            result = minimize(
                _objective_and_grad,
                x0,
                method='L-BFGS-B',
                jac=True,
                options={'maxiter': 200, 'ftol': 1e-10, 'gtol': 1e-8}
            )
            
            if result.success or result.fun < best_score:
                h_candidate = 1.0 / (1.0 + np.exp(-result.x))
                h_candidate = np.clip(h_candidate, 0.0, 1.0)
                h_candidate = _normalize(h_candidate)
                
                score = _max_overlap(h_candidate)
                if score < best_score:
                    best_score = score
                    h_best = h_candidate.copy()
                    
        except Exception:
            continue
        
        if time.time() > DEADLINE:
            break
    
    # Final polish: tighter optimization from best solution
    if time.time() < DEADLINE - 5.0:
        try:
            h_best_x = np.log(h_best / (1.0 - h_best))  # Inverse sigmoid
            h_best_x = np.clip(h_best_x, -10, 10)
            
            result = minimize(
                _objective_and_grad,
                h_best_x,
                method='L-BFGS-B',
                jac=True,
                options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-10}
            )
            
            if result.success:
                h_polished = 1.0 / (1.0 + np.exp(-result.x))
                h_polished = np.clip(h_polished, 0.0, 1.0)
                h_polished = _normalize(h_polished)
                
                score = _max_overlap(h_polished)
                if score < best_score:
                    best_score = score
                    h_best = h_polished.copy()
        except Exception:
            pass
    
    # Ensure constraints are met
    h_best = np.clip(h_best, 0.0, 1.0)
    h_best = _normalize(h_best)
    
    return h_best
