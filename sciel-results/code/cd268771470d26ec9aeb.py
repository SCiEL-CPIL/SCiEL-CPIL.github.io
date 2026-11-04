"""
Strategy: Subgradient Minimax Optimization for Signed Autocorrelation.

The problem is to minimize C3' = max_k |conv[k]| / (sum x)^2.
Due to homogeneity, we fix sum(x) = 1 and minimize f(x) = max_k |x^T A_k x|.
This is a convex minimization problem (max of convex quadratic forms).
We use a Projecting Subgradient Method:
1. Compute current x.
2. Find the lag k* that maximizes |conv[k*]|.
3. Compute the subgradient g of f at x, which is +/- 2 * (A_{k*} x) (symmetrized).
4. Update x = x - step * g, then project back to the hyperplane sum(x)=1.
5. Use a decreasing step size schedule.

This approach is structurally different from the seed's alternating minimax 
(which only adjusts for the single worst lag with fixed step sizes) and 
local search (which is stochastic). It provides a global descent direction 
based on the convex geometry of the objective.
"""

import numpy as np
import time

N = 400

# Precompute FFT size
_FFT_SIZE = 1
while _FFT_SIZE < 2 * N - 1:
    _FFT_SIZE *= 2

def _autoconv_max_abs(x):
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

def _autoconv_values(x):
    fx = np.fft.rfft(x, _FFT_SIZE)
    conv = np.fft.irfft(fx * fx, _FFT_SIZE)
    return conv[:2 * N - 1]

def _subgradient_minimax(x_init, deadline, rng):
    """
    Minimize f(x) = max_k |x^T A_k x| subject to sum(x) = 1.
    Uses projected subgradient descent.
    """
    x = x_init.copy()
    s = np.sum(x)
    if abs(s) < 1e-12:
        x = np.ones(N) / N
    else:
        x = x / s
    
    best_x = x.copy()
    best_val = _metric(best_x)
    
    # Step size parameters
    # f(x) is convex. We use a standard diminishing step size.
    # Since the constraint is a hyperplane, projection is O(N).
    # We need to balance convergence speed against the 150s limit.
    
    # Initialize step size large enough to make progress, then decay.
    # The gradient magnitude scales with ||x||^2 ~ 1/N. 
    # We want step * ||g|| ~ 1/sqrt(t).
    
    # Let's use a heuristic step size that adapts to the current function value.
    # Or simply a fixed schedule: c / sqrt(t + 1).
    # c needs to be tuned. ||g|| is roughly 2 * ||x||_inf * N? No.
    # g = 2 * (A_k x). Elements of A_k x are sums of x.
    # Let's estimate ||g||.
    
    conv = _autoconv_values(x)
    abs_conv = np.abs(conv)
    k_star = int(np.argmax(abs_conv))
    
    # Compute subgradient
    # conv[k] = sum_j x[j] x[j+k] (for k>=0)
    # d conv[k] / d x[i] = x[i+k] (if i+k < N) + x[i-k] (if i-k >= 0)
    # Let's compute the gradient vector g directly.
    
    def _get_subgrad(x, k):
        g = np.zeros(N)
        if k == 0:
            g = 2.0 * x
        else:
            # k > 0
            # d/dx[i] sum_j x[j]x[j+k]
            # Term 1: j=i -> x[i+k] (if i+k < N)
            # Term 2: j+k=i -> x[i-k] (if i-k >= 0)
            g[:N-k] += x[k:]  # x[i+k] for i < N-k
            g[k:] += x[:N-k]  # x[i-k] for i >= k
        return g
    
    g = _get_subgrad(x, abs(k_star))
    # Sign of the subgradient: we want to decrease |conv[k_star]|.
    # If conv[k_star] > 0, we decrease conv -> -g
    # If conv[k_star] < 0, we increase conv -> +g
    sign = np.sign(conv[k_star])
    g = -sign * g
    
    # Normalize gradient to unit norm for stable step sizing?
    # Or use raw gradient with adaptive step.
    g_norm = np.linalg.norm(g)
    if g_norm < 1e-15:
        return best_x, best_val
    g = g / g_norm
    
    # Step size
    # Let's try a few different step size schedules.
    # Schedule 1: constant step with projection.
    # Schedule 2: decaying step.
    
    # We will use a simple decaying step: alpha_t = C / (t + 1)
    # C is chosen based on initial gradient magnitude and function value.
    # f(x) <= t. We want to reduce t.
    
    # Let's use a more robust approach: Line search or Backtracking.
    # But line search is expensive.
    # Let's use a fixed step size that is large enough to make progress 
    # but small enough to not oscillate.
    
    # Initial step size
    alpha = 0.05
    # Decay rate
    decay = 0.999
    
    current_val = best_val
    no_improve_count = 0
    
    for t in range(200000):
        if time.time() > deadline:
            break
        
        # Update x
        x_new = x - alpha * g
        
        # Project to sum(x) = 1
        # x_proj = x_new - (sum(x_new) - 1) / N
        x_new -= (np.sum(x_new) - 1.0) / N
        
        # Evaluate
        conv_new = _autoconv_values(x_new)
        abs_conv_new = np.abs(conv_new)
        k_new = int(np.argmax(abs_conv_new))
        val_new = abs_conv_new[k_new]
        
        if val_new < current_val:
            current_val = val_new
            if val_new < best_val:
                best_val = val_new
                best_x = x_new.copy()
            no_improve_count = 0
            x = x_new
        else:
            no_improve_count += 1
            # If no improvement, reduce step size more aggressively
            if no_improve_count > 50:
                alpha *= 0.9
                no_improve_count = 0
                # If step size becomes too small, restart with larger step?
                if alpha < 1e-6:
                    alpha = 0.05
                    # Maybe randomize slightly?
                    # For now, just break if stuck too long
                    if no_improve_count > 500:
                        break
        
        # Recompute subgradient for next step
        g = _get_subgrad(x, abs(k_new))
        sign_new = np.sign(conv_new[k_new])
        g = -sign_new * g
        g_norm = np.linalg.norm(g)
        if g_norm < 1e-15:
            break
        g = g / g_norm
        
        # Decay step size
        alpha *= decay
        
    return best_x, best_val

def _polish_local_search(x, deadline, rng):
    """
    Fast local search to polish the result from subgradient method.
    """
    best_x = x.copy()
    best_val = _metric(best_x)
    
    # Normalize
    s = np.sum(best_x)
    if abs(s) > 1e-12:
        best_x = best_x / s
    
    no_improve = 0
    step_scale = 0.05
    
    for round_idx in range(50000):
        if time.time() > deadline:
            break
        
        if no_improve > 100:
            step_scale *= 0.8
            no_improve = 0
            if step_scale < 0.001:
                break
        
        # Pick a random index
        idx = int(rng.integers(0, N))
        
        # Try a small perturbation
        best_delta = 0
        best_v = best_val
        
        # Try a few random deltas
        for _ in range(5):
            delta = rng.normal(0, step_scale)
            x_test = best_x.copy()
            x_test[idx] += delta
            # Re-normalize
            s_test = np.sum(x_test)
            if abs(s_test) > 1e-12:
                x_test = x_test / s_test
            v_test = _metric(x_test)
            if v_test < best_v:
                best_v = v_test
                best_delta = delta
        
        if best_delta != 0:
            best_x[idx] += best_delta
            s_test = np.sum(best_x)
            if abs(s_test) > 1e-12:
                best_x = best_x / s_test
            best_val = best_v
            no_improve = 0
        else:
            no_improve += 1
            
    return best_x, best_val

def construct():
    deadline = time.time() + 145.0
    
    # Initial guess: uniform signal
    # For uniform signal x = 1/N, conv[k] = (N-k)/N^2.
    # max conv = 1/N. metric = 1/N / 1 = 1/N = 0.0025.
    # Wait, metric is max_abs_conv / (sum x)^2.
    # sum x = 1.
    # conv[0] = sum x^2 = N * (1/N)^2 = 1/N.
    # conv[N-1] = 1/N^2.
    # Max is conv[0] = 1/N = 0.0025.
    
    # But we can do better by making the spectrum flat.
    # A signal with flat spectrum has conv[k] ~ 0 for k != 0.
    # But conv[0] is always sum x^2.
    # By Cauchy-Schwarz, sum x^2 >= (sum x)^2 / N = 1/N.
    # So the minimum possible value of conv[0] is 1/N.
    # Thus the minimum possible metric is at least 1/N.
    # If we can make all other conv[k] zero, we achieve 1/N.
    # This is a "Barker code" or "Golay sequence" like property, but for real numbers.
    # For real numbers, perfect autocorrelation is impossible for N > 2.
    # So we minimize the max of |conv[k]| for k != 0, while keeping conv[0] minimal.
    
    # Actually, the metric is max_k |conv[k]|.
    # conv[0] is part of the max.
    # So we want to minimize max( sum x^2, max_{k!=0} |conv[k]| ).
    
    # Start with a random signal or a structured one.
    rng = np.random.default_rng(2024)
    
    # Try a few initializations
    inits = [
        np.ones(N),
        rng.normal(size=N),
        np.cos(np.linspace(0, 2*np.pi, N, endpoint=False)),
    ]
    
    best_x = None
    best_val = float('inf')
    
    for init in inits:
        s = np.sum(init)
        if abs(s) > 1e-12:
            x_norm = init / s
        else:
            x_norm = init
            x_norm -= np.mean(x_norm)
            s = np.sum(x_norm)
            if abs(s) < 1e-12:
                continue
            x_norm = x_norm / s
        
        v = _metric(x_norm)
        if v < best_val:
            best_val = v
            best_x = x_norm.copy()
            
    if best_x is None:
        return np.ones(N)
        
    # Phase 1: Subgradient Minimax
    if time.time() < deadline - 30:
        x_sub, val_sub = _subgradient_minimax(best_x, deadline - 30, rng)
        if val_sub < best_val:
            best_val = val_sub
            best_x = x_sub
    
    # Phase 2: Local Search Polish
    if time.time() < deadline - 5:
        x_pol, val_pol = _polish_local_search(best_x, deadline - 5, rng)
        if val_pol < best_val:
            best_val = val_pol
            best_x = x_pol
            
    return best_x

# EVOLVE-BLOCK-END
