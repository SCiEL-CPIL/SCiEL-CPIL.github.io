"""
Strategy: Non-Harmonic Sinusoidal Basis Optimization with Log-Sum-Exp Surrogate.

This approach differs structurally from the current program's chirp-pair + local
random refinement in two key ways:
1. **Representation**: Instead of chirps (quadratic phase) or direct 400-dim vectors,
   we use a basis of K non-harmonic complex sinusoids: 
   h[n] = sum_{k=1}^K A_k * cos(2*pi*f_k*n/N + phi_k)
   This allows for flexible sidelobe shaping that is not restricted to polynomial
   phase or discrete DFT frequencies.
2. **Optimization**: We use L-BFGS-B with a differentiable log-sum-exp surrogate
   for the max autocorrelation term. This enables efficient gradient-based descent
   in the low-dimensional parameter space (~3K parameters) without relying on
   stochastic local search or differential evolution.
"""

import numpy as np
import time
from scipy.optimize import minimize

N = 400
_DEADLINE = 145.0  # Safety margin for 150s limit


def _evaluate(h):
    """Compute the exact task value."""
    s = np.sum(h)
    if s == 0 or not np.isfinite(s):
        return np.inf
    conv = np.convolve(h, h)
    return abs(2.0 * N * np.max(conv) / (s * s))


def _h_from_params(params):
    """
    Construct h from non-harmonic sinusoidal parameters.
    params = [f_1, phi_1, A_1, f_2, phi_2, A_2, ..., f_K, phi_K, A_K]
    """
    K = len(params) // 3
    n = np.arange(N)
    t = n / N
    h = np.zeros(N)
    for k in range(K):
        f = params[3*k]
        phi = params[3*k + 1]
        A = params[3*k + 2]
        h += A * np.cos(2 * np.pi * f * t + phi)
    return h


def _surrogate_value(params, tau):
    """
    Smooth surrogate for the task value using log-sum-exp.
    Minimizing this drives down the peak of the autocorrelation.
    """
    h = _h_from_params(params)
    s = np.sum(h)
    if abs(s) < 1e-10:
        return 1e10
    
    conv = np.convolve(h, h)
    
    # Log-sum-exp trick for numerical stability
    max_conv = np.max(conv)
    log_sum_exp = max_conv + np.log(np.sum(np.exp((conv - max_conv) / tau)))
    
    # The objective is roughly 2*N*max_conv / s^2
    # We approximate max_conv with log_sum_exp/tau for smoothness
    val = abs(2.0 * N * (log_sum_exp / tau) / (s * s))
    
    return val


def construct():
    t0 = time.time()
    deadline = t0 + _DEADLINE
    
    best_h = np.ones(N)
    best_val = _evaluate(best_h)  # Should be 2.0
    
    # Try different numbers of sinusoids K
    K_values = [5, 8, 12]
    tau_values = [0.5, 1.0, 2.0]
    
    rng = np.random.default_rng(42)
    
    for K in K_values:
        if time.time() > deadline - 30:
            break
            
        for tau in tau_values:
            if time.time() > deadline - 20:
                break
            
            # Initialize parameters randomly
            # Frequencies: spread out, avoid 0 and 0.5 (Nyquist)
            # Amplitudes: positive
            # Phases: random
            params_init = np.zeros(3 * K)
            for k in range(K):
                # Random frequencies in [0.01, 0.49]
                params_init[3*k] = rng.uniform(0.01, 0.49)
                # Random phase
                params_init[3*k + 1] = rng.uniform(0, 2 * np.pi)
                # Random amplitude
                params_init[3*k + 2] = rng.uniform(0.5, 2.0)
            
            # Define objective for L-BFGS-B
            def objective(params):
                return _surrogate_value(params, tau)
            
            # Bounds
            bounds = []
            for k in range(K):
                bounds.append((0.01, 0.49))  # f
                bounds.append((0, 2 * np.pi)) # phi
                bounds.append((0.1, 5.0))     # A
            
            try:
                # Use L-BFGS-B for gradient-based optimization
                result = minimize(
                    objective,
                    params_init,
                    method='L-BFGS-B',
                    bounds=bounds,
                    options={'maxiter': 200, 'ftol': 1e-8, 'gtol': 1e-5}
                )
                
                if result.success or result.fun < best_val:
                    h_candidate = _h_from_params(result.x)
                    v_candidate = _evaluate(h_candidate)
                    
                    if v_candidate < best_val:
                        best_val = v_candidate
                        best_h = h_candidate.copy()
                        
            except Exception:
                pass
    
    # Final local refinement on the best solution found using coordinate descent
    if time.time() < deadline - 10:
        h = best_h.copy()
        current_val = best_val
        step_size = 0.05
        rng_refine = np.random.default_rng(123)
        
        while time.time() < deadline - 5:
            improved = False
            for _ in range(50):  # Max 50 steps per epoch
                idx = rng_refine.integers(0, N)
                h_new = h.copy()
                h_new[idx] += rng_refine.normal(0, step_size)
                
                v_new = _evaluate(h_new)
                if v_new < current_val:
                    current_val = v_new
                    h = h_new
                    improved = True
                    step_size *= 0.9
                else:
                    step_size = min(step_size * 1.05, 0.1)
            
            if not improved:
                break
        
        if current_val < best_val:
            best_val = current_val
            best_h = h
    
    # Ensure validity
    if not np.all(np.isfinite(best_h)) or np.sum(best_h) == 0:
        return np.ones(N)
    
    return best_h
