"""
Strategy: Low-dimensional parametric optimization of Golay-like sequences.
We model h as a sum of M complex Gaussian-modulated sinusoids (chirps).
h[n] = sum_{k=1}^M A_k * exp(i * (2*pi*f_k*n + pi*beta_k*n^2/N)) * exp(-((n-c_k)/sigma)^2)
We optimize the real parameters (A_k, f_k, beta_k, c_k, sigma) to minimize the objective.
Since the objective is scale-invariant, we fix the norm or sum to avoid the 0/0 issue,
or simply normalize h inside the evaluation.
The key insight is that the max of the autocorrelation is a smooth function of the
parameters almost everywhere, allowing Nelder-Mead to work effectively on this
reduced dimensionality (5*M parameters instead of 400).
"""

import numpy as np
import time
from scipy.optimize import minimize

N = 400
_DEADLINE = 145.0

def _evaluate(h):
    """Compute the task value."""
    s = np.sum(h)
    if s == 0 or not np.isfinite(s):
        return np.inf
    # Normalize to avoid overflow/underflow in intermediate steps if needed, 
    # but the formula is scale invariant: (2*N*max(conv))/sum^2
    # conv scales with h^2, sum^2 scales with h^2. So ratio is invariant.
    conv = np.convolve(h, h)
    max_conv = np.max(conv)
    val = abs(2.0 * N * max_conv / (s * s))
    if not np.isfinite(val):
        return np.inf
    return val

def _build_h(params, M):
    """
    Construct h from parameters.
    params layout for each component k: [A, f, beta, c, sigma]
    Total length = 5*M
    """
    n = np.arange(N)
    h = np.zeros(N)
    for k in range(M):
        A = params[5*k]
        f = params[5*k + 1]
        beta = params[5*k + 2]
        c = params[5*k + 3]
        sigma = params[5*k + 4]
        
        # Avoid division by zero or extremely small sigma
        sigma = max(abs(sigma), 1.0)
        
        # Phase: 2*pi*f*n + pi*beta*n^2/N
        # Using n/N for beta to keep it in a reasonable range
        phase = 2.0 * np.pi * f * n + np.pi * beta * (n / N) ** 2
        # Gaussian envelope centered at c
        envelope = np.exp(-((n - c) ** 2) / (2.0 * sigma ** 2))
        
        # Complex component
        comp = A * np.exp(1j * phase) * envelope
        h += np.real(comp)
        
    return h

def _objective(params, M):
    """
    Objective function for Nelder-Mead.
    """
    try:
        h = _build_h(params, M)
        if not np.all(np.isfinite(h)):
            return 1e10
        val = _evaluate(h)
        if not np.isfinite(val):
            return 1e10
        return val
    except Exception:
        return 1e10

def _optimize_golay_like(M, init_params, deadline):
    """
    Run Nelder-Mead on the parametric model.
    """
    t0 = time.time()
    try:
        res = minimize(
            _objective,
            init_params,
            args=(M,),
            method='Nelder-Mead',
            options={'maxiter': 500, 'xatol': 1e-8, 'fatol': 1e-8, 'adaptive': True}
        )
        return res.x, res.fun
    except Exception:
        return init_params, _objective(init_params, M)

def construct():
    """
    Construct step heights to minimize the autocorrelation inequality value.
    """
    t0 = time.time()
    deadline = t0 + _DEADLINE
    
    rng = np.random.default_rng(42)
    
    best_val = 2.0
    best_h = np.ones(N)
    
    # We will try different numbers of components M
    # M=1 is a single chirp. M=2 is a pair (like Golay).
    # Higher M allows more complex cancellation.
    
    # Phase 1: Coarse search over M=1 and M=2 with random initializations
    # We focus on low frequencies because high frequencies create sharp peaks in autocorrelation
    # unless perfectly canceled.
    
    for M in [1, 2, 3]:
        if time.time() > deadline - 10:
            break
            
        n_trials = 10 if M == 1 else 5
        
        for trial in range(n_trials):
            if time.time() > deadline - 5:
                break
                
            # Initialize parameters
            params = []
            for k in range(M):
                A = rng.uniform(0.5, 2.0)
                # Low frequency range
                f = rng.uniform(0.1, 2.0)
                # Chirp rate
                beta = rng.uniform(-5.0, 5.0)
                # Center
                c = rng.uniform(0.2 * N, 0.8 * N)
                # Width
                sigma = rng.uniform(20.0, 100.0)
                params.extend([A, f, beta, c, sigma])
            
            # Optimize
            opt_params, opt_val = _optimize_golay_like(M, np.array(params), deadline)
            
            if np.isfinite(opt_val) and opt_val < best_val:
                best_val = opt_val
                best_h = _build_h(opt_params, M)
                
    # Phase 2: Refine the best solution with local perturbations on the full vector
    # This allows the shape to deviate from the strict parametric form if beneficial
    if time.time() < deadline - 10:
        # Use a simple local refinement on the 400-dim vector
        # We use a gradient-free local search (Nelder-Mead on subsets or random steps)
        # Since 400 dims is too high for NM, we do random step refinement
        
        current_h = best_h.copy()
        current_val = best_val
        step_size = 0.1
        rng_refine = np.random.default_rng(123)
        
        for i in range(5000):
            if time.time() > deadline - 2:
                break
            
            # Perturb a few random indices
            n_perturb = rng_refine.integers(1, 10)
            idx = rng_refine.choice(N, size=n_perturb, replace=False)
            h_new = current_h.copy()
            h_new[idx] += rng_refine.normal(0, step_size, n_perturb)
            
            v_new = _evaluate(h_new)
            if np.isfinite(v_new) and v_new < current_val - 1e-10:
                current_val = v_new
                current_h = h_new
                step_size *= 0.99
            else:
                step_size *= 0.999
                
        if current_val < best_val:
            best_val = current_val
            best_h = current_h
            
    # Final check
    final_val = _evaluate(best_h)
    if not np.isfinite(final_val) or final_val >= 2.0:
        best_h = np.ones(N)
        
    return best_h

