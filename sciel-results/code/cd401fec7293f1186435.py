import numpy as np
import time
from scipy.optimize import minimize
from scipy.special import logsumexp

N = 400
_DEADLINE = 145.0

def _compute_value(h):
    """Compute the exact task value."""
    s = np.sum(h)
    if s == 0 or not np.isfinite(s):
        return np.inf
    conv = np.convolve(h, h)
    return abs(2.0 * N * np.max(conv) / (s * s))

def _surrogate_objective(h_flat, beta=50.0):
    """
    Smooth surrogate for the max of the convolution.
    We assume h is scaled such that sum(h) is fixed (e.g., N).
    If we parameterize with sum constraint, we can just minimize soft_max(conv).
    """
    # Reconstruct h from flat array if needed, but here we pass h directly 
    # and handle constraint via parameterization in the wrapper.
    
    s = np.sum(h_flat)
    # To make it scale invariant and robust, we normalize h in the objective 
    # calculation? No, if we fix sum in parameterization, s is constant.
    
    conv = np.convolve(h_flat, h_flat)
    
    # logsumexp for max
    # conv can be large, so subtract max for numerical stability
    max_conv = np.max(conv)
    # If max_conv is very negative, logsumexp might underflow? 
    # exp(beta * (conv - max_conv)) is stable.
    weights = np.exp(beta * (conv - max_conv))
    soft_max = max_conv + np.log(np.sum(weights)) / beta
    
    # The objective to minimize is soft_max.
    # Since we fix sum(h)=N, minimizing soft_max minimizes V.
    # V = 2 * N * max_conv / N^2 = 2 * max_conv / N.
    
    if not np.isfinite(soft_max):
        return 1e10
        
    return soft_max

def construct():
    t0 = time.time()
    deadline = t0 + _DEADLINE
    
    best_h = np.ones(N)
    best_val = _compute_value(best_h)
    
    # We parameterize h using N-1 variables x_0 ... x_{N-2}
    # h_i = x_i for i < N-1
    # h_{N-1} = N - sum(x_0 ... x_{N-2})
    
    def _unpack(x):
        h = np.empty(N)
        h[:N-1] = x
        h[N-1] = N - np.sum(x)
        return h

    def _obj(x):
        if time.time() > deadline - 5:
            raise ValueError("Timeout")
        h = _unpack(x)
        return _surrogate_objective(h, beta=50.0)
        
    # Initial guess: random small deviations from 1
    # Or use a chirp sequence (Zadoff-Chu real part) scaled to sum N
    
    # Try a few initializations
    
    inits = []
    
    # 1. Random normal, scaled
    np.random.seed(42)
    x_rand = np.random.normal(1.0, 0.1, N-1)
    inits.append(x_rand)
    
    # 2. Zadoff-Chu real part
    # z[n] = exp(-j pi root n(n+1)/N)
    # h[n] = Re(z[n])
    # Scale to sum N
    
    candidate_roots = [1, 3, 7, 9, 11, 13, 17, 19, 21, 23, 27, 29, 31, 33, 37, 39]
    
    for root in candidate_roots[:8]: # Limit count to save time
        if time.time() > deadline - 30:
            break
        n = np.arange(N)
        phase = -np.pi * root * n * (n + 1) / N
        z = np.exp(1j * phase)
        h_zc = np.real(z)
        
        # Scale to sum N
        s_zc = np.sum(h_zc)
        if abs(s_zc) < 1e-10:
            continue
        h_zc_scaled = h_zc * (N / s_zc)
        
        # Extract x (first N-1 elements)
        x_zc = h_zc_scaled[:N-1]
        inits.append(x_zc)
        
        # Evaluate initial value for this init
        h_init = _unpack(x_zc)
        val_init = _compute_value(h_init)
        if val_init < best_val:
            best_val = val_init
            best_h = h_init.copy()
            
    # Run optimization for each init
    
    for x_init in inits:
        if time.time() > deadline - 10:
            break
            
        try:
            res = minimize(
                _obj,
                x_init,
                method='L-BFGS-B',
                options={'maxiter': 100, 'ftol': 1e-10, 'gtol': 1e-6, 'maxfun': 200}
            )
            
            if res.success:
                h_res = _unpack(res.x)
                val_res = _compute_value(h_res)
                if val_res < best_val:
                    best_val = val_res
                    best_h = h_res.copy()
                    
        except Exception:
            pass

    # Final check
    final_val = _compute_value(best_h)
    if not np.isfinite(final_val) or final_val >= 2.0:
        best_h = np.ones(N)
        
    return best_h
