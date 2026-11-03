"""Erdos minimum overlap: minimize max overlap of h with shifted 1-h.
Strategy: Spectral Parameterization.
Represent h as a sum of low-frequency cosine/sine terms.
This reduces the search space from N=100 to ~2*M coefficients.
Uses global optimization (Differential Evolution) on the reduced space,
followed by local refinement using SLSQP on the full N-dimensional space
initialized by the best spectral solution.
"""
import time
import numpy as np
from scipy.optimize import minimize, differential_evolution

N = 100
_DEADLINE = time.time() + 110  # Safety margin before 120s kill

def _linear_overlap_values(h):
    """
    Compute the linear overlap values C_k for k=0..N-1.
    C_k = (2/N) * (N/2 - sum_{i=k}^{N-1} h_i h_{i-k})
    Vectorized using FFT for speed.
    """
    n = len(h)
    n_fft = 2 * n
    h_pad = np.zeros(n_fft)
    h_pad[:n] = h
    H = np.fft.rfft(h_pad, n=n_fft)
    corr = np.fft.irfft(H * np.conj(H), n=n_fft)
    r = corr[:n]
    cs = (2.0 / n) * (n / 2.0 - r)
    return cs

def _score(h):
    """
    Compute the maximum linear overlap.
    """
    h = np.asarray(h, dtype=np.float64)
    cs = _linear_overlap_values(h)
    return float(np.max(cs))

def _fix_sum(h):
    """
    Adjust h to have sum exactly N/2, keeping values in [0, 1].
    """
    h = np.array(h, dtype=np.float64, copy=True)
    target = N / 2.0
    diff = target - np.sum(h)
    h += diff / N
    h = np.clip(h, 0.0, 1.0)
    
    # If clipping changed the sum, do a second pass
    diff = target - np.sum(h)
    if abs(diff) > 1e-12:
        mask = (h > 0) & (h < 1)
        if np.any(mask):
            h[mask] += diff / np.sum(mask)
            h = np.clip(h, 0.0, 1.0)
            
    # Final tiny correction on one element to ensure exact sum in floating point
    diff = target - np.sum(h)
    if abs(diff) > 1e-15:
        if diff > 0:
            idx = np.argmin(h)
        else:
            idx = np.argmax(h)
        h[idx] += diff
        h = np.clip(h, 0.0, 1.0)
        
    return h

def _spectral_to_h(params, M):
    """
    Convert spectral coefficients to height sequence.
    params = [a0, a1, b1, a2, b2, ..., aM, bM]
    h_i = a0/2 + sum_{k=1}^M (a_k cos(2*pi*k*i/N) + b_k sin(2*pi*k*i/N))
    """
    i = np.arange(N)
    h = np.zeros(N)
    h += params[0] / 2.0  # DC component
    
    for k in range(1, M + 1):
        a_k = params[2 * k - 1]
        b_k = params[2 * k]
        h += a_k * np.cos(2 * np.pi * k * i / N)
        h += b_k * np.sin(2 * np.pi * k * i / N)
        
    return h

def _spectral_objective(params, M):
    """
    Objective function for spectral optimization.
    Returns the max overlap score.
    """
    h = _spectral_to_h(params, M)
    h = _fix_sum(h)
    return _score(h)

def _smooth_max_grad(h, beta):
    """
    Gradient of the smooth maximum objective.
    Returns (value, gradient)
    """
    h = np.asarray(h, dtype=np.float64)
    n = len(h)
    cs = _linear_overlap_values(h)
    m = np.max(cs)
    exp_terms = np.exp(beta * (cs - m))
    sum_exp = np.sum(exp_terms)
    w = exp_terms / sum_exp  # Weights for the soft max
    
    val = m + np.log(sum_exp) / beta
    
    grad = np.zeros(n)
    
    # C_k = 1 - (2/n) * r_k
    # dC_k/dh_j = -(2/n) * ( h_{j-k} (if j>=k) + h_{j+k} (if j+k<n) )
    
    # Vectorized gradient computation
    for k in range(n):
        if w[k] == 0.0:
            continue
        factor = -(2.0 / n) * w[k]
        
        # Term 1: h_{j-k} for j >= k
        if k > 0:
            grad[k:] += factor * h[:n-k]
        # Term 2: h_{j+k} for j+k < n
        if k < n:
            grad[:n-k] += factor * h[k:]
            
    return val, grad

def construct():
    """Construct step heights minimizing maximum overlap with shifted complement."""
    t0 = time.time()
    np.random.seed(42)
    
    best_h = None
    best_score = float('inf')
    
    def _update_best(h):
        nonlocal best_h, best_score
        h = _fix_sum(h)
        s = _score(h)
        if s < best_score:
            best_score = s
            best_h = h.copy()
    
    # 1. Spectral Global Optimization
    # Choose M=5 (11 parameters total: 1 DC + 5 cos + 5 sin)
    # This is a low-dimensional search space.
    M = 5
    n_params = 2 * M + 1
    
    # Bounds for spectral coefficients
    # To ensure h stays reasonably within [0,1] before clipping,
    # we bound coefficients. Max sum of cos terms is M, so a_k ~ 1/M is safe.
    # DC component a0 should be around 1.0 (since mean is 0.5, a0/2 = 0.5 => a0=1)
    # Let's set wide bounds to allow global search to find the right scale.
    bounds = [(0.5, 1.5)] + [(-1.0, 1.0)] * (2 * M)
    
    # Use Differential Evolution for global search
    # This is robust for small dimensions.
    try:
        if time.time() - t0 < _DEADLINE - 20:
            # maxiter=100, popsize=15 is reasonable for 11 dims
            res_de = differential_evolution(
                _spectral_objective, 
                bounds, 
                args=(M,),
                maxiter=100, 
                popsize=15,
                tol=1e-6,
                seed=42,
                polish=False # We will polish later
            )
            if res_de.success or res_de.fun < best_score:
                h_best_spectral = _spectral_to_h(res_de.x, M)
                _update_best(h_best_spectral)
    except Exception:
        pass

    # 2. Local Refinement on Full Space
    # Use the best spectral solution as a starting point for SLSQP on N=100 vars.
    # This allows the solution to deviate from the strict spectral form if beneficial.
    if best_h is not None and time.time() - t0 < _DEADLINE - 10:
        h0 = best_h.copy()
        
        def _run_optimization(h0, beta, maxiter, ftol):
            nonlocal best_h, best_score
            
            cons = [{'type': 'eq', 'fun': lambda x: np.sum(x) - N/2.0,
                     'jac': lambda x: np.ones(N)}]
            bounds = [(0.0, 1.0)] * N
            
            def obj_func(x):
                val, _ = _smooth_max_grad(x, beta)
                return val
                
            def jac_func(x):
                _, grad = _smooth_max_grad(x, beta)
                return grad
                
            try:
                res = minimize(obj_func, h0.copy(), jac=jac_func, method='SLSQP',
                               bounds=bounds, constraints=cons,
                               options={'maxiter': maxiter, 'ftol': ftol})
                if res.success or res.fun < best_score:
                    _update_best(res.x)
            except Exception:
                pass

        # Phase 2a: Refine spectral start
        _run_optimization(h0, beta=50.0, maxiter=200, ftol=1e-9)
        
        # Phase 2b: Try a few random starts if time permits
        if time.time() - t0 < _DEADLINE - 5:
            for _ in range(2):
                if time.time() - t0 > _DEADLINE - 2:
                    break
                h_rand = np.random.uniform(0, 1, N)
                h_rand = _fix_sum(h_rand)
                _update_best(h_rand)
                _run_optimization(h_rand, beta=50.0, maxiter=100, ftol=1e-8)

    if best_h is None:
        # Fallback: Cosine wave
        i = np.arange(N)
        h_cos = 0.5 + 0.5 * np.cos(2.0 * np.pi * i / N)
        best_h = _fix_sum(h_cos)
    
    best_h = _fix_sum(best_h)
    return best_h
