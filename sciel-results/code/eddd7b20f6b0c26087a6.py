"""Erdos minimum overlap: minimize max overlap of h with shifted 1-h.
Strategy: High-Entropy Random Restart with Local SLSQP Refinement.
The previous strategy was biased towards low-frequency structures (cosines), 
which lead to large negative dips in the autocorrelation, increasing the max overlap.
This strategy uses random (white-noise-like) initializations which have flat 
spectra and thus flatter autocorrelation profiles. Intensive local refinement 
(SLSQP) is then applied to multiple random starts to polish the solution.
"""
import time
import numpy as np
from scipy.optimize import minimize

N = 100
_DEADLINE = time.time() + 110  # Safety margin before 120s kill

def _linear_overlap_values(h):
    """
    Compute the linear overlap values C_k for k=0..N-1.
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

    # Strategy: Random Restart with SLSQP
    # We avoid the low-frequency bias of the previous spectral approach.
    # Random uniform distributions have flat power spectra, leading to 
    # autocorrelations that are close to zero for non-zero lags, 
    # which minimizes the max overlap (targeting ~0.5).
    
    # Number of restarts to attempt
    n_restarts = 10
    
    for i in range(n_restarts):
        if time.time() - t0 > _DEADLINE - 10:
            break
            
        # Generate random initialization
        # Using uniform [0, 1]
        h_rand = np.random.uniform(0, 1, N)
        h_rand = _fix_sum(h_rand)
        
        # Quick score to see if it's already good (unlikely to be best, but safe)
        s_init = _score(h_rand)
        if s_init < best_score:
            best_score = s_init
            best_h = h_rand.copy()
            
        # Local Refinement using SLSQP
        # We use a smooth max approximation to make the objective differentiable
        # and provide gradients to SLSQP.
        
        h0 = h_rand.copy()
        
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
                # SLSQP is robust for this size
                res = minimize(obj_func, h0.copy(), jac=jac_func, method='SLSQP',
                               bounds=bounds, constraints=cons,
                               options={'maxiter': maxiter, 'ftol': ftol})
                if res.success or res.fun < best_score:
                    _update_best(res.x)
            except Exception:
                pass

        # Run optimization with moderate beta
        _run_optimization(h0, beta=50.0, maxiter=200, ftol=1e-9)
        
        # If time permits, run a second pass with higher beta for sharper max
        if time.time() - t0 < _DEADLINE - 5:
             _run_optimization(best_h, beta=100.0, maxiter=100, ftol=1e-10)

    if best_h is None:
        # Fallback: Random
        best_h = _fix_sum(np.random.uniform(0, 1, N))
    
    best_h = _fix_sum(best_h)
    return best_h
