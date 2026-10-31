import numpy as np
import time
from scipy.optimize import minimize

N = 100
_DEADLINE = time.time() + 110  # Safety margin before 120s kill

def _linear_overlap_values(h):
    """
    Compute the linear overlap values C_k for k=0..N-1.
    C_k = (2/N) * (N/2 - sum_{i=k}^{N-1} h_i h_{i-k})
    This corresponds to the integral of h(x) * (1 - h(x-k)) over the domain,
    assuming h is zero outside [0, N].
    Vectorized using FFT for speed.
    """
    n = len(h)
    # r_k = sum_{i=k}^{n-1} h[i] * h[i-k] for k=0..n-1
    # This is the autocorrelation. Use FFT for O(n log n) instead of O(n^2).
    # Zero-pad to avoid circular convolution issues
    n_fft = 2 * n
    h_pad = np.zeros(n_fft)
    h_pad[:n] = h
    H = np.fft.rfft(h_pad, n=n_fft)
    # Autocorrelation: ifft(|H|^2) gives circular autocorrelation of padded signal
    # For linear autocorrelation of length-n signal, we need:
    # r[k] = sum_{i=0}^{n-1-k} h[i+k] * h[i] for k=0..n-1
    # Using FFT: correlate h with h
    corr = np.fft.irfft(H * np.conj(H), n=n_fft)
    # corr[k] for k=0..n-1 gives r_k = sum_{i=k}^{n-1} h[i]*h[i-k]
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

def _smooth_max(h, beta):
    """
    Smooth maximum of the linear overlap values using log-sum-exp.
    """
    h = np.asarray(h, dtype=np.float64)
    cs = _linear_overlap_values(h)
    m = np.max(cs)
    # Avoid overflow in exp
    return m + np.log(np.sum(np.exp(beta * (cs - m)))) / beta

def _smooth_max_grad(h, beta):
    """
    Gradient of the smooth maximum objective.
    Returns (value, gradient)
    Vectorized gradient computation.
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
    
    # Vectorized computation:
    # For each k, the contribution to grad[j] is w[k] * (-(2/n)) * (h[j-k] + h[j+k])
    # where h[j-k] is 0 if j<k and h[j+k] is 0 if j+k>=n
    
    # We can compute this using convolution or by building the matrix
    # Let's use a loop over k but vectorize over j for speed
    for k in range(n):
        if w[k] == 0.0:
            continue
        factor = -(2.0 / n) * w[k]
        
        # Term 1: h[j-k] for j >= k, i.e., h[0:n-k] contributes to grad[k:n]
        # grad[k:n] += factor * h[0:n-k]
        if n - k > 0:
            grad[k:] += factor * h[:n-k]
        
        # Term 2: h[j+k] for j+k < n, i.e., h[k:n] contributes to grad[0:n-k]
        # grad[0:n-k] += factor * h[k:n]
        if n - k > 0:
            grad[:n-k] += factor * h[k:]
            
    return val, grad

def _fix_sum(h):
    """
    Adjust h to have sum exactly N/2, keeping values in [0, 1].
    """
    h = np.array(h, dtype=np.float64, copy=True)
    target = N / 2.0
    diff = target - np.sum(h)
    # Distribute the difference evenly, but clip
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

def construct():
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
    
    # 1. Analytical Start: Cosine Wave
    # h_i = 0.5 + 0.5 * cos(2 * pi * i / N)
    i = np.arange(N)
    h_cos = 0.5 + 0.5 * np.cos(2.0 * np.pi * i / N)
    h_cos = _fix_sum(h_cos)
    _update_best(h_cos)
    
    # 2. Optimization with Analytical Gradient
    # Use SLSQP with the smooth max objective and its gradient
    
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

    # Phase 1: Refine Cosine Start
    if time.time() - t0 < _DEADLINE - 30:
        _run_optimization(h_cos, beta=50.0, maxiter=200, ftol=1e-9)
        
    # Phase 2: Try other starts if time permits
    # Random starts
    if time.time() - t0 < _DEADLINE - 20:
        for _ in range(3):
            if time.time() - t0 > _DEADLINE - 10:
                break
            h_rand = np.random.uniform(0, 1, N)
            h_rand = _fix_sum(h_rand)
            _update_best(h_rand)
            _run_optimization(h_rand, beta=50.0, maxiter=100, ftol=1e-8)
            
    # Phase 3: Final Polish with higher beta
    if best_h is not None and time.time() - t0 < _DEADLINE - 5:
        _run_optimization(best_h, beta=100.0, maxiter=300, ftol=1e-12)
        
    if best_h is None:
        best_h = np.full(N, 0.5)
    
    best_h = _fix_sum(best_h)
    return best_h
