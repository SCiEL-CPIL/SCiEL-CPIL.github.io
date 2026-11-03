"""Erdos minimum overlap: minimize max overlap of h with shifted 1-h.
Uses FFT-accelerated overlap computation with smooth-max optimization (log-sum-exp surrogate).
Multiple initializations: cosine, constant, and random, each refined with SLSQP.
Epigraph-style smoothing via high beta for near-exact max tracking."""
import time
import numpy as np
from scipy.optimize import minimize

N = 100
_DEADLINE = time.time() + 135  # Safety margin before 150s kill


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
    """Compute the maximum linear overlap."""
    h = np.asarray(h, dtype=np.float64)
    cs = _linear_overlap_values(h)
    return float(np.max(cs))


def _smooth_max_val_grad(h, beta):
    """
    Smooth maximum of the linear overlap values using log-sum-exp.
    Returns (value, gradient) vectorized via FFT-based gradient.
    """
    h = np.asarray(h, dtype=np.float64)
    n = len(h)
    cs = _linear_overlap_values(h)
    m = np.max(cs)
    exp_terms = np.exp(beta * (cs - m))
    sum_exp = np.sum(exp_terms)
    w = exp_terms / sum_exp  # Weights for soft max

    val = m + np.log(sum_exp) / beta

    # Gradient: dC_k/dh_j = -(2/n) * (h_{j-k} if j>=k + h_{j+k} if j+k<n)
    # Vectorized: for each k, factor = -(2/n) * w[k]
    # grad += factor * (h shifted appropriately)
    # Use convolution-like approach:
    # grad_j = -(2/n) * sum_k w[k] * (h_{j-k}*[j>=k] + h_{j+k}*[j+k<n])
    # This is: -(2/n) * (sum_k w[k]*h_{j-k} + sum_k w[k]*h_{j+k})
    # = -(2/n) * (corr_wh[j] + corr_wh_rev[j]) where corr_wh = convolve(w, h)
    
    # Compute convolve(w, h) for the first term and convolve(w, h reversed) for second
    # Actually, let's do it with explicit loops for clarity since N=100 is small
    grad = np.zeros(n)
    
    for k in range(n):
        if w[k] < 1e-15:
            continue
        factor = -(2.0 / n) * w[k]
        
        # Term 1: h_{j-k} for j >= k  => indices k..n-1 get h[0..n-1-k]
        if k > 0:
            grad[k:] += factor * h[:n-k]
        # Term 2: h_{j+k} for j+k < n => indices 0..n-1-k get h[k..n-1]
        if k < n:
            grad[:n-k] += factor * h[k:]
            
    return val, grad


def _fix_sum(h):
    """Adjust h to have sum exactly N/2, keeping values in [0, 1]."""
    h = np.array(h, dtype=np.float64, copy=True)
    target = N / 2.0
    diff = target - np.sum(h)
    h += diff / N
    h = np.clip(h, 0.0, 1.0)
    
    diff = target - np.sum(h)
    if abs(diff) > 1e-12:
        mask = (h > 0) & (h < 1)
        if np.any(mask):
            h[mask] += diff / np.sum(mask)
            h = np.clip(h, 0.0, 1.0)
            
    diff = target - np.sum(h)
    if abs(diff) > 1e-15:
        if diff > 0:
            idx = np.argmin(h)
        else:
            idx = np.argmax(h)
        h[idx] += diff
        h = np.clip(h, 0.0, 1.0)
        
    return h


def _run_slsqp(h0, beta, maxiter, ftol):
    """Run SLSQP optimization with smooth max objective."""
    cons = [{'type': 'eq', 'fun': lambda x: np.sum(x) - N/2.0,
             'jac': lambda x: np.ones(N)}]
    bounds = [(0.0, 1.0)] * N
    
    def obj_func(x):
        val, _ = _smooth_max_val_grad(x, beta)
        return val
        
    def jac_func(x):
        _, grad = _smooth_max_val_grad(x, beta)
        return grad
        
    try:
        res = minimize(obj_func, h0.copy(), jac=jac_func, method='SLSQP',
                       bounds=bounds, constraints=cons,
                       options={'maxiter': maxiter, 'ftol': ftol})
        return res.x
    except Exception:
        return h0


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
    
    # Initial candidates
    i = np.arange(N)
    
    # Cosine wave: h_i = 0.5 + 0.5 * cos(2*pi*i/N)
    h_cos = 0.5 + 0.5 * np.cos(2.0 * np.pi * i / N)
    _update_best(h_cos)
    
    # Constant wave
    h_const = np.full(N, 0.5)
    _update_best(h_const)
    
    # Sine-based: h_i = 0.5 + 0.4 * sin(2*pi*i/N)
    h_sin = 0.5 + 0.4 * np.sin(2.0 * np.pi * i / N)
    _update_best(h_sin)
    
    # Two-tone: alternate between 0.2 and 0.8
    h_alt = np.where(i % 2 == 0, 0.8, 0.2)
    _update_best(h_alt)
    
    # Optimization passes
    def _try_optimize(h0, beta, maxiter, ftol, label=""):
        nonlocal best_h, best_score
        if time.time() - t0 > _DEADLINE - 15:
            return
        h_opt = _run_slsqp(h0, beta, maxiter, ftol)
        _update_best(h_opt)
    
    # Phase 1: Refine cosine with moderate beta
    _try_optimize(h_cos, beta=30.0, maxiter=300, ftol=1e-9, label="cos_b30")
    
    # Phase 2: Refine constant
    _try_optimize(h_const, beta=30.0, maxiter=300, ftol=1e-9, label="const_b30")
    
    # Phase 3: Refine sine
    _try_optimize(h_sin, beta=30.0, maxiter=200, ftol=1e-9, label="sin_b30")
    
    # Phase 4: Random starts
    for _ in range(4):
        if time.time() - t0 > _DEADLINE - 25:
            break
        h_rand = np.random.uniform(0.1, 0.9, N)
        h_rand = _fix_sum(h_rand)
        _update_best(h_rand)
        _try_optimize(h_rand, beta=30.0, maxiter=200, ftol=1e-9, label="rand_b30")
    
    # Phase 5: Polish best with high beta
    if best_h is not None and time.time() - t0 < _DEADLINE - 10:
        _try_optimize(best_h, beta=80.0, maxiter=500, ftol=1e-12, label="polish_b80")
    
    # Phase 6: Final polish with very high beta
    if best_h is not None and time.time() - t0 < _DEADLINE - 5:
        _try_optimize(best_h, beta=200.0, maxiter=300, ftol=1e-13, label="polish_b200")
    
    if best_h is None:
        best_h = np.full(N, 0.5)
    
    best_h = _fix_sum(best_h)
    return best_h
