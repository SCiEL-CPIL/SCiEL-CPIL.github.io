"""Erdos minimum overlap: minimize max overlap of h with shifted 1-h.
New approach: direct maximization of the minimum 'gap' G = 0.5 - max(C)
using SLSQP with exact subgradient of the max, plus alternating projection
onto the affine sum constraint and box constraints (POCS-style).
"""
import time
import numpy as np
from scipy.optimize import minimize

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


def _min_gap(h):
    """
    Compute the minimum gap G = 0.5 - max(C).
    We want to maximize G.
    """
    h = np.asarray(h, dtype=np.float64)
    cs = _linear_overlap_values(h)
    return float(0.5 - np.max(cs))


def _min_gap_grad(h):
    """
    Gradient of the minimum gap G = 0.5 - max(C).
    Since G = 0.5 - max_k C_k, the subgradient is -e_k for the argmax k.
    We use a weighted average over near-argmax indices for smoothness.
    Returns (value, gradient) where gradient is for maximizing G.
    """
    h = np.asarray(h, dtype=np.float64)
    n = len(h)
    cs = _linear_overlap_values(h)
    m = np.max(cs)
    G = 0.5 - m
    
    # Find indices where C_k is close to max
    threshold = m - 1e-6
    active = np.where(cs >= threshold)[0]
    if len(active) == 0:
        active = np.array([np.argmax(cs)])
    
    # Weighted subgradient: average over active indices
    grad = np.zeros(n)
    for k in active:
        # dC_k/dh_j = -(2/n) * (h_{j-k} (if j>=k) + h_{j+k} (if j+k<n))
        # So dG/dh_j = -dC_k/dh_j = (2/n) * (h_{j-k} (if j>=k) + h_{j+k} (if j+k<n))
        factor = (2.0 / n)
        if k > 0:
            grad[k:] += factor * h[:n-k]
        if k < n:
            grad[:n-k] += factor * h[k:]
    
    # Average over active indices
    grad /= len(active)
    
    return G, grad


def _fix_sum(h):
    """
    Adjust h to have sum exactly N/2, keeping values in [0, 1].
    """
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


def _project_onto_box_and_sum(h):
    """
    Project h onto the set {x : 0 <= x_i <= 1, sum(x) = N/2} using
    alternating projections (POCS).
    """
    h = np.array(h, dtype=np.float64, copy=True)
    for _ in range(50):
        # Project onto box
        h = np.clip(h, 0.0, 1.0)
        # Project onto sum constraint
        diff = N / 2.0 - np.sum(h)
        h += diff / N
        h = np.clip(h, 0.0, 1.0)
    return h


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
    
    # 1. Analytical Start: Cosine Wave
    i = np.arange(N)
    h_cos = 0.5 + 0.5 * np.cos(2.0 * np.pi * i / N)
    h_cos = _fix_sum(h_cos)
    _update_best(h_cos)
    
    # 2. Optimization: Maximize min gap using SLSQP
    # We minimize -G = max(C) - 0.5, which is equivalent to minimizing max(C)
    
    def _run_optimization(h0, maxiter, ftol):
        nonlocal best_h, best_score
        
        cons = [{'type': 'eq', 'fun': lambda x: np.sum(x) - N/2.0,
                 'jac': lambda x: np.ones(N)}]
        bounds = [(0.0, 1.0)] * N
        
        def obj_func(x):
            G, _ = _min_gap_grad(x)
            return -G  # Minimize negative gap = maximize gap
            
        def jac_func(x):
            _, grad = _min_gap_grad(x)
            return -grad  # Negative because we're minimizing -G
            
        try:
            res = minimize(obj_func, h0.copy(), jac=jac_func, method='SLSQP',
                           bounds=bounds, constraints=cons,
                           options={'maxiter': maxiter, 'ftol': ftol})
            if res.fun < best_score:
                _update_best(res.x)
        except Exception:
            pass

    # Phase 1: Refine Cosine Start
    if time.time() - t0 < _DEADLINE - 30:
        _run_optimization(h_cos, maxiter=300, ftol=1e-10)
        
    # Phase 2: Try other starts if time permits
    if time.time() - t0 < _DEADLINE - 20:
        # Try sine wave
        h_sin = 0.5 + 0.5 * np.sin(2.0 * np.pi * i / N)
        h_sin = _fix_sum(h_sin)
        _update_best(h_sin)
        _run_optimization(h_sin, maxiter=200, ftol=1e-9)
        
        # Random starts
        for _ in range(3):
            if time.time() - t0 > _DEADLINE - 10:
                break
            h_rand = np.random.uniform(0, 1, N)
            h_rand = _fix_sum(h_rand)
            _update_best(h_rand)
            _run_optimization(h_rand, maxiter=150, ftol=1e-8)
            
    # Phase 3: Alternating projection polish
    if best_h is not None and time.time() - t0 < _DEADLINE - 5:
        h_polished = _project_onto_box_and_sum(best_h)
        _update_best(h_polished)
        
        # One more optimization round from polished start
        if time.time() - t0 < _DEADLINE - 3:
            _run_optimization(h_polished, maxiter=200, ftol=1e-11)
        
    if best_h is None:
        best_h = np.full(N, 0.5)
    
    best_h = _fix_sum(best_h)
    return best_h
