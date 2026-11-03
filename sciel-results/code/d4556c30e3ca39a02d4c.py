"""Erdos minimum overlap: minimize max overlap of h with shifted 1-h.
Strategy: Direct algebraic construction + 1D Root Finding + Active Set Equalization.
Instead of iterative SLSQP on a smooth max, we construct patterns that force
symmetry in the overlap values, then scale/amplitude-tune to minimize the peak.
"""
import time
import numpy as np
from scipy.optimize import minimize_scalar

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


def _construct_pattern(period, amp, phase=0):
    """
    Construct a periodic step pattern.
    h_i = 0.5 + amp * sin(2*pi*i/period + phase)
    Then fix sum and bounds.
    """
    i = np.arange(N)
    h = 0.5 + amp * np.sin(2.0 * np.pi * i / period + phase)
    h = _fix_sum(h)
    return h


def _optimize_amplitude(pattern_func, amp_range=(0.0, 1.0)):
    """
    Find the amplitude that minimizes the max overlap for a given pattern generator.
    """
    def obj(amp):
        h = pattern_func(amp)
        if h is None:
            return 1.0
        return _score(h)
    
    res = minimize_scalar(obj, bounds=amp_range, method='bounded', options={'xatol': 1e-6})
    if res.success:
        best_amp = res.x
        best_h = pattern_func(best_amp)
        best_s = res.fun
        return best_h, best_s
    else:
        # Fallback
        mid = (amp_range[0] + amp_range[1]) / 2
        h = pattern_func(mid)
        return h, _score(h)


def _equalize_active_set(h, max_iter=50, tol=1e-8):
    """
    Refine h by identifying the 'active' shifts (those close to the max)
    and adjusting h to equalize their overlap values.
    This is a local refinement step.
    """
    h = np.array(h, dtype=np.float64, copy=True)
    for _ in range(max_iter):
        cs = _linear_overlap_values(h)
        max_c = np.max(cs)
        # Identify active set: shifts within tol of max
        active = cs > (max_c - 1e-4)
        
        if not np.any(active):
            break
            
        # If only one active, we can't equalize, just return
        if np.sum(active) < 2:
            break
        
        # We want to adjust h such that C_k for k in active set are equal.
        # This is a linear system: dC_k/dh_j * dh = target - C_k
        # Since we are doing a simple refinement, we use a projected gradient step
        # on the active set only, aiming to reduce the variance of C_k.
        
        # Calculate gradient for active shifts only
        n = len(h)
        grad = np.zeros(n)
        
        # For each active shift k, compute gradient contribution
        # C_k = 1 - (2/n) * sum_{i=k}^{n-1} h[i] h[i-k]
        # dC_k/dh_j = -(2/n) * ( h_{j-k} (if j>=k) + h_{j+k} (if j+k<n) )
        
        # We want to minimize sum_{k in active} (C_k - mean(C_active))^2
        # This is equivalent to making them equal.
        
        # Simple heuristic: Move h in direction that reduces the highest C_k
        # and increases the lowest C_k in the active set.
        
        max_idx = np.argmax(cs)
        min_idx = np.argmin(cs[active]) + np.where(active)[0][0] # Approximate
        
        # Actually, let's just do a small step in the direction that reduces the max
        # by adjusting the specific h values that contribute most to C_max.
        
        # Gradient of C_max w.r.t h
        k_max = max_idx
        g = np.zeros(n)
        factor = -(2.0 / n)
        if k_max > 0:
            g[k_max:] += factor * h[:n-k_max]
        g[:n-k_max] += factor * h[k_max:]
        
        # Step in negative gradient direction (to decrease C_max)
        # But we must maintain sum(h) = N/2.
        # Project gradient onto subspace sum(dh)=0
        g = g - np.mean(g)
        
        # Normalize
        norm_g = np.linalg.norm(g)
        if norm_g < 1e-12:
            break
            
        # Adaptive step size
        step = 0.01
        h_new = h - step * (g / norm_g)
        h_new = _fix_sum(h_new)
        
        new_score = _score(h_new)
        if new_score < max_c - tol:
            h = h_new
        else:
            # Try smaller step
            step *= 0.5
            if step < 1e-10:
                break
            h_new = h - step * (g / norm_g)
            h_new = _fix_sum(h_new)
            if _score(h_new) < max_c:
                h = h_new
            else:
                break
                
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
    
    # 1. Try various periodic patterns
    # The overlap C_k depends on autocorrelation.
    # A periodic signal has strong autocorrelation peaks at multiples of the period.
    # We want to flatten these peaks.
    
    periods_to_try = [2, 3, 4, 5, 6, 7, 10, 12, 14, 16, 20, 25, 50]
    
    for p in periods_to_try:
        if time.time() - t0 > _DEADLINE - 20:
            break
            
        # Define pattern generator for this period
        def gen(amp, p=p):
            return _construct_pattern(p, amp)
            
        h_cand, s_cand = _optimize_amplitude(gen, amp_range=(0.0, 1.0))
        if h_cand is not None:
            _update_best(h_cand)
            
            # Refine with equalization
            if time.time() - t0 < _DEADLINE - 10:
                h_refined = _equalize_active_set(h_cand)
                _update_best(h_refined)
                
    # 2. Try random starts with equalization
    if time.time() - t0 < _DEADLINE - 10:
        for _ in range(5):
            if time.time() - t0 > _DEADLINE - 5:
                break
            h_rand = np.random.uniform(0, 1, N)
            h_rand = _fix_sum(h_rand)
            _update_best(h_rand)
            
            if time.time() - t0 < _DEADLINE - 5:
                h_refined = _equalize_active_set(h_rand)
                _update_best(h_refined)

    if best_h is None:
        best_h = np.full(N, 0.5)
    
    best_h = _fix_sum(best_h)
    return best_h
