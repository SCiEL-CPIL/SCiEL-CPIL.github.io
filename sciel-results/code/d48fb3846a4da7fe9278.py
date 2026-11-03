"""Erdos minimum overlap: minimize max overlap of h with shifted 1-h.
Approach: Multi-start Nelder-Mead optimization.
The problem is a non-convex quadratic programming problem (minimizing the max of quadratic forms).
Gradient methods can get stuck. We use Nelder-Mead, a derivative-free method, 
starting from multiple diverse initial points to escape local minima.
We handle the sum constraint by optimizing N-1 free variables and determining the 
N-th to satisfy the sum, with a penalty for boundary violations.
"""
import time
import numpy as np
from scipy.optimize import minimize

N = 100
_DEADLINE = time.time() + 115


def _linear_overlap_values(h):
    """
    Compute the linear overlap values C_k for k=0..N-1.
    C_k = (2/N) * (N/2 - sum_{i=k}^{N-1} h_i h_{i-k})
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


def _construct_h_from_free(x):
    """
    Reconstruct h from N-1 free variables x.
    h[i] = x[i] for i < N-1
    h[N-1] = N/2 - sum(x)
    Returns h and a penalty flag if out of bounds.
    """
    h = np.empty(N)
    h[:N-1] = x
    h[N-1] = N / 2.0 - np.sum(x)
    # Check bounds
    if np.any(h < -1e-9) or np.any(h > 1 + 1e-9):
        return None
    # Clip tiny errors
    h = np.clip(h, 0.0, 1.0)
    return h


def _objective(x):
    """
    Objective function for Nelder-Mead.
    x has length N-1.
    """
    h = _construct_h_from_free(x)
    if h is None:
        return 10.0  # Penalty for out of bounds
    return _score(h)


def construct():
    """Construct step heights minimizing maximum overlap with shifted complement."""
    t0 = time.time()
    np.random.seed(42)
    
    best_h = None
    best_score = float('inf')
    
    def _update_best(h):
        nonlocal best_h, best_score
        s = _score(h)
        if s < best_score:
            best_score = s
            best_h = h.copy()
            
    # Generate diverse initial points
    # We need to ensure the sum of the first N-1 is such that the last is in [0,1].
    # Sum of all N is N/2.
    # If we pick random h in [0,1] with sum N/2, we can project it.
    # For Nelder-Mead, we start with x (first N-1) derived from a valid h.
    
    initial_xs = []
    
    # 1. Uniform start
    h0 = np.full(N, 0.5)
    x0 = h0[:N-1].copy()
    initial_xs.append(x0)
    
    # 2. Cosine start
    i = np.arange(N)
    h_cos = 0.5 + 0.5 * np.cos(2.0 * np.pi * i / N)
    # Project to ensure sum is N/2 and bounds
    # Simple projection: scale? No, just clip and adjust sum?
    # For init, we just need a valid x.
    # Let's just use the random restarts mostly.
    
    # Random starts
    for _ in range(20):
        h_rand = np.random.uniform(0, 1, N)
        # Project to sum N/2
        # Simple: scale? No, we need to keep in [0,1].
        # Use the projection from the previous code or a simpler one.
        # Since N is small, we can do a quick bisection projection.
        target = N / 2.0
        s = np.sum(h_rand)
        # If sum is too high, we need to reduce.
        # Let's just generate h such that sum is close.
        # A better way: generate N-1 uniform, set last to balance, check bounds.
        # Retry until valid.
        for _retry in range(100):
            x_rand = np.random.uniform(0, 1, N-1)
            h_last = N/2.0 - np.sum(x_rand)
            if 0 <= h_last <= 1:
                initial_xs.append(x_rand)
                break
    
    # Run Nelder-Mead from each start
    for x0 in initial_xs:
        if time.time() > _DEADLINE - 10:
            break
            
        try:
            res = minimize(_objective, x0, method='Nelder-Mead', 
                           options={'maxiter': 2000, 'xatol': 1e-8, 'fatol': 1e-10})
            x_opt = res.x
            h_opt = _construct_h_from_free(x_opt)
            if h_opt is not None:
                _update_best(h_opt)
        except Exception:
            pass
            
    if best_h is None:
        # Fallback
        best_h = np.full(N, 0.5)
        
    return best_h
