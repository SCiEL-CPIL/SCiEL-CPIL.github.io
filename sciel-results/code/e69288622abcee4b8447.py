"""Optimize 50 nonnegative step heights on [-1/4,1/4] to maximize
   (sum h_i^2) / (sum h_i * sup_k (h*h)_k)
   where (h*h)_k = sum_i h_i h_{i+k} is the autoconvolution at lag k.

Structural change: parameterize the heights via a small set of low-order
cosine (spectral) coefficients. This reduces the 50-dimensional search to a
~6-dimensional one, allowing exhaustive/random search over the spectral
space followed by local refinement. The spectral parameterization naturally
encodes band-limited, smooth signals where the autoconvolution peak is
linked to the spectral width.
"""
import time
import numpy as np
from scipy.optimize import minimize, differential_evolution

N = 50

def _autoconv_max(h):
    """Compute max_k sum_i h[i] * h[i-k] for the given height array."""
    c = np.correlate(h, h, mode='full')
    return c.max()

def _ratio(h):
    """Compute the ratio (sum h^2) / (sum h * max autoconv)."""
    h = np.asarray(h, dtype=float)
    l2_sq = np.sum(h**2)
    l1 = np.sum(h)
    if l1 < 1e-12:
        return 0.0
    ac_max = _autoconv_max(h)
    if ac_max < 1e-12:
        return 0.0
    return l2_sq / (l1 * ac_max)

def _neg_ratio(h):
    return -_ratio(h)

def _spectral_to_heights(params):
    """
    Convert spectral parameters to 50 nonnegative heights.
    
    params = [a0, a1, b1, a2, b2, a3, b3]
    h[i] = a0 + a1*cos(2*pi*i/N) + b1*sin(2*pi*i/N)
              + a2*cos(4*pi*i/N) + b2*sin(4*pi*i/N)
              + a3*cos(6*pi*i/N) + b3*sin(6*pi*i/N)
    
    We clip to nonnegative and normalize.
    """
    a0 = params[0]
    a1, b1 = params[1], params[2]
    a2, b2 = params[3], params[4]
    a3, b3 = params[5], params[6]
    
    i = np.arange(N, dtype=float)
    h = (a0 
         + a1 * np.cos(2 * np.pi * i / N) + b1 * np.sin(2 * np.pi * i / N)
         + a2 * np.cos(4 * np.pi * i / N) + b2 * np.sin(4 * np.pi * i / N)
         + a3 * np.cos(6 * np.pi * i / N) + b3 * np.sin(6 * np.pi * i / N))
    
    # Clip to nonnegative
    h = np.maximum(h, 0.0)
    
    # Normalize so max is 1 (scale invariance: ratio is scale-invariant)
    hmax = h.max()
    if hmax > 1e-12:
        h = h / hmax
    return h

def _neg_ratio_spectral(params):
    h = _spectral_to_heights(params)
    return -_ratio(h)

def _coordinate_descent(h, max_iter=300, tol=1e-10):
    """Coordinate descent: optimize one coordinate at a time."""
    h = np.asarray(h, dtype=float).copy()
    n = len(h)
    rng_local = np.random.default_rng(hash(tuple(h)) % 2**31)
    for it in range(max_iter):
        improved = False
        for i in range(n):
            h_old = h[i]
            candidates = [0.0]
            if h_old > 1e-6:
                candidates.extend([h_old * 0.5, h_old * 1.5, h_old * 0.1, h_old * 5.0])
            else:
                candidates.extend([0.01, 0.1, 0.5, 1.0])
            for _ in range(2):
                candidates.append(rng_local.exponential(scale=max(h_old, 0.1)))
            
            best_val = _ratio(h)
            best_h_i = h_old
            for v in candidates:
                h[i] = v
                r = _ratio(h)
                if r > best_val:
                    best_val = r
                    best_h_i = v
            h[i] = best_h_i
            if abs(best_h_i - h_old) > tol:
                improved = True
        if not improved:
            break
    return h

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin
    
    best_h = np.ones(N)
    best_score = _ratio(best_h)
    rng = np.random.default_rng(42)
    
    # Strategy 1: Sparse signals (single spike, few spikes)
    for pos in range(N):
        h = np.zeros(N)
        h[pos] = 1.0
        r = _ratio(h)
        if r > best_score:
            best_score = r
            best_h = h.copy()
    
    # Two spikes
    for i in range(N):
        for j in range(i+1, N):
            if time.time() > deadline - 80:
                break
            h = np.zeros(N)
            h[i] = 1.0
            h[j] = 1.0
            r = _ratio(h)
            if r > best_score:
                best_score = r
                best_h = h.copy()
    
    # Strategy 2: Spectral parameterization search
    # The ratio is scale-invariant, so we work with normalized heights.
    # Search over 7 spectral parameters.
    
    # Random search in spectral space
    n_spectral = 2000
    for k in range(n_spectral):
        if time.time() > deadline - 50:
            break
        # Random spectral params
        params = rng.uniform(-1.0, 1.0, size=7)
        # Ensure a0 is large enough to keep h mostly nonnegative
        params[0] = abs(params[0]) + 0.5
        h = _spectral_to_heights(params)
        if h.max() < 1e-12:
            continue
        r = _ratio(h)
        if r > best_score:
            best_score = r
            best_h = h.copy()
    
    # Differential evolution in spectral space
    if time.time() < deadline - 30:
        try:
            bounds = [(-2.0, 2.0)] * 7
            bounds[0] = (0.1, 3.0)  # a0 positive
            res = differential_evolution(
                _neg_ratio_spectral, bounds,
                seed=42, maxiter=50, popsize=15,
                tol=1e-8, polish=True
            )
            h_spec = _spectral_to_heights(res.x)
            if h_spec.max() > 1e-12:
                r = _ratio(h_spec)
                if r > best_score:
                    best_score = r
                    best_h = h_spec.copy()
        except Exception:
            pass
    
    # Strategy 3: L-BFGS-B refinement in spectral space from best spectral params
    # Find best spectral params by trying to match best_h
    if time.time() < deadline - 20:
        # Try L-BFGS-B from a few spectral starting points
        spectral_starts = []
        # All zeros (flat)
        spectral_starts.append(np.array([1.0, 0, 0, 0, 0, 0, 0]))
        # Some structured starts
        for a0 in [0.5, 1.0, 2.0]:
            for a1 in [-0.5, 0.5]:
                spectral_starts.append(np.array([a0, a1, 0, 0, 0, 0, 0]))
        # From the random search best (if any found)
        
        for s0 in spectral_starts:
            if time.time() > deadline - 10:
                break
            try:
                res = minimize(_neg_ratio_spectral, s0, method='L-BFGS-B',
                              bounds=[(0.01, 5.0)] + [(-2.0, 2.0)] * 6,
                              options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-10})
                h_spec = _spectral_to_heights(res.x)
                if h_spec.max() > 1e-12:
                    r = _ratio(h_spec)
                    if r > best_score:
                        best_score = r
                        best_h = h_spec.copy()
            except Exception:
                pass
    
    # Strategy 4: L-BFGS-B refinement directly on heights from best
    if time.time() < deadline - 15:
        try:
            h0 = best_h.copy()
            if h0.max() > 0:
                h0 = h0 / h0.max()
            res = minimize(_neg_ratio, h0, method='L-BFGS-B',
                          bounds=[(0, None)] * N,
                          options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-10})
            if res.fun < _neg_ratio(best_h):
                best_h = np.maximum(res.x, 0)
                best_score = -res.fun
        except Exception:
            pass
    
    # Strategy 5: Coordinate descent from diverse starts
    starts = []
    # Sparse random
    for _ in range(15):
        h = np.zeros(N)
        k = rng.integers(1, 8)
        idx = rng.choice(N, size=k, replace=False)
        h[idx] = rng.uniform(0.5, 2.0, size=k)
        starts.append(h)
    # Dense random
    for _ in range(8):
        starts.append(rng.exponential(scale=1.0, size=N))
    # Structured
    x = np.linspace(0, N-1, N)
    for sigma in [2, 4, 6, 10]:
        starts.append(np.exp(-0.5 * ((x - N//2) / sigma)**2))
    # Alternating
    alt = np.zeros(N)
    alt[::2] = 1.0
    starts.append(alt)
    # Block
    for blk in [5, 10, 25]:
        h = np.zeros(N)
        h[:blk] = 1.0
        starts.append(h)
    # From spectral best
    starts.append(best_h.copy())
    
    for h0 in starts:
        if time.time() > deadline - 5:
            break
        try:
            h_cd = _coordinate_descent(h0, max_iter=150)
            r = _ratio(h_cd)
            if r > best_score:
                best_score = r
                best_h = h_cd
        except Exception:
            pass
    
    # Final refinement
    if time.time() < deadline - 3:
        try:
            h_cd = _coordinate_descent(best_h, max_iter=200)
            r = _ratio(h_cd)
            if r > best_score:
                best_score = r
                best_h = h_cd
        except Exception:
            pass
    
    best_h = np.maximum(best_h, 0.0)
    return list(best_h)
