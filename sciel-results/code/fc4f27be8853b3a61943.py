"""Optimized program: sparse signed sequence with local sign-flip optimization."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.signal import fftconvolve

_START_TIME = time.time()
_TIME_LIMIT = 115.0  # safety margin before 120s (evaluator kills at 150s, but we want buffer)


def _autoconv_max_abs(h):
    """Compute max |autoconvolution| of h."""
    ac = fftconvolve(h, h, mode='full')
    return np.max(np.abs(ac))


def _squared_integral(h):
    """Compute (sum h)^2."""
    s = np.sum(h)
    return s * s


def _score(h):
    """Compute the objective: max|autoconv| / (sum h)^2. Lower is better."""
    si = _squared_integral(h)
    if si == 0:
        return np.inf
    return _autoconv_max_abs(h) / si


def _local_sign_flip_optimize(h, max_rounds=50, time_budget=None):
    """Iteratively flip signs of individual elements to reduce the objective."""
    best_h = h.copy()
    best_score = _score(best_h)
    
    for _ in range(max_rounds):
        if time_budget is not None:
            if time.time() - _START_TIME > time_budget:
                break
        
        improved = False
        # Try flipping each position
        for i in range(len(best_h)):
            if time_budget is not None:
                if time.time() - _START_TIME > time_budget:
                    return best_h, best_score
            
            best_h[i] *= -1
            new_score = _score(best_h)
            if new_score < best_score:
                best_score = new_score
                improved = True
            else:
                best_h[i] *= -1  # revert
        
        if not improved:
            break
    
    return best_h, best_score


def construct():
    """Construct a signed step function minimizing max|autoconv|/(sum h)^2."""
    # Strategy: sparse signed sequence
    # Use a subset of positions with alternating signs to reduce peak autoconvolution
    
    # Start with a sparse pattern: select ~50% of positions
    # Use a structured pattern to avoid large autoconvolution peaks
    
    # Initialize with a random sparse signed sequence
    np.random.seed(42)
    
    # Try different sparsity levels and patterns
    best_h = None
    best_score = np.inf
    
    # Pattern 1: Random sparse with 50% density
    for trial in range(5):
        if time.time() - _START_TIME > _TIME_LIMIT:
            break
        
        if trial == 0:
            # Dense alternating
            h = np.ones(N)
            h[1::2] = -1
        elif trial == 1:
            # Sparse: only even positions
            h = np.zeros(N)
            h[0::2] = 1
            h[2::4] = -1  # alternate among selected
        elif trial == 2:
            # Random sparse 50%
            mask = np.random.rand(N) < 0.5
            h = np.zeros(N)
            h[mask] = np.random.choice([-1, 1], size=np.sum(mask))
        elif trial == 3:
            # Sparse 30% with random signs
            mask = np.random.rand(N) < 0.3
            h = np.zeros(N)
            h[mask] = np.random.choice([-1, 1], size=np.sum(mask))
        else:
            # Block pattern: +1+1-1-1 repeating
            h = np.tile(np.array([1, 1, -1, -1]), N // 4 + 1)[:N]
        
        h, sc = _local_sign_flip_optimize(h, max_rounds=30, time_budget=_TIME_LIMIT - 5)
        
        if sc < best_score:
            best_score = sc
            best_h = h.copy()
    
    # Final pass with more rounds if time allows
    if best_h is not None and time.time() - _START_TIME < _TIME_LIMIT - 10:
        best_h, best_score = _local_sign_flip_optimize(
            best_h, max_rounds=100, time_budget=_TIME_LIMIT - 5
        )
    
    if best_h is None:
        best_h = np.ones(N)
    
    return best_h
# EVOLVE-BLOCK-END
