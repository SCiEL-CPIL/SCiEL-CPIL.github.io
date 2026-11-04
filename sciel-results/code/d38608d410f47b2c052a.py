"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.fft import fft, ifft


def _autoconv(x):
    """Full autoconvolution via FFT (circular, zero-padded to 2N-1)."""
    n = len(x)
    size = 2 * n - 1
    # next power of 2 for speed
    size = 1 << (size - 1).bit_length()
    X = fft(x, size)
    ac = ifft(X * X).real[:2 * n - 1]
    return ac


def _score(x):
    """C3' = max |autoconv| / (integral)^2. Lower is better."""
    ac = _autoconv(x)
    integral = x.sum()
    if integral == 0:
        return float('inf')
    return np.max(np.abs(ac)) / (integral * integral)


def _flip(x, i):
    """Flip sign at position i."""
    x[i] = -x[i]


def _try_improve(x, max_iter=200000, deadline=None):
    """Local search: flip signs to reduce max |autoconv| / integral^2."""
    n = len(x)
    best_x = x.copy()
    best_score = _score(best_x)
    
    # Precompute autoconvolution
    ac = _autoconv(best_x)
    integral = best_x.sum()
    integral_sq = integral * integral
    
    # For a flip at position i, the new autoconvolution changes:
    # new_ac[k] = old_ac[k] - 2*x[i]*sum_{j} x[j]*delta(k - i - j) - 2*x[i]*sum_{j} x[j]*delta(k - i + j)
    # This is complex; instead use direct update for small changes.
    # Actually, for binary ±1, flipping x[i] from s to -s changes autoconv by:
    # Δac[k] = -2*s * (sum_j x[j] * [k == i+j] + sum_j x[j] * [k == i-j])
    #        = -2*s * (x[k-i] + x[i-k]) with appropriate bounds
    # Wait, autoconv[k] = sum_j x[j]*x[k-j]. 
    # If we flip x[i], new_x[i] = -x[i].
    # new_ac[k] = sum_j new_x[j]*new_x[k-j]
    # The terms involving x[i] are: x[i]*x[k-i] + x[k-i]*x[i] (if i != k-i) or 2*x[i]^2 (if i == k-i)
    # Actually: sum_j x[j]*x[k-j] includes term j=i: x[i]*x[k-i] and term j=k-i: x[k-i]*x[i]
    # If i != k-i, both terms are present: 2*x[i]*x[k-i]
    # If i == k-i (i.e., 2i=k), only one term: x[i]^2
    # After flip: x[i] -> -x[i]
    # New contribution from j=i: (-x[i])*x[k-i] = -x[i]*x[k-i]
    # New contribution from j=k-i: x[k-i]*(-x[i]) = -x[i]*x[k-i]
    # So change = -2*x[i]*x[k-i] - 2*x[i]*x[k-i] = -4*x[i]*x[k-i] if i != k-i
    # If i == k-i: old = x[i]^2, new = (-x[i])^2 = x[i]^2, change = 0
    # Wait that's not right. Let me redo.
    
    # ac[k] = sum_j x[j] * x[k-j]
    # Terms where x[i] appears: j=i gives x[i]*x[k-i], j=k-i gives x[k-i]*x[i]
    # These are the same term if i = k-i, i.e., k=2i.
    # So:
    # if k != 2i: ac[k] includes 2*x[i]*x[k-i] (from j=i and j=k-i)
    #   after flip: 2*(-x[i])*x[k-i] = -2*x[i]*x[k-i]
    #   delta = -4*x[i]*x[k-i]
    # if k == 2i: ac[k] includes x[i]^2 (only one term since j=i=k-i)
    #   after flip: (-x[i])^2 = x[i]^2
    #   delta = 0
    
    # Also integral changes: new_integral = integral - 2*x[i]
    
    # For efficiency, let's just compute the new score for candidate flips using the formula.
    
    iter_count = 0
    no_improve = 0
    while iter_count < max_iter:
        if deadline is not None and time.time() > deadline:
            break
        
        best_delta = 0
        best_pos = -1
        # Sample a subset of positions for speed
        # For N=400, we can try all positions
        for i in range(n):
            if deadline is not None and time.time() > deadline:
                break
            xi = best_x[i]
            # Compute new integral
            new_int = integral - 2 * xi
            new_int_sq = new_int * new_int
            if new_int_sq == 0:
                continue
            
            # Compute new max |ac|
            # We need to update ac for all k
            # This is O(n) per flip, so O(n^2) per iteration - too slow
            # Instead, let's just compute the full autoconv for the candidate
            # But that's also O(n log n) per flip...
            # For N=400, let's just do direct computation for the changed positions
            
            # Actually, let's use a different approach: compute the full new ac via FFT
            # but that's expensive. Let's limit to checking a few positions per iter.
            
            # Alternative: just compute new ac directly for this flip
            new_x = best_x.copy()
            new_x[i] = -xi
            new_ac = _autoconv(new_x)
            new_max = np.max(np.abs(new_ac))
            new_score_val = new_max / new_int_sq
            
            if new_score_val < best_score - 1e-12:
                best_delta = best_score - new_score_val
                best_pos = i
        
        if best_pos >= 0:
            _flip(best_x, best_pos)
            ac = _autoconv(best_x)
            integral = best_x.sum()
            integral_sq = integral * integral
            best_score = _score(best_x)
            no_improve = 0
        else:
            no_improve += 1
            if no_improve > 50:
                break
        
        iter_count += 1
    
    return best_x, best_score


def construct():
    deadline = time.time() + 110  # Safety margin
    
    # Start with random ±1
    np.random.seed(42)
    x = np.random.choice([-1, 1], size=N).astype(float)
    
    best_x = x.copy()
    best_score = _score(best_x)
    
    # Local search from initial random
    best_x, best_score = _try_improve(best_x, max_iter=50000, deadline=deadline)
    
    # Multiple random restarts
    restart = 0
    while time.time() < deadline - 5:
        restart += 1
        x = np.random.choice([-1, 1], size=N).astype(float)
        # Quick local search
        x, s = _try_improve(x, max_iter=10000, deadline=deadline)
        if s < best_score:
            best_score = s
            best_x = x
    
    return best_x
# EVOLVE-BLOCK-END
