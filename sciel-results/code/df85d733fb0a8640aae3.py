"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

N = 400

# EVOLVE-BLOCK-START

def _autoconv_fft(x):
    """Compute linear autoconvolution (full mode) via FFT."""
    n = len(x)
    n_fft = 1 << (2 * n - 1).bit_length()
    fx = np.fft.rfft(x, n_fft)
    ac = np.fft.irfft(fx * fx, n_fft)[:2 * n - 1]
    return ac


def _score(x):
    """Compute max |autoconv| / (sum x)^2."""
    integral = np.sum(x)
    if abs(integral) < 1e-12:
        return float('inf')
    ac = _autoconv_fft(x)
    return np.max(np.abs(ac)) / (integral * integral)


def _score_and_ac(x):
    """Return (score, autoconv array, integral)."""
    integral = np.sum(x)
    if abs(integral) < 1e-12:
        return float('inf'), None, integral
    ac = _autoconv_fft(x)
    score = np.max(np.abs(ac)) / (integral * integral)
    return score, ac, integral


def _greedy_flip(x, ac, integral, max_passes=50, time_budget=10.0):
    """
    Greedy sign-flip local search.
    For each position j, compute the change in autoconv if we flip x[j].
    Only accept if it reduces the max |ac|.
    """
    start = time.time()
    n = len(x)
    current_score, _, _ = _score_and_ac(x)
    if not np.isfinite(current_score):
        return x, current_score
    
    for _pass in range(max_passes):
        if time.time() - start > time_budget:
            break
        improved = False
        for j in range(n):
            if time.time() - start > time_budget:
                break
            # Flipping x[j] changes ac by -2*x[j]*x shifted by j
            # ac_new[k] = ac[k] - 2*x[j]*x[k-j] for valid k
            # Actually, autoconv ac[k] = sum_i x[i]*x[k-i]
            # If we flip x[j] to -x[j], the change in ac is:
            # delta_ac[k] = -2*x[j]*x[k-j] (for k where both indices valid)
            # More precisely, ac[k] involves x[i]*x[k-i]. 
            # The terms involving x[j] are: x[j]*x[k-j] and x[k-j]*x[j] if j != k-j
            # Actually ac[k] = sum_i x[i]*x[k-i]. The term with i=j is x[j]*x[k-j].
            # The term with i=k-j is x[k-j]*x[j]. These are the same if we're summing over all i.
            # Wait, no. ac[k] = sum_{i} x[i] * x[k-i]. 
            # If we change x[j] to -x[j], then for each k, the terms that involve x[j] are:
            #   i = j: x[j]*x[k-j] -> -x[j]*x[k-j], change = -2*x[j]*x[k-j]
            #   i = k-j: x[k-j]*x[j] -> x[k-j]*(-x[j]), change = -2*x[k-j]*x[j]
            # But wait, if j == k-j (i.e., k=2j), then both terms are the same term x[j]*x[j], 
            # and flipping changes it from x[j]^2 to x[j]^2 (since (-x[j])^2 = x[j]^2). 
            # So for k=2j, no change.
            # For k != 2j, the two terms i=j and i=k-j are distinct, so total change is -4*x[j]*x[k-j]?
            # No wait. Let me re-derive.
            # ac[k] = sum_i x[i] * x[k-i]
            # Terms involving x[j]: when i=j, term is x[j]*x[k-j]. When i=k-j, term is x[k-j]*x[j].
            # If j != k-j, these are two separate terms in the sum.
            # After flipping x[j] -> -x[j]:
            #   i=j term: -x[j]*x[k-j], change: -2*x[j]*x[k-j]
            #   i=k-j term: x[k-j]*(-x[j]), change: -2*x[k-j]*x[j]
            # Total change for k (j != k-j): -4*x[j]*x[k-j]
            # For j == k-j (k=2j): only one term x[j]*x[j], flipping gives (-x[j])^2 = x[j]^2, no change.
            
            # So delta_ac[k] = -4*x[j]*x[k-j] for k != 2j, 0 for k=2j.
            # But we need to be careful about bounds. k ranges from 0 to 2n-2.
            # x[k-j] is valid when 0 <= k-j < n, i.e., j <= k <= j+n-1.
            
            # Let's compute the new max efficiently.
            # new_ac = ac.copy()
            # For k in range(max(0, j), min(2*n-1, j+n)):
            #     if k != 2*j:
            #         new_ac[k] -= 4 * x[j] * x[k-j]
            
            # This is O(n) per flip, O(n^2) per pass, O(n^2 * passes) total.
            # With n=400, passes=50, that's 400*400*50 = 8M operations, fast enough.
            
            new_ac = ac.copy()
            lo = max(0, j)
            hi = min(2 * n - 1, j + n)
            for k in range(lo, hi):
                if k != 2 * j:
                    new_ac[k] -= 4.0 * x[j] * x[k - j]
            
            new_max = np.max(np.abs(new_ac))
            new_score = new_max / (integral * integral)
            
            if new_score < current_score - 1e-12:
                x[j] = -x[j]
                ac = new_ac
                current_score = new_score
                improved = True
        
        if not improved:
            break
    
    return x, current_score


def _random_search(n_trials=2000, time_budget=60.0):
    """Random search with greedy refinement."""
    start = time.time()
    best_x = None
    best_score = float('inf')
    
    for trial in range(n_trials):
        if time.time() - start > time_budget:
            break
        
        x = np.random.choice([-1.0, 1.0], size=N)
        sc, ac, intg = _score_and_ac(x)
        
        if sc < best_score:
            best_score = sc
            best_x = x.copy()
        
        # Refine with greedy
        if time.time() - start < time_budget - 2.0:
            refined_x, refined_score = _greedy_flip(x, ac, intg, max_passes=20, time_budget=1.0)
            if refined_score < best_score:
                best_score = refined_score
                best_x = refined_x.copy()
    
    return best_x, best_score


def _structured_initial():
    """Generate a few structured starting points."""
    candidates = []
    
    # All ones
    candidates.append(np.ones(N))
    
    # Alternating
    candidates.append(np.tile([1.0, -1.0], N // 2))
    
    # Random
    for _ in range(20):
        candidates.append(np.random.choice([-1.0, 1.0], size=N))
    
    return candidates


def construct():
    """Construct a signed step function minimizing the maximum absolute autoconvolution 
    divided by squared integral."""
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s limit
    
    best_x = None
    best_score = float('inf')
    
    # Phase 1: Try structured and random initial points
    for x in _structured_initial():
        if time.time() > deadline - 30:
            break
        sc, ac, intg = _score_and_ac(x)
        if sc < best_score:
            best_score = sc
            best_x = x.copy()
    
    # Phase 2: Random search with greedy refinement
    if time.time() < deadline - 20:
        remaining = min(60.0, deadline - time.time() - 10.0)
        x_rand, score_rand = _random_search(n_trials=5000, time_budget=remaining)
        if x_rand is not None and score_rand < best_score:
            best_score = score_rand
            best_x = x_rand.copy()
    
    # Phase 3: Final intensive greedy refinement
    if time.time() < deadline - 5 and best_x is not None:
        sc, ac, intg = _score_and_ac(best_x)
        refined_x, refined_score = _greedy_flip(best_x, ac, intg, max_passes=100, 
                                                 time_budget=min(20.0, deadline - time.time() - 3.0))
        if refined_score < best_score:
            best_score = refined_score
            best_x = refined_x.copy()
    
    return best_x

# EVOLVE-BLOCK-END
