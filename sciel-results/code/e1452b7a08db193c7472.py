import time
import random
import itertools

_DEADLINE = 85.0  # safety margin before 90s kill
_START = time.time()

def _elapsed():
    return time.time() - _START

def _square_nonzero_count(coeffs):
    """Return number of nonzero entries in the exact square of coeffs (ascending degree)."""
    n = len(coeffs)
    # convolution
    size = 2 * n - 1
    sq = [0] * size
    for i in range(n):
        ai = coeffs[i]
        if ai == 0:
            continue
        for j in range(i, n):
            aj = coeffs[j]
            if aj == 0:
                continue
            s = 2 * ai * aj
            if i == j:
                sq[i + j] += s
            else:
                sq[i + j] += s
                sq[j + i] += s
    # trim trailing zeros
    while len(sq) > 1 and sq[-1] == 0:
        sq.pop()
    return sum(1 for v in sq if v != 0), sq

def _validate(coeffs):
    if len(coeffs) > 41:
        return False
    if len(coeffs) == 0 or coeffs[-1] == 0:
        return False
    for v in coeffs:
        if not isinstance(v, int) or v < -64 or v > 64:
            return False
    nz = sum(1 for v in coeffs if v != 0)
    if nz != 12:
        return False
    return True

def _score(coeffs):
    if not _validate(coeffs):
        return None
    cnt, _ = _square_nonzero_count(coeffs)
    return cnt

def _random_exponents(rng):
    """Generate a random sorted list of 12 exponents in [0, 40]."""
    exps = sorted(rng.sample(range(0, 41), 12))
    return exps

def _build_coeffs(exps, coeffs_at_positions):
    """Place given coefficients at given exponent positions; return full coeff list."""
    max_deg = max(exps)
    arr = [0] * (max_deg + 1)
    for e, c in zip(exps, coeffs_at_positions):
        arr[e] = c
    return arr

def _local_search_coeffs(exps, rng, max_iters=200):
    """Hill-climb on coefficients for fixed exponents to minimize square sparsity."""
    # start with random small coefficients
    best_coeffs = [rng.choice([-2, -1, 1, 2]) for _ in range(12)]
    best_arr = _build_coeffs(exps, best_coeffs)
    best_score, _ = _square_nonzero_count(best_arr)
    
    for _ in range(max_iters):
        if _elapsed() > _DEADLINE:
            break
        # pick a random position to perturb
        pos = rng.randrange(12)
        old_val = best_coeffs[pos]
        # try a few new values
        candidates = [v for v in [-4, -3, -2, -1, 1, 2, 3, 4] if v != old_val]
        rng.shuffle(candidates)
        improved = False
        for new_val in candidates:
            best_coeffs[pos] = new_val
            arr = _build_coeffs(exps, best_coeffs)
            sc, _ = _square_nonzero_count(arr)
            if sc < best_score:
                best_score = sc
                improved = True
                break
        if not improved:
            best_coeffs[pos] = old_val
    
    return best_arr, best_score

def construct():
    rng = random.Random(42)
    best_score = 999
    best_coeffs = None
    
    # Start with a known-good baseline
    baseline = [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    # Actually let's just use 12 ones at positions 0..11
    baseline = [1]*12
    sc, _ = _square_nonzero_count(baseline)
    best_score = sc
    best_coeffs = baseline
    
    # Try structured exponent patterns
    structured_patterns = [
        # Arithmetic progression
        list(range(12)),
        # Two clusters
        [0,1,2,3,4,5,20,21,22,23,24,25],
        # Symmetric
        [0,1,2,3,4,5,6,7,8,9,10,11],
        # Sparse
        [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22],
        # Dense then sparse
        [0,1,2,3,4,5,10,20,30,35,38,40],
        # Pair structure for collisions
        [0,1,2,3,4,5,0,1,2,3,4,5],  # duplicate exponents won't work, use distinct
    ]
    
    # Use distinct exponents only
    structured_patterns = [
        list(range(12)),
        [0,1,2,3,4,5,20,21,22,23,24,25],
        [0,2,4,6,8,10,12,14,16,18,20,22],
        [0,1,2,3,4,5,10,20,30,35,38,40],
        [0,1,2,3,4,5,6,7,8,9,10,40],
        [0,1,3,7,12,18,24,30,34,36,38,40],
    ]
    
    for exps in structured_patterns:
        if _elapsed() > _DEADLINE:
            break
        if len(set(exps)) != 12:
            continue
        if max(exps) > 40:
            continue
        arr, sc = _local_search_coeffs(exps, rng, max_iters=100)
        if sc < best_score:
            best_score = sc
            best_coeffs = arr
        if best_score <= 13:
            break
    
    # Randomized search over exponent sets
    max_random_trials = 2000
    for trial in range(max_random_trials):
        if _elapsed() > _DEADLINE:
            break
        exps = _random_exponents(rng)
        arr, sc = _local_search_coeffs(exps, rng, max_iters=50)
        if sc < best_score:
            best_score = sc
            best_coeffs = arr
        if best_score <= 13:
            break
    
    # Ensure final validity
    if not _validate(best_coeffs):
        best_coeffs = [1]*12
    
    return best_coeffs
