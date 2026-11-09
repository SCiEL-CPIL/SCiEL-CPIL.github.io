"""Greedy covering-code search with parameterized heuristic and deadline guard."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds
import itertools

_DEADLINE = 120.0  # seconds; evaluator kills at 150s, keep safety margin


def _idx_to_word(idx, n, q):
    """Convert a linear index to a word over {0,1,...} with n digits (base q)."""
    digits = [0] * n
    for i in range(n):
        digits[i] = idx % q
        idx //= q
    return digits


def _word_to_idx(word, q, n):
    """Convert a word to a linear index."""
    idx = 0
    for i in range(n):
        idx = idx * q + word[i]
    return idx


def _hamming(a, b):
    return sum(1 for x, y in zip(a, b) if x != y)


def _build_distance_matrix(words, q, n, radius):
    """
    Build a boolean matrix A where A[i, j] is True if word i is within
    Hamming distance radius of word j.
    words: list of lists
    """
    N = len(words)
    # Use numpy for vectorized distance calculation if possible, 
    # but for N=729, a simple loop might be faster than overhead if not optimized.
    # However, we can vectorize by broadcasting.
    words_arr = np.array(words, dtype=np.int8)
    # dist_matrix[i, j] = hamming(words[i], words[j])
    # This creates a (N, N, n) array which is 729*729*6 ~ 3.2M bytes. Very small.
    diff = words_arr[:, np.newaxis, :] != words_arr[np.newaxis, :, :]
    dists = diff.sum(axis=2)
    # A[i, j] = 1 if dists[i, j] <= radius
    A = (dists <= radius).astype(np.int8)
    return A


def _solve_milp(words, q, n, radius, time_limit=100):
    """
    Solve the set covering problem using MILP.
    Minimize sum(x_j) subject to sum_j(A[i, j] * x_j) >= 1 for all i.
    """
    N = len(words)
    A = _build_distance_matrix(words, q, n, radius)
    
    # Objective: minimize sum(x_j)
    c = np.ones(N)
    
    # Constraints: A @ x >= 1
    # LinearConstraint(A, lb=1, ub=np.inf)
    # Note: milp in scipy expects constraints in form lb <= A @ x <= ub
    # So we use lb=1, ub=np.inf
    cons = LinearConstraint(A, lb=1, ub=np.inf)
    
    # Bounds: x_j in {0, 1}
    # milp uses integrality to specify integer variables.
    # Bounds are 0 <= x_j <= 1
    integrality = np.ones(N)
    bounds = Bounds(lb=0, ub=1)
    
    # Solve
    try:
        res = milp(c=c, constraints=cons, integrality=integrality, bounds=bounds,
                   options={"time_limit": time_limit, "mip_rel_gap": 0.0})
        
        if res.status == 0: # Optimal
            # res.x contains the solution
            x = res.x.astype(int)
            code = [words[j] for j in range(N) if x[j] == 1]
            return code
        else:
            return None
    except Exception:
        return None


def _greedy_fallback(words, q, n, radius):
    """
    Fast greedy heuristic to produce a valid (though possibly suboptimal) code.
    """
    N = len(words)
    words_arr = np.array(words, dtype=np.int8)
    covered = np.zeros(N, dtype=bool)
    code = []
    
    # Precompute distances? Or compute on fly.
    # For N=729, precomputing dist matrix is fast.
    diff = words_arr[:, np.newaxis, :] != words_arr[np.newaxis, :, :]
    dists = diff.sum(axis=2)
    
    while not covered.all():
        # Find candidate that covers most uncovered
        # new_covered[i] = sum(dists[i, j] <= radius and not covered[j])
        # We can use matrix multiplication or logical ops.
        # mask of uncovered
        un_cov_mask = ~covered
        # For each candidate i, count how many j it covers that are currently uncovered
        # valid[i, j] = dists[i, j] <= radius
        valid = (dists <= radius)
        # new_counts[i] = sum(valid[i, :] & un_cov_mask)
        new_counts = (valid & un_cov_mask[np.newaxis, :]).sum(axis=1)
        
        # Pick best
        best_idx = np.argmax(new_counts)
        if new_counts[best_idx] == 0:
            break # Should not happen if code is valid, but safety
            
        code.append(list(words[best_idx]))
        covered |= valid[best_idx, :]
        
    return code


def _verify_code(code, q, n, radius):
    """Verify that every word is within radius of some codeword."""
    total = q ** n
    if total > 50000:
        # Sampling for very large spaces (not expected here for q=3, n=6)
        rng = np.random.default_rng(42)
        for _ in range(10000):
            w = list(rng.integers(0, q, size=n))
            found = False
            for cw in code:
                if _hamming(w, cw) <= radius:
                    found = True
                    break
            if not found:
                return False
        return True
    else:
        # Exact check
        words_list = [_idx_to_word(i, n, q) for i in range(total)]
        words_arr = np.array(words_list, dtype=np.int8)
        code_arr = np.array(code, dtype=np.int8)
        
        # Check if every row in words_arr is covered by some row in code_arr
        # dists[i, j] <= radius for some j
        diff = words_arr[:, np.newaxis, :] != code_arr[np.newaxis, :, :]
        dists = diff.sum(axis=2)
        covered = (dists <= radius).any(axis=1)
        return covered.all()


def _direct_sum_tetracode(q, n, radius):
    """Direct sum of ternary tetracode with all tails."""
    tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
    if n == 4:
        return [list(h) for h in tetracode]
    if n < 4:
        # Fallback for small n
        return [[0] * n]
    return [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]


def construct(q, n, radius):
    """Construct a covering code: choose as few words as possible so every word
    is within Hamming distance radius of some chosen word."""
    start_time = time.time()
    time_limit = _DEADLINE - (start_time - start_time) - 10 # 10s safety margin
    
    # Generate all words
    total = q ** n
    if total > 100000:
        # Too large for exact MILP or greedy with full matrix in reasonable time?
        # But for q=3, n=6, total=729. 
        # If total is huge, we might need a different strategy, but the task implies small n.
        # Let's assume total is manageable.
        pass
        
    words_list = [_idx_to_word(i, n, q) for i in range(total)]
    
    # 1. Try MILP
    # Adjust time limit based on remaining time
    elapsed = time.time() - start_time
    remaining = _DEADLINE - elapsed - 5 # 5s buffer for verification and return
    
    if remaining > 5:
        try:
            code = _solve_milp(words_list, q, n, radius, time_limit=remaining)
            if code is not None and len(code) > 0:
                # Verify just in case
                if _verify_code(code, q, n, radius):
                    return code
        except Exception:
            pass
    
    # 2. Fallback to Greedy
    elapsed = time.time() - start_time
    remaining = _DEADLINE - elapsed - 5
    
    if remaining > 1:
        try:
            code = _greedy_fallback(words_list, q, n, radius)
            if _verify_code(code, q, n, radius):
                return code
        except Exception:
            pass
            
    # 3. Fallback to Tetracode construction
    try:
        code = _direct_sum_tetracode(q, n, radius)
        if _verify_code(code, q, n, radius):
            return code
    except Exception:
        pass
        
    # 4. Last resort: All words (valid but huge)
    return words_list

# EVOLVE-BLOCK-END
