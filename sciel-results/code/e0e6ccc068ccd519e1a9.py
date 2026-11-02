"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math

def _score_set(U):
    """Compute the score for a given set U."""
    U = list(set(U))
    if 0 not in U:
        U.append(0)
    n = len(U)
    if n < 2:
        return 0.0
    max_u = max(U)
    if max_u == 0:
        return 0.0
    # Compute U+U
    sum_set = set()
    for a in U:
        for b in U:
            sum_set.add(a + b)
    # Compute U-U
    diff_set = set()
    for a in U:
        for b in U:
            diff_set.add(a - b)
    num = len(diff_set)
    den = len(sum_set)
    if den == 0:
        return 0.0
    ratio = num / den
    if ratio <= 0:
        return 0.0
    log_ratio = math.log(ratio)
    log_denom = math.log(2 * max_u + 1)
    if log_denom == 0:
        return 0.0
    return 1.0 + log_ratio / log_denom

def construct():
    start_time = time.time()
    deadline = start_time + 140  # safety margin
    
    best_set = [0, 1, 3]
    best_score = _score_set(best_set)
    
    # Try geometric progressions with various bases and lengths
    # U = {0, base, base^2, ..., base^(n-1)}
    for base in range(2, 20):
        if time.time() > deadline:
            break
        for n in range(2, 50):
            if time.time() > deadline:
                break
            max_val = base ** (n - 1)
            if max_val > 10**9:
                break
            U = [base ** i for i in range(n)]
            score = _score_set(U)
            if score > best_score:
                best_score = score
                best_set = U[:]
    
    # Try mixed constructions: combine different geometric blocks
    # U = {0} union {a_i * base^j} for various blocks
    # Simpler: try sets like {0, 1, b1, b1*b2, b1*b2*b3, ...}
    for n in range(2, 30):
        if time.time() > deadline:
            break
        # Try product of first n primes or similar
        U = [0]
        prod = 1
        for i in range(1, n):
            # Use rapidly growing sequence
            prod *= (2 * i + 1)
            U.append(prod)
        score = _score_set(U)
        if score > best_score:
            best_score = score
            best_set = U[:]
    
    # Try the specific construction from the paper style
    # Use a set where elements are chosen to maximize spread
    # Greedy: start with 0, then add elements that maximize score incrementally
    if time.time() < deadline - 30:
        U = [0]
        # Greedy construction
        for _ in range(50):
            if time.time() > deadline - 10:
                break
            best_candidate = None
            best_inc_score = best_score
            # Try adding elements up to some limit
            max_current = max(U) if U else 0
            for candidate in range(1, max_current + 100):
                if candidate in U:
                    continue
                U_test = U + [candidate]
                score = _score_set(U_test)
                if score > best_inc_score:
                    best_inc_score = score
                    best_candidate = candidate
            if best_candidate is not None:
                U.append(best_candidate)
                best_score = best_inc_score
                best_set = U[:]
    
    return best_set
# EVOLVE-BLOCK-END
