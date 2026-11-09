"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from math import isqrt

_P = {'n': 40, 'maximum_size': 18, 'deadline': 60.0}


def _sieve_primes(n):
    """Return all primes in 2..n (both inclusive) via Sieve of Eratosthenes."""
    if n < 2:
        return []
    is_prime = [True] * (n + 1)
    is_prime[0] = is_prime[1] = False
    limit = isqrt(n)
    for i in range(2, limit + 1):
        if is_prime[i]:
            step = i
            start = i * i
            is_prime[start:n + 1:step] = [False] * ((n - start) // step + 1)
    return [i for i in range(2, n + 1) if is_prime[i]]


def _is_valid(set_):
    """Verify all 2**k subset products (incl. empty product = 1) are distinct."""
    a = sorted(set_)
    k = len(a)
    prods = [1]
    for x in a:
        prods += [p * x for p in prods]
    if len(prods) != len(set(prods)):
        return False
    return True


def _greedy_max_set(candidates, deadline, initial_best=None):
    """Greedy: try to build largest valid set by adding elements one by one."""
    best = initial_best if initial_best is not None else []
    # If we start with a good set, we try to add more
    for x in candidates:
        if time.time() > deadline:
            break
        if x in best:
            continue
        trial = best + [x]
        if _is_valid(trial):
            best = trial
    return best


def construct():
    """Find a large set in 2..40 with all distinct subset products."""
    t0 = time.time()
    deadline = t0 + _P['deadline'] - 5.0  # safety margin
    n = _P['n']

    candidates = list(range(2, n + 1))
    primes = _sieve_primes(n)
    
    # Known optimal structure:
    # Superincreasing powers of small primes (2, 3, 5, 7)
    # And all large primes (11-37)
    # 2: 2 (2^1), 4 (2^2), 16 (2^4). Next 2^8=256 > 40.
    # 3: 3 (3^1), 9 (3^2). Next 3^4=81 > 40.
    # 5: 5 (5^1), 25 (5^2). Next 5^4=625 > 40.
    # 7: 7 (7^1). Next 7^2=49 > 40.
    # Large primes: 11, 13, 17, 19, 23, 29, 31, 37.
    
    small_powers = [2, 4, 16, 3, 9, 5, 25, 7]
    large_primes = [p for p in primes if p >= 11]
    
    initial_best = small_powers + large_primes
    
    # Verify initial best
    if not _is_valid(initial_best):
        # Fallback to greedy if hardcoded set is somehow invalid (should not happen)
        initial_best = []

    # Try to improve (should fail, but ensures we explore if hardcoded set is suboptimal)
    # We pass the initial best to greedy to see if we can add more
    # Note: Greedy adds elements. It doesn't swap.
    # We should also try swaps.
    
    best = _greedy_max_set(candidates, deadline, initial_best=initial_best)
    
    # Local search: try to swap elements to increase size or maintain size with better structure
    # Since 16 is likely max, we just ensure we don't drop below 16.
    
    excluded = [x for x in candidates if x not in best]
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for ex in list(excluded):
            if time.time() > deadline:
                break
            for i, b in enumerate(best):
                trial = best[:i] + best[i+1:] + [ex]
                if _is_valid(trial):
                    # Only accept if it improves size (or same size, but we prefer stability)
                    if len(trial) > len(best):
                        best = trial
                        excluded = [x for x in candidates if x not in best]
                        improved = True
                        break

    # Final validation
    if _is_valid(best):
        return list(best)

    # Fallback
    return list(primes)
# EVOLVE-BLOCK-END
