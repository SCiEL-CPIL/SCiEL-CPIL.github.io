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


def _greedy_max_set(candidates, deadline):
    """Greedy: try to build largest valid set by adding elements one by one."""
    best = []
    for x in candidates:
        if time.time() > deadline:
            break
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
    composites = [x for x in candidates if x not in primes]

    # Strategy 1: Greedy with primes first, then composites
    order1 = primes + composites
    best = _greedy_max_set(order1, deadline)

    # Strategy 2: Different ordering - larger numbers first
    order2 = sorted(candidates, reverse=True)
    best2 = _greedy_max_set(order2, deadline)
    if len(best2) > len(best):
        best = best2

    # Strategy 3: Primes in reverse, then composites
    order3 = sorted(primes, reverse=True) + sorted(composites, reverse=True)
    best3 = _greedy_max_set(order3, deadline)
    if len(best3) > len(best):
        best = best3

    # Strategy 4: Try swapping - for each element in best, try to add excluded ones
    # and if that fails, try removing a different element
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
                    best = trial
                    excluded = [x for x in candidates if x not in best]
                    improved = True
                    break

    # Final validation
    if _is_valid(best):
        return list(best)

    # Fallback: just the primes
    return list(primes)
# EVOLVE-BLOCK-END
