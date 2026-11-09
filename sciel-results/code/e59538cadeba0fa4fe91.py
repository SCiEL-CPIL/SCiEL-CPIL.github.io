"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math

_SMALL_PRIMES = [2, 3, 5, 7, 11, 13, 17, 19, 23]
_SMALL_PRIME_SET = set(_SMALL_PRIMES)

def _extract_small_factor(m):
    """Extract the part of m composed of primes strictly below 24."""
    result = 1
    for p in _SMALL_PRIMES:
        while m % p == 0:
            result *= p
            m //= p
    return result

def _evaluate(n):
    """Count distinct extracted factors for n+1..n+24."""
    factors = set()
    for i in range(1, 25):
        factors.add(_extract_small_factor(n + i))
    return len(factors)

def construct():
    start_time = time.time()
    deadline = start_time + 80.0  # safety margin before 90s kill
    
    best_n = 1000
    best_val = _evaluate(1000)
    
    # Phase 1: Scan a good chunk of the range
    # We want to minimize distinct factors. The minimum possible is 1
    # (all 24 numbers have the same small-prime factor part).
    # That's very unlikely. Let's aim for small counts.
    
    # Strategy: scan broadly, then refine around good candidates.
    # Use step to cover range quickly.
    
    # First, do a coarse scan with step 100 to find promising regions
    step = 100
    for n in range(1000, 1000001, step):
        if time.time() > deadline:
            break
        val = _evaluate(n)
        if val < best_val:
            best_val = val
            best_n = n
    
    # Phase 2: Fine scan around best region
    if best_val < 10:  # if we found something good
        lo = max(1000, best_n - 500)
        hi = min(1000000, best_n + 500)
        for n in range(lo, hi + 1):
            if time.time() > deadline:
                break
            val = _evaluate(n)
            if val < best_val:
                best_val = val
                best_n = n
    
    # Phase 3: If still high, do a denser scan
    if best_val > 5:
        # Try a medium-step scan over more of the range
        step2 = 10
        for n in range(1000, 1000001, step2):
            if time.time() > deadline:
                break
            val = _evaluate(n)
            if val < best_val:
                best_val = val
                best_n = n
                # Refine locally
                if time.time() < deadline - 5:
                    lo = max(1000, best_n - 20)
                    hi = min(1000000, best_n + 20)
                    for m in range(lo, hi + 1):
                        if time.time() > deadline:
                            break
                        v2 = _evaluate(m)
                        if v2 < best_val:
                            best_val = v2
                            best_n = m
    
    # Phase 4: Final local refinement around best
    if time.time() < deadline - 2:
        lo = max(1000, best_n - 50)
        hi = min(1000000, best_n + 50)
        for n in range(lo, hi + 1):
            if time.time() > deadline:
                break
            val = _evaluate(n)
            if val < best_val:
                best_val = val
                best_n = n
    
    return best_n
# EVOLVE-BLOCK-END
