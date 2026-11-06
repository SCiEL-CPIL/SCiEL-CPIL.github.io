"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from math import gcd
from functools import lru_cache

def _is_prime(n):
    if n < 2:
        return False
    if n < 4:
        return True
    if n % 2 == 0 or n % 3 == 0:
        return False
    i = 5
    while i * i <= n:
        if n % i == 0 or n % (i + 2) == 0:
            return False
        i += 6
    return True

def _score(set_a):
    return sum(1.0 / (512 - a) for a in set_a)

def construct():
    start_time = time.time()
    deadline = start_time + 115  # 5s safety margin
    
    # Strategy 1: All primes below 512
    primes = [p for p in range(2, 512) if _is_prime(p)]
    score_primes = _score(primes)
    best_set = primes
    best_score = score_primes
    
    # Strategy 2: Greedy from high to low
    chosen = []
    for a in range(511, 0, -1):
        if time.time() > deadline:
            break
        if all(gcd(a, b) == 1 for b in chosen):
            chosen.append(a)
    score_greedy = _score(chosen)
    if score_greedy > best_score:
        best_score = score_greedy
        best_set = chosen
    
    # Strategy 3: Greedy but skip some primes to allow composites
    # Try: for each prime p, consider not selecting p and see if we can add composites
    # This is complex, so let's try a different approach:
    # Start with all primes, then try to swap out small primes for larger composites
    
    # Actually, let's try: select primes in descending order, but allow composites
    # that are coprime to all selected so far, prioritizing by 1/(512-a)
    
    # Better: Use a priority queue approach
    # Candidates sorted by 1/(512-a) descending = a descending
    candidates = list(range(511, 0, -1))
    chosen2 = []
    for a in candidates:
        if time.time() > deadline:
            break
        if all(gcd(a, b) == 1 for b in chosen2):
            chosen2.append(a)
    score2 = _score(chosen2)
    if score2 > best_score:
        best_score = score2
        best_set = chosen2
    
    # The greedy from high to low should be same as strategy 2
    # Let's verify the prime-only strategy is best
    
    return best_set
# EVOLVE-BLOCK-END
