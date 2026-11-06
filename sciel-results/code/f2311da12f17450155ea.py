import math
import time
from math import gcd

def _primes_up_to(limit):
    sieve = [True] * (limit + 1)
    sieve[0] = sieve[1] = False
    for i in range(2, int(limit**0.5) + 1):
        if sieve[i]:
            for j in range(i*i, limit + 1, i):
                sieve[j] = False
    return [i for i, v in enumerate(sieve) if v]

_PRIMES = _primes_up_to(1000)
_PRIME_SET = set(_PRIMES)

def _tau_perp(primes, exponents):
    """Compute tau_perp score from prime factorization using bitmask for coprime check."""
    k = len(primes)
    if k == 0:
        return 0
    # Generate all divisors with prime-support bitmask
    divisors = [(1, 0)]
    for i, (p, e) in enumerate(zip(primes, exponents)):
        if e == 0:
            continue
        current = list(divisors)
        pk = 1
        for _ in range(e):
            pk *= p
            for d, m in current:
                divisors.append((d * pk, m | (1 << i)))
    divisors.sort(key=lambda x: x[0])
    if len(divisors) < 2:
        return 0
    count = 0
    for i in range(len(divisors) - 1):
        if (divisors[i][1] & divisors[i+1][1]) == 0:
            count += 1
    return count

def _compute_n(primes, exponents):
    n = 1
    for p, e in zip(primes, exponents):
        if e > 0:
            n *= p ** e
    return n

def construct():
    start_time = time.time()
    deadline = start_time + 118.0
    
    max_n = 10**18
    
    # Start with best known primorial
    best_primes = [2, 3, 5, 7, 11, 13, 17, 19]
    best_exps = [1] * len(best_primes)
    best_score = _tau_perp(best_primes, best_exps)
    best_n = _compute_n(best_primes, best_exps)
    
    def try_candidate(cand_primes, cand_exps):
        nonlocal best_score, best_n, best_primes, best_exps
        n = _compute_n(cand_primes, cand_exps)
        if n < 2 or n > max_n:
            return False
        s = _tau_perp(cand_primes, cand_exps)
        if s > best_score:
            best_score = s
            best_n = n
            best_primes = list(cand_primes)
            best_exps = list(cand_exps)
            return True
        return False
    
    # Try primorials of various sizes
    prod = 1
    for i in range(min(20, len(_PRIMES))):
        if prod * _PRIMES[i] > max_n:
            break
        prod *= _PRIMES[i]
        exps = [1] * (i + 1)
        try_candidate(_PRIMES[:i+1], exps)
    
    # Local search
    for iteration in range(300):
        if time.time() > deadline:
            break
        
        improved = False
        cur_primes = best_primes
        cur_exps = best_exps
        cur_n = best_n
        
        # Move 1: Swap a prime - only try primes within a reasonable range
        for idx in range(len(cur_primes)):
            if time.time() > deadline:
                break
            old_p = cur_primes[idx]
            e = cur_exps[idx]
            # Only try primes within factor of 10 of old_p
            lo = max(2, old_p // 10)
            hi = old_p * 10
            for new_p in _PRIMES:
                if new_p < lo or new_p > hi:
                    continue
                if new_p == old_p or new_p in cur_primes:
                    continue
                # Compute new_n exactly
                base = cur_n // (old_p ** e)
                new_n = base * (new_p ** e)
                if new_n < 2 or new_n > max_n:
                    continue
                cand_primes = list(cur_primes)
                cand_exps = list(cur_exps)
                cand_primes[idx] = new_p
                if try_candidate(cand_primes, cand_exps):
                    improved = True
                    cur_primes = best_primes
                    cur_exps = best_exps
                    cur_n = best_n
                    break
        
        # Move 2: Change exponent
        for idx in range(len(cur_primes)):
            if time.time() > deadline:
                break
            for new_e in range(0, 4):
                if new_e == cur_exps[idx]:
                    continue
                cand_primes = list(cur_primes)
                cand_exps = list(cur_exps)
                cand_exps[idx] = new_e
                if try_candidate(cand_primes, cand_exps):
                    improved = True
                    cur_primes = best_primes
                    cur_exps = best_exps
                    cur_n = best_n
                    break
        
        # Move 3: Add a prime
        for new_p in _PRIMES:
            if time.time() > deadline:
                break
            if new_p in cur_primes:
                continue
            if cur_n * new_p > max_n:
                continue
            cand_primes = list(cur_primes) + [new_p]
            cand_exps = list(cur_exps) + [1]
            if try_candidate(cand_primes, cand_exps):
                improved = True
                cur_primes = best_primes
                cur_exps = best_exps
                cur_n = best_n
                break
        
        # Move 4: Remove a prime
        for idx in range(len(cur_primes)):
            if time.time() > deadline:
                break
            if cur_exps[idx] == 0:
                continue
            cand_primes = list(cur_primes)
            cand_exps = list(cur_exps)
            cand_exps[idx] = 0
            if try_candidate(cand_primes, cand_exps):
                improved = True
                cur_primes = best_primes
                cur_exps = best_exps
                cur_n = best_n
                break
        
        if not improved:
            # Perturbation
            for _ in range(5):
                if time.time() > deadline:
                    break
                cand_primes = list(best_primes)
                cand_exps = list(best_exps)
                if not cand_primes:
                    continue
                idx = (iteration * 7 + _ * 13) % len(cand_primes)
                new_p = _PRIMES[(iteration * 31 + _ * 17) % len(_PRIMES)]
                if new_p in cand_primes:
                    continue
                old_p = cand_primes[idx]
                e = cand_exps[idx]
                base = best_n // (old_p ** e)
                new_n = base * (new_p ** e)
                if new_n < 2 or new_n > max_n:
                    continue
                cand_primes[idx] = new_p
                if try_candidate(cand_primes, cand_exps):
                    break
    
    return best_n
