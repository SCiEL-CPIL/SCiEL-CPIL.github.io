"""Search for squarefree n maximizing coprime adjacent divisor pairs."""
import time
import math
from itertools import combinations

DEADLINE = 140.0
MAX_N = 10**18

_ALL_PRIMES = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97]

def _divisors_of_squarefree(primes):
    """Generate all divisors of a squarefree number given its prime list."""
    divs = [1]
    for p in primes:
        divs = divs + [d * p for d in divs]
    divs.sort()
    return divs

def _tau_perp(divs):
    """Count adjacent coprime pairs in sorted divisor list."""
    c = 0
    for i in range(1, len(divs)):
        if math.gcd(divs[i-1], divs[i]) == 1:
            c += 1
    return c

def _score_primes(primes):
    """Score for squarefree number with given primes."""
    n = 1
    for p in primes:
        n *= p
    if n > MAX_N:
        return -1, n
    divs = _divisors_of_squarefree(primes)
    return _tau_perp(divs), n

def construct():
    start = time.time()
    
    best_score = 0
    best_n = 2 * 3 * 5 * 7 * 11 * 13 * 17 * 19  # seed value
    
    def _time_ok():
        return time.time() - start < DEADLINE - 5
    
    # Evaluate the seed
    seed_primes = [2, 3, 5, 7, 11, 13, 17, 19]
    s, n = _score_primes(seed_primes)
    best_score = s
    best_n = n
    
    # Phase 1: Try primorials of first k primes for k=2..15
    for k in range(2, 16):
        if not _time_ok():
            break
        primes_k = _ALL_PRIMES[:k]
        n_prod = 1
        for p in primes_k:
            n_prod *= p
        if n_prod > MAX_N:
            break
        s, _ = _score_primes(primes_k)
        if s > best_score:
            best_score = s
            best_n = n_prod
    
    # Phase 2: Single swap on 14 and 15 prime bases
    for base_size in [14, 15]:
        if not _time_ok():
            break
        base = _ALL_PRIMES[:base_size]
        base_n = 1
        for p in base:
            base_n *= p
        if base_n > MAX_N:
            continue
        
        s, _ = _score_primes(base)
        if s > best_score:
            best_score = s
            best_n = base_n
        
        # Try replacing one prime with a larger unused one
        for i in range(base_size):
            for j in range(base_size, len(_ALL_PRIMES)):
                if not _time_ok():
                    break
                new_n = base_n // base[i] * _ALL_PRIMES[j]
                if new_n > MAX_N:
                    continue
                candidate = base[:i] + [_ALL_PRIMES[j]] + base[i+1:]
                candidate.sort()
                s2, _ = _score_primes(candidate)
                if s2 > best_score:
                    best_score = s2
                    best_n = new_n
    
    # Phase 3: Double swap on 14-prime base (more room for larger primes)
    base14 = _ALL_PRIMES[:14]
    base14_n = 1
    for p in base14:
        base14_n *= p
    
    if base14_n <= MAX_N:
        for i in range(14):
            for j in range(i + 1, 14):
                for k in range(14, len(_ALL_PRIMES)):
                    for l in range(k + 1, len(_ALL_PRIMES)):
                        if not _time_ok():
                            break
                        new_n = base14_n // base14[i] // base14[j] * _ALL_PRIMES[k] * _ALL_PRIMES[l]
                        if new_n > MAX_N:
                            continue
                        candidate = [p for idx, p in enumerate(base14) if idx != i and idx != j]
                        candidate.append(_ALL_PRIMES[k])
                        candidate.append(_ALL_PRIMES[l])
                        candidate.sort()
                        s2, _ = _score_primes(candidate)
                        if s2 > best_score:
                            best_score = s2
                            best_n = new_n
                    if not _time_ok():
                        break
                if not _time_ok():
                    break
            if not _time_ok():
                break
    
    # Phase 4: Triple swap on 13-prime base
    base13 = _ALL_PRIMES[:13]
    base13_n = 1
    for p in base13:
        base13_n *= p
    
    if base13_n <= MAX_N and _time_ok():
        for i in range(13):
            for j in range(i + 1, 13):
                for k in range(j + 1, 13):
                    for a in range(13, len(_ALL_PRIMES)):
                        for b in range(a + 1, len(_ALL_PRIMES)):
                            for c in range(b + 1, len(_ALL_PRIMES)):
                                if not _time_ok():
                                    break
                                new_n = base13_n // base13[i] // base13[j] // base13[k] * _ALL_PRIMES[a] * _ALL_PRIMES[b] * _ALL_PRIMES[c]
                                if new_n > MAX_N:
                                    continue
                                candidate = [p for idx, p in enumerate(base13) if idx != i and idx != j and idx != k]
                                candidate.append(_ALL_PRIMES[a])
                                candidate.append(_ALL_PRIMES[b])
                                candidate.append(_ALL_PRIMES[c])
                                candidate.sort()
                                s2, _ = _score_primes(candidate)
                                if s2 > best_score:
                                    best_score = s2
                                    best_n = new_n
                            if not _time_ok():
                                break
                        if not _time_ok():
                            break
                    if not _time_ok():
                        break
                if not _time_ok():
                    break
            if not _time_ok():
                break
    
    # Phase 5: Try all C(16,15) subsets from first 16 primes
    if _time_ok():
        for subset in combinations(_ALL_PRIMES[:16], 15):
            if not _time_ok():
                break
            s2, n2 = _score_primes(list(subset))
            if n2 <= MAX_N and s2 > best_score:
                best_score = s2
                best_n = n2
    
    # Phase 6: Try all C(16,14) subsets from first 16 primes
    if _time_ok():
        for subset in combinations(_ALL_PRIMES[:16], 14):
            if not _time_ok():
                break
            s2, n2 = _score_primes(list(subset))
            if n2 <= MAX_N and s2 > best_score:
                best_score = s2
                best_n = n2
    
    # Phase 7: Try C(17,15) subsets from first 17 primes
    if _time_ok():
        for subset in combinations(_ALL_PRIMES[:17], 15):
            if not _time_ok():
                break
            s2, n2 = _score_primes(list(subset))
            if n2 <= MAX_N and s2 > best_score:
                best_score = s2
                best_n = n2
    
    return best_n
