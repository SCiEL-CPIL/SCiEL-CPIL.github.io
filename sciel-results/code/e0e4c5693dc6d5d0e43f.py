import time
import math
from itertools import product

def _score(n):
    """Compute tau_perp(n): number of adjacent coprime pairs in sorted divisors."""
    if n < 2:
        return 0
    # Factorize n
    factors = []
    temp = n
    d = 2
    while d * d <= temp:
        if temp % d == 0:
            cnt = 0
            while temp % d == 0:
                temp //= d
                cnt += 1
            factors.append((d, cnt))
        d += 1 if d == 2 else 2
    if temp > 1:
        factors.append((temp, 1))
    
    # Generate all divisors
    divisors = [1]
    for p, e in factors:
        new_divs = []
        for d in divisors:
            for i in range(e + 1):
                new_divs.append(d * (p ** i))
        divisors = new_divs
    divisors.sort()
    
    # Count adjacent coprime pairs
    score = 0
    for i in range(len(divisors) - 1):
        if math.gcd(divisors[i], divisors[i+1]) == 1:
            score += 1
    return score

def construct():
    start = time.time()
    deadline = start + 145.0
    
    # First, try squarefree numbers with many primes
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47]
    
    best_n = 2 * 3 * 5 * 7 * 11 * 13 * 17 * 19  # baseline
    best_score = _score(best_n)
    
    # Try all subsets of primes (squarefree) up to size 15
    # Product of first 15 primes ~ 6.1e17, first 16 ~ 3.3e18
    # So we can use up to 15 primes squarefree
    for k in range(8, 16):  # try 8 to 15 primes
        if time.time() > deadline - 10:
            break
        # Try consecutive primes starting from different positions
        for start_idx in range(0, len(primes) - k + 1):
            if time.time() > deadline - 10:
                break
            n = 1
            for i in range(k):
                n *= primes[start_idx + i]
            if n > 10**18:
                break
            s = _score(n)
            if s > best_score:
                best_score = s
                best_n = n
    
    # Also try some numbers with exponents > 1
    # E.g., 2^a * 3 * 5 * 7 * ... with small a
    for a2 in range(1, 7):
        if time.time() > deadline - 10:
            break
        base = 2 ** a2
        for k in range(5, 14):
            if time.time() > deadline - 10:
                break
            for start_idx in range(1, len(primes) - k + 1):  # skip 2
                n = base
                for i in range(k):
                    n *= primes[start_idx + i]
                if n > 10**18:
                    break
                s = _score(n)
                if s > best_score:
                    best_score = s
                    best_n = n
    
    return best_n
