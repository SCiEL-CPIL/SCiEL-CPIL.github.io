"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from math import isqrt, gcd
from collections import defaultdict

def _sigma(n):
    """Compute sigma(n) - sum of divisors of n."""
    if n <= 1:
        return n
    result = 1
    d = 2
    temp = n
    while d * d <= temp:
        if temp % d == 0:
            exp = 0
            while temp % d == 0:
                temp //= d
                exp += 1
            result *= (d**(exp+1) - 1) // (d - 1)
        d += 1
    if temp > 1:
        result *= (temp + 1)
    return result

def _divisors(n):
    """Generate all divisors of n."""
    divs = []
    for i in range(1, isqrt(n) + 1):
        if n % i == 0:
            divs.append(i)
            if i != n // i:
                divs.append(n // i)
    return divs

def _find_solutions_for_M(M):
    """Find all k such that k * sigma(k) = M."""
    solutions = []
    for k in _divisors(M):
        if k > isqrt(M):
            continue
        if k * _sigma(k) == M:
            solutions.append(k)
    return solutions

def _search_best_M(deadline):
    """Search for M with many solutions."""
    best_M = 333312
    best_k = [336, 372, 434]
    best_count = 3
    
    # Strategy: enumerate M values that are products of small primes with various exponents
    # These tend to have many divisors
    small_primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31]
    
    # Generate candidate M values from products of small primes
    # We'll use a recursive approach to generate products up to 10^18
    candidates = set()
    
    def gen_products(idx, current, max_prime_idx):
        if time.time() > deadline:
            return
        if current > 10**18:
            return
        if current > 1:
            candidates.add(current)
        if idx >= max_prime_idx:
            return
        p = small_primes[idx]
        val = 1
        while val * p <= 10**18:
            val *= p
            gen_products(idx + 1, current * val, max_prime_idx)
            if time.time() > deadline:
                return
    
    # Generate candidates from products of first few primes
    gen_products(0, 1, 8)
    
    # Sort by number of divisors (heuristic: more divisors = more potential solutions)
    # Actually, let's just try them in a reasonable order
    # Try candidates that are "balanced" - not too prime-heavy
    
    # For efficiency, let's prioritize M values around certain ranges
    # and check them
    
    checked = set()
    for M in sorted(candidates):
        if time.time() > deadline:
            break
        if M in checked:
            continue
        checked.add(M)
        
        if M < 1000:  # Skip very small
            continue
        if M > 10**18:
            continue
            
        sols = _find_solutions_for_M(M)
        if len(sols) > best_count:
            best_count = len(sols)
            best_M = M
            best_k = sols
            if best_count >= 10:  # Good enough, can stop early
                break
    
    return best_M, best_k

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Leave 10s safety margin
    
    # First, try the known good answer
    M0, k0 = 333312, [336, 372, 434]
    best_M, best_k = M0, k0
    
    # Search for better M
    try:
        found_M, found_k = _search_best_M(deadline)
        if len(found_k) > len(best_k):
            best_M, best_k = found_M, found_k
    except Exception:
        pass
    
    # Verify the result
    if not best_k:
        best_M, best_k = M0, k0
    
    # Final verification
    valid_k = []
    for k in best_k:
        if isinstance(k, int) and not isinstance(k, bool) and 1 <= k <= isqrt(best_M):
            if k * _sigma(k) == best_M:
                valid_k.append(k)
    
    if not valid_k:
        best_M, best_k = M0, k0
        valid_k = [k for k in best_k if k * _sigma(k) == best_M and k <= isqrt(best_M)]
    
    # Ensure types are correct
    best_M = int(best_M)
    best_k = [int(k) for k in valid_k]
    
    return {'M': best_M, 'k': best_k}
# EVOLVE-BLOCK-END
