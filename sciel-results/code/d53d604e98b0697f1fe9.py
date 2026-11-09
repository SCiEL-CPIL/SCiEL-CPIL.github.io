"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from math import isqrt
from collections import defaultdict

def _generate_powerful(limit):
    """Generate all powerful numbers up to limit using efficient enumeration."""
    powerful_set = set()
    powerful_set.add(1)
    
    # A powerful number n has the form prod(p_i^(e_i)) where e_i >= 2.
    # Equivalently, n = m^2 * b where b is squarefree and b | m.
    # This gives a unique representation for each powerful number.
    # 
    # We iterate over m from 1 to sqrt(limit), and for each m, enumerate
    # all squarefree divisors b of m such that m^2 * b <= limit.
    
    sqrt_limit = isqrt(limit)
    
    for m in range(1, sqrt_limit + 1):
        # Factor m to get its distinct prime factors
        temp = m
        primes = []
        for p in range(2, isqrt(temp) + 1):
            if temp % p == 0:
                primes.append(p)
                while temp % p == 0:
                    temp //= p
        if temp > 1:
            primes.append(temp)
        
        # Enumerate all subsets of primes to get squarefree divisors
        n_primes = len(primes)
        for mask in range(1 << n_primes):
            b = 1
            for i in range(n_primes):
                if mask & (1 << i):
                    b *= primes[i]
            if b > limit // (m * m):
                continue
            powerful_set.add(m * m * b)
    
    return sorted(powerful_set)

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s kill
    
    # Generate powerful numbers up to a limit that allows good search
    # For N up to ~50M, we need powerful numbers up to 50M
    # Number of powerful numbers up to N is ~ sqrt(N) * constant
    # For N=50M, that's ~ 7000-8000 powerful numbers
    # O(P^2) with P=8000 is 64M operations, feasible in ~30-60s with optimization
    
    N_MAX = 50_000_000
    
    powerful = _generate_powerful(N_MAX)
    powerful_set = set(powerful)
    P = len(powerful)
    
    elapsed = time.time() - start_time
    if elapsed > deadline - 30:
        # Fallback: return a small valid result
        return {'N': 2, 'pairs': [[1, 1]]}
    
    # Count representations: for each pair (a, b) with a <= b, a+b = N
    # Use the fact that powerful is sorted
    import bisect
    
    count = defaultdict(int)
    
    # For each a in powerful, find all b in powerful with a <= b and a+b <= N_MAX
    # Since powerful is sorted, we can use bisect to find the range
    for i, a in enumerate(powerful):
        if 2 * a > N_MAX:
            break
        if time.time() > deadline - 10:
            break
        
        # b must be >= a and <= N_MAX - a
        lo = bisect.bisect_left(powerful, a)
        hi = bisect.bisect_right(powerful, N_MAX - a)
        
        # Count the number of valid b's and add to count
        # For efficiency, just count without storing pairs (we'll reconstruct later)
        count[a]  # touch to ensure key exists? No, we need count per N
        
        for j in range(lo, hi):
            N_val = a + powerful[j]
            count[N_val] += 1
    
    if time.time() > deadline - 5:
        # Return best found so far
        if count:
            best_N = max(count, key=lambda n: count[n])
            pairs = []
            for a in powerful:
                if 2 * a > best_N:
                    break
                b = best_N - a
                if b >= a and b in powerful_set:
                    pairs.append([a, b])
            return {'N': best_N, 'pairs': pairs}
        else:
            return {'N': 2, 'pairs': [[1, 1]]}
    
    if not count:
        return {'N': 2, 'pairs': [[1, 1]]}
    
    best_N = max(count, key=lambda n: count[n])
    
    # Reconstruct the pairs for best_N
    pairs = []
    for a in powerful:
        if 2 * a > best_N:
            break
        b = best_N - a
        if b >= a and b in powerful_set:
            pairs.append([a, b])
    
    return {'N': best_N, 'pairs': pairs}
# EVOLVE-BLOCK-END
