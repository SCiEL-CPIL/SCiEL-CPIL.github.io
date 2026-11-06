"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from math import isqrt, gcd

def _gen_powerful(limit):
    """Generate all powerful numbers <= limit."""
    pows = set()
    # powerful number = product of p^(2*e) for various primes
    # A powerful number has form a^2 * b^3 where b is squarefree
    # Actually: n is powerful iff for every prime p|n, p^2|n
    # So n = product p_i^(2*k_i) for k_i >= 1
    # Equivalently n = a^2 * b where b is squarefree and for each p|b, p^2|n
    # Let's use: n = a^2 * b^3 where b is squarefree (this covers all powerful numbers)
    # Wait: if n = product p^(2*k), then write k = 2*m + r where r in {0,1}
    # If r=0: p^(2k) = (p^k)^2
    # If r=1: p^(2k+1) is NOT powerful since p^1 divides n but p^2 may not... 
    # Actually powerful means p^2 | n for all p | n. So exponents must be >= 2.
    # So n = product p^(e_p) where e_p >= 2 for all p | n.
    # We can write n = a^2 * b where b is squarefree... no.
    # Better: n = (product p^(floor(e_p/2)))^2 * (product p^(e_p % 2))
    # But e_p >= 2, so e_p % 2 can be 0 or 1.
    # If e_p % 2 == 1, then we have an extra p factor. But p^2 already divides n, so this is fine.
    # The standard parametrization: n = a^2 * b^3 where b is squarefree.
    # Check: if n = a^2 * b^3 with b squarefree, then for each p | b, exponent in b^3 is 3, 
    # plus possibly from a^2. Total exponent >= 3 >= 2. For p | a but p not in b, exponent is even >= 2.
    # Conversely, any powerful n: write n = product p^(e_p), e_p >= 2.
    # Let b = product of p where e_p is odd. Then b is squarefree.
    # n / b^3 = product p^(e_p - 3) for p|b, and p^(e_p) for p not in b.
    # For p|b: e_p - 3 >= -1... problem if e_p = 2.
    # So the a^2 * b^3 form doesn't directly work for e_p = 2.
    
    # Direct approach: enumerate by factoring structure.
    # n is powerful iff n = product p_i^(2*k_i) with k_i >= 1.
    # Equivalently, n = m^2 * r where r is squarefree and for each p|r, the exponent of p in n is odd (>=3).
    # Hmm, let me just do: for each squarefree b, for each a, check if a^2 * b^3 <= limit.
    # But that misses numbers like 4 = 2^2 (b=1, a=2 works: 4*1=4).
    # 8 = 2^3: a=1, b=2: 1*8=8. Yes.
    # 16 = 2^4: a=2, b=1: 4*1=4... no. a=4,b=1: 16. Yes.
    # 12 = 2^2 * 3: not powerful (3^1).
    # 72 = 2^3 * 3^2: powerful. a^2*b^3: b must include primes with odd exponent.
    # 72 = 2^3 * 3^2. Odd exponents: 2 (exp 3). So b=2. b^3=8. 72/8=9=3^2. a=3. 9*8=72. Yes.
    
    # So: for squarefree b, for a from 1 to isqrt(limit // b^3), add a^2 * b^3.
    # But we also need b=1 case (perfect squares).
    
    # Generate squarefree numbers up to some bound for b.
    # b^3 <= limit, so b <= limit^(1/3).
    b_max = int(limit ** (1/3)) + 1
    
    # Sieve for squarefree numbers up to b_max
    is_sqfree = [True] * (b_max + 1)
    is_sqfree[0] = False
    for i in range(2, isqrt(b_max) + 1):
        if is_sqfree[i]:
            sq = i * i
            for j in range(sq, b_max + 1, sq):
                is_sqfree[j] = False
    
    squarefree_b = [i for i in range(1, b_max + 1) if is_sqfree[i]]
    
    for b in squarefree_b:
        b3 = b ** 3
        if b3 > limit:
            continue
        a_max = isqrt(limit // b3)
        for a in range(1, a_max + 1):
            n = a * a * b3
            if n <= limit:
                pows.add(n)
    
    # Also add 1
    pows.add(1)
    return sorted(pows)

def construct():
    start_time = time.time()
    deadline = start_time + 130  # safety margin
    
    # Phase 1: Generate powerful numbers up to a reasonable limit
    # For N up to 10^18, we need powerful numbers up to 10^18
    # But generating all powerful up to 10^18 might be too many.
    # Let's estimate: number of powerful numbers <= x is ~ C * x^(1/2)
    # For x = 10^18, that's ~ C * 10^9 which is too many.
    # 
    # Alternative: don't generate all. Instead, search for N directly.
    # Strategy: pick candidate N values, count representations.
    # 
    # Better structural approach: 
    # 1. Generate powerful numbers up to a moderate limit (say 10^9)
    # 2. Use these to find N with many pairs where both a,b <= 10^9
    # 3. For larger N, use a different strategy
    
    # Actually, let's think differently. The score is the number of pairs.
    # We want N with many representations as sum of two powerful numbers.
    # 
    # Key insight: if we have a large set of powerful numbers P, then for a given N,
    # the number of pairs is |{a in P : a <= N/2 and N-a in P}|.
    # 
    # To maximize this, we want N such that many elements of P pair up to sum to N.
    # 
    # Strategy: 
    # 1. Generate powerful numbers up to some limit L
    # 2. For various candidate N values, count pairs
    # 3. Return the best found
    
    # Generate powerful numbers up to 10^9 (manageable)
    L = 10**9
    powerful = _gen_powerful(L)
    pow_set = set(powerful)
    n_pow = len(powerful)
    
    # For efficiency, use numpy for pair counting
    import numpy as np
    pow_arr = np.array(powerful, dtype=np.int64)
    
    # Candidate N values: try various ranges
    # The optimal N should be around 2*L to allow many pairs
    # But N can be up to 10^18. For N > 2*L, only powerful numbers <= L can pair.
    # For N in [2, 2*L], we can have more flexibility.
    
    # Let's search in the range [L, 2*L] where both a and b can be up to L
    best_N = None
    best_count = 0
    best_pairs = []
    
    # Sample candidate N values
    # Use random sampling plus some structured candidates
    random.seed(42)
    
    # Try N values that are sums of small powerful numbers (likely to have many representations)
    # Also try N = 2 * p for various powerful p (symmetric)
    
    candidates = set()
    # Add some structured candidates
    for p in powerful[:500]:
        candidates.add(2 * p)
        candidates.add(p + 1)
        if p + 2 <= 2 * L:
            candidates.add(p + 2)
    
    # Add random candidates in [1, 2*L]
    for _ in range(2000):
        candidates.add(random.randint(1, min(2 * L, 10**18)))
    
    # Add candidates near 2*L
    for offset in range(0, 1000, 10):
        candidates.add(2 * L - offset)
        candidates.add(2 * L + offset)
    
    candidates = sorted(candidates)
    
    for N in candidates:
        if time.time() > deadline - 30:
            break
        if N < 2 or N > 10**18:
            continue
        
        # Count pairs: a <= b, a + b = N, a,b powerful
        # a ranges from max(1, N-L) to min(L, N//2)
        a_lo = max(1, N - L)
        a_hi = min(L, N // 2)
        
        if a_lo > a_hi:
            continue
        
        # Use vectorized check
        # Find powerful numbers in [a_lo, a_hi]
        # Binary search
        left = np.searchsorted(pow_arr, a_lo, side='left')
        right = np.searchsorted(pow_arr, a_hi, side='right')
        
        if right <= left:
            continue
        
        a_candidates = pow_arr[left:right]
        b_candidates = N - a_candidates
        
        # Check which b_candidates are in pow_set
        # Vectorized membership test
        mask = np.array([b in pow_set for b in b_candidates])
        count = int(np.sum(mask))
        
        if count > best_count:
            best_count = count
            best_N = N
            valid_a = a_candidates[mask]
            best_pairs = [[int(a), int(N - a)] for a in valid_a]
    
    # If we found something good, return it
    if best_N is not None and best_count > 0:
        return {'N': best_N, 'pairs': best_pairs}
    
    # Fallback: return a trivial result
    return {'N': 2, 'pairs': [[1, 1]]}
# EVOLVE-BLOCK-END
