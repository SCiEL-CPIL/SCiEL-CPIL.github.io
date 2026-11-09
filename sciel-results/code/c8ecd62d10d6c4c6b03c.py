# EVOLVE-BLOCK-START
import time
from math import isqrt
import numpy as np

def _generate_powerful(limit):
    """Generate all powerful numbers up to limit.
    
    A powerful number n has the form m^2 * b where b is squarefree and
    b divides m (i.e., every prime factor of b also divides m).
    Equivalently, n = prod p_i^(2*e_i + f_i) where f_i in {0,1} and
    if f_i = 1 then e_i >= 1.
    
    We enumerate by the square part m and the squarefree part b.
    """
    sqrt_limit = isqrt(limit)
    
    # Sieve for smallest prime factor up to sqrt_limit
    max_spf = isqrt(sqrt_limit) + 2
    spf = list(range(max_spf))
    for i in range(2, isqrt(max_spf) + 1):
        if spf[i] == i:
            for j in range(i * i, max_spf, i):
                if spf[j] == j:
                    spf[j] = i
    
    def factorize(n):
        """Return list of distinct prime factors of n."""
        factors = []
        while n > 1:
            if n < max_spf:
                p = spf[n]
            else:
                p = n  # n is prime
            factors.append(p)
            while n % p == 0:
                n //= p
        return factors
    
    powerful = set([1])
    
    for m in range(1, sqrt_limit + 1):
        m2 = m * m
        if m2 > limit:
            break
        # Get distinct prime factors of m
        primes = factorize(m)
        n_primes = len(primes)
        
        # Enumerate all subsets of primes to form squarefree b
        # b must be a product of a subset of primes(m)
        for mask in range(1 << n_primes):
            b = 1
            for i in range(n_primes):
                if mask & (1 << i):
                    b *= primes[i]
            val = m2 * b
            if val <= limit:
                powerful.add(val)
            else:
                break
    
    return sorted(powerful)

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s kill
    
    # Phase 1: Generate powerful numbers up to a reasonable limit
    # We need powerful numbers up to about 10^9 to get good coverage
    # For N up to 10^18, we need powerful numbers up to N/2 at minimum
    # But we can only generate up to ~10^9 in reasonable time
    # Strategy: generate powerful numbers up to L, then for candidate N,
    # check pairs (a, b) where a, b are powerful and a + b = N
    
    L = 500_000_000  # 5 * 10^8
    powerful = _generate_powerful(L)
    powerful_set = set(powerful)
    P = len(powerful)
    
    elapsed = time.time() - start_time
    if elapsed > deadline - 80:
        # Fallback
        return {'N': 2, 'pairs': [[1, 1]]}
    
    pw_arr = np.array(powerful, dtype=np.int64)
    
    # Phase 2: Generate candidate N values
    # Key insight: N with many representations tends to be a sum of two
    # "rich" powerful numbers. Also, N = 2p where p is powerful gives
    # at least one representation (p, p).
    # 
    # Better strategy: look for N that can be written as a + b in many ways.
    # If we fix a set of powerful numbers, N = a + b for many pairs.
    # 
    # Approach: For each powerful number a, N = a + b where b is also powerful.
    # So N is in the set {a + b : a, b in powerful, a <= b}.
    # We want N that appears many times in this sumset.
    #
    # Efficient approach: use the fact that if N is even and N/2 is powerful,
    # we get (N/2, N/2). More generally, we want N such that N - a is powerful
    # for many a in powerful.
    #
    # Let's try: pick powerful numbers with many divisors as "anchors"
    # and build N around them.
    
    # Score powerful numbers by their "richness" (number of divisors)
    def divisor_count(n):
        """Count divisors of n using trial division up to sqrt(n)."""
        if n <= 1:
            return 1
        count = 1
        temp = n
        for p in range(2, min(isqrt(n) + 1, 10000)):
            if temp % p == 0:
                exp = 0
                while temp % p == 0:
                    temp //= p
                    exp += 1
                count *= (exp + 1)
        if temp > 1:
            count *= 2
        return count
    
    # Get top powerful numbers by divisor count
    # Limit to avoid too much computation
    top_powerful = []
    for p in powerful:
        if p > 1:
            d = divisor_count(p)
            top_powerful.append((d, p))
        if len(top_powerful) > 5000:
            break
    
    top_powerful.sort(reverse=True)
    top_pw = [p for _, p in top_powerful[:500]]
    
    # Generate candidate N values from sums of top powerful numbers
    candidates = set()
    for i in range(len(top_pw)):
        for j in range(i, len(top_pw)):
            N = top_pw[i] + top_pw[j]
            if N <= 10**18:
                candidates.add(N)
    
    # Also add 2*p for top powerful numbers
    for p in top_pw[:300]:
        if 2 * p <= 10**18:
            candidates.add(2 * p)
    
    candidates = sorted(candidates)
    
    # Phase 3: Verify candidates
    best_N = 2
    best_pairs = [[1, 1]]
    best_count = 1
    
    # For efficiency, use numpy to check many candidates at once
    # For each N, we need to count a in powerful such that a <= N/2 and N-a in powerful
    
    # Pre-compute: for fast membership, use the set
    # Process candidates in order of "promisingness"
    
    checked = 0
    for N in candidates:
        if time.time() > deadline - 20:
            break
        
        # a must satisfy: a <= N/2, a in powerful, N-a in powerful
        half_N = N // 2
        # Find powerful numbers <= half_N
        # Use binary search on sorted powerful array
        idx = np.searchsorted(pw_arr, half_N, side='right')
        a_candidates = pw_arr[:idx]
        
        if len(a_candidates) == 0:
            continue
        
        b_values = N - a_candidates
        # Check which b_values are in powerful_set
        # Convert to Python list for set lookup (numpy doesn't support set ops well)
        b_list = b_values.tolist()
        a_list = a_candidates.tolist()
        
        valid = [a for a, b in zip(a_list, b_list) if b in powerful_set]
        
        count = len(valid)
        if count > best_count:
            best_count = count
            best_N = N
            best_pairs = [[a, N - a] for a in valid]
        
        checked += 1
        if checked % 50 == 0 and time.time() > deadline - 40:
            break
    
    # Phase 4: Local search around best N
    if time.time() < deadline - 30 and best_N > 2:
        for delta in range(1, 500):
            for N_try in [best_N + delta, best_N - delta]:
                if N_try < 2 or N_try > 10**18:
                    continue
                if time.time() > deadline - 15:
                    break
                
                half_N = N_try // 2
                idx = np.searchsorted(pw_arr, half_N, side='right')
                a_candidates = pw_arr[:idx]
                
                if len(a_candidates) == 0:
                    continue
                
                b_values = N_try - a_candidates
                b_list = b_values.tolist()
                a_list = a_candidates.tolist()
                
                valid = [a for a, b in zip(a_list, b_list) if b in powerful_set]
                
                count = len(valid)
                if count > best_count:
                    best_count = count
                    best_N = N_try
                    best_pairs = [[a, N_try - a] for a in valid]
    
    # Phase 5: If time permits, try larger powerful numbers by extending the set
    # Generate powerful numbers in a higher range
    if time.time() < deadline - 50:
        # Generate powerful numbers in range [L, 2*L]
        L2 = 1_000_000_000
        powerful2 = _generate_powerful(L2)
        # Filter to only those > L
        powerful2_high = [p for p in powerful2 if p > L]
        if powerful2_high:
            powerful2_set = set(powerful2_high)
            # Add to our set
            powerful_set.update(powerful2_high)
            
            # Re-verify best N with extended set
            if best_N > 2:
                half_N = best_N // 2
                # Need to check both old and new powerful numbers
                all_pw = list(powerful_set)
                all_pw.sort()
                all_pw_arr = np.array(all_pw, dtype=np.int64)
                
                idx = np.searchsorted(all_pw_arr, half_N, side='right')
                a_candidates = all_pw_arr[:idx]
                b_values = best_N - a_candidates
                b_list = b_values.tolist()
                a_list = a_candidates.tolist()
                
                valid = [a for a, b in zip(a_list, b_list) if b in powerful_set]
                count = len(valid)
                if count > best_count:
                    best_count = count
                    best_pairs = [[a, best_N - a] for a in valid]
    
    return {'N': best_N, 'pairs': best_pairs}
# EVOLVE-BLOCK-END
