"""
Strategy: Candidate-based search for sums of two powerful numbers.

1. Generate powerful numbers up to a small limit $L_{small}$ (e.g., $10^7$).
2. Compute all pairwise sums for this small set to identify "hot" candidates $N$ 
   with high representation counts.
3. Generate powerful numbers up to a much larger limit $L_{large}$ (e.g., $10^{11}$ or $10^{12}$).
4. For each hot candidate $N$, count the number of representations $(a,b)$ such that 
   $a+b=N$ using the larger set. 
5. Return the $N$ with the maximum count found.

This approach avoids the $O(P^2)$ memory/time explosion of storing all sums for large $P$, 
instead trading it for $O(K \cdot P_{large})$ where $K$ is the number of candidates. 
This allows exploring $N$ up to $2 \times 10^{12}$, significantly increasing the chance 
of finding a higher representation count than the brute-force limit of $2 \times 10^8$.
"""

# EVOLVE-BLOCK-START
import time
from math import isqrt
import numpy as np
import bisect

def _generate_powerfuls(limit):
    """
    Generate all powerful numbers <= limit.
    A powerful number has the form a^2 * b^3 where b is square-free.
    We iterate b (square-free) and a.
    """
    powerfuls = set()
    # b must be square-free and b^3 <= limit
    max_b = int(limit ** (1/3)) + 2
    
    # Primes up to sqrt(max_b) for square-free check
    sqrt_max_b = isqrt(max_b)
    primes = []
    for x in range(2, sqrt_max_b + 1):
        is_p = True
        for p in primes:
            if p * p > x:
                break
            if x % p == 0:
                is_p = False
                break
        if is_p:
            primes.append(x)
    
    for b in range(1, max_b + 1):
        # Check if b is square-free
        is_sq_free = True
        for p in primes:
            if p * p > b:
                break
            if b % (p * p) == 0:
                is_sq_free = False
                break
        if not is_sq_free:
            continue
        
        b3 = b ** 3
        if b3 > limit:
            break
        
        max_a = isqrt(limit // b3)
        for a in range(1, max_a + 1):
            val = a * a * b3
            if val <= limit:
                powerfuls.add(val)
    
    return sorted(powerfuls)

def construct():
    start_time = time.time()
    deadline = start_time + 145  # Safety margin for 150s limit
    
    # Phase 1: Find candidates using a small limit
    # L_small = 10^7 -> P ~ 3000-4000. P^2/2 ~ 10^7 pairs. Fast.
    L_small = 10_000_000
    powerfuls_small = _generate_powerfuls(L_small)
    P_small = len(powerfuls_small)
    arr_small = np.array(powerfuls_small, dtype=np.int64)
    
    # Compute all sums for small set to find top candidates
    # Using np.triu_indices for i <= j
    i_idx, j_idx = np.triu_indices(P_small)
    sums_small = arr_small[i_idx] + arr_small[j_idx]
    
    # Count frequencies
    unique_sums, counts = np.unique(sums_small, return_counts=True)
    
    # Get top 200 candidates
    # We want candidates that are likely to be valid in the larger set.
    # Since the density is similar, high counts in small set suggest high counts in large set.
    top_idx = np.argsort(-counts)[:200]
    candidates = unique_sums[top_idx].tolist()
    
    # Ensure candidates are at least 2
    candidates = [c for c in candidates if c >= 2]
    
    elapsed = time.time() - start_time
    remaining = deadline - elapsed
    
    # Phase 2: Verify and extend with a larger limit
    # We want to go as large as possible.
    # L_large = 10^11 -> P ~ 10^5-2*10^5. 
    # Counting for one candidate takes O(P_large) ~ 10^5 operations.
    # 200 candidates * 10^5 ops = 2*10^7 ops. Very fast in Python if optimized?
    # Actually, Python loops are slow. 2*10^7 iterations might take ~2-5 seconds.
    # We can afford L_large = 10^12 -> P ~ 3*10^5.
    # 200 * 3*10^5 = 6*10^7 iterations. Might take 10-20s.
    # Let's try L_large = 10^11 first, then increase if time permits.
    
    best_N = 2
    best_pairs = [[1, 1]]
    best_count = 1
    
    # Try L_large = 10^11
    L_large = 10_000_000_000
    powerfuls_large = _generate_powerfuls(L_large)
    powerfuls_set_large = set(powerfuls_large)
    # powerfuls_list_large is already sorted from _generate_powerfuls
    
    # Filter candidates: N must be <= 2 * max(powerfuls_large)
    max_pow_large = powerfuls_large[-1] if powerfuls_large else 1
    valid_candidates = [N for N in candidates if N <= 2 * max_pow_large]
    
    # Sort candidates by their small-set count (descending) to prioritize
    # We lost the counts in the list, so let's re-sort based on the previous step
    # Actually, we can just iterate. The top ones are likely better.
    
    for N in valid_candidates:
        if time.time() > deadline - 10:
            break
            
        count = 0
        pairs = []
        # Iterate a in powerfuls_large
        # Optimization: break if a > N/2
        half_N = N // 2
        # Use bisect to find the end index for a <= N/2
        end_idx = bisect.bisect_right(powerfuls_large, half_N)
        
        for i in range(end_idx):
            a = powerfuls_large[i]
            b = N - a
            if b in powerfuls_set_large:
                count += 1
                pairs.append([a, b])
        
        if count > best_count:
            best_count = count
            best_N = N
            best_pairs = pairs
    
    # If we have time, try an even larger limit
    if time.time() < deadline - 20:
        L_large2 = 100_000_000_000 # 10^11
        # Only if previous was 10^10
        if L_large < L_large2:
             powerfuls_large2 = _generate_powerfuls(L_large2)
             powerfuls_set_large2 = set(powerfuls_large2)
             max_pow_large2 = powerfuls_large2[-1] if powerfuls_large2 else 1
             
             # Re-evaluate candidates against the larger set
             # We should also consider that new candidates might emerge? 
             # No, the candidates from the small set are the best bet.
             # But we might have missed some if they were low in small set but high in large?
             # Unlikely, as density is similar.
             
             for N in valid_candidates:
                 if N > 2 * max_pow_large2:
                     continue
                 if time.time() > deadline - 5:
                     break
                 
                 count = 0
                 pairs = []
                 half_N = N // 2
                 end_idx = bisect.bisect_right(powerfuls_large2, half_N)
                 
                 for i in range(end_idx):
                     a = powerfuls_large2[i]
                     b = N - a
                     if b in powerfuls_set_large2:
                         count += 1
                         pairs.append([a, b])
                 
                 if count > best_count:
                     best_count = count
                     best_N = N
                     best_pairs = pairs

    # Fallback
    if not best_pairs:
        return {'N': 2, 'pairs': [[1, 1]]}
    
    return {'N': best_N, 'pairs': best_pairs}

# EVOLVE-BLOCK-END
