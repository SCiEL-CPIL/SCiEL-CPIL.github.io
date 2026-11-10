import time
import numpy as np
from math import gcd, isqrt

def construct():
    start_time = time.time()
    deadline = start_time + 85  # 5s margin

    MAX_N = 256

    # 1. Generate primes and smallest prime factor (SPF)
    def get_spf(n):
        spf = list(range(n + 1))
        for i in range(2, isqrt(n) + 1):
            if spf[i] == i:  # i is prime
                for j in range(i * i, n + 1, i):
                    if spf[j] == j:
                        spf[j] = i
        return spf

    spf = get_spf(MAX_N)
    primes = [i for i in range(2, MAX_N + 1) if spf[i] == i]
    num_primes = len(primes)
    
    # Map prime to index for bitmask operations? 
    # Bitmask of 54 bits fits in a Python int (arbitrary precision).
    prime_to_bit = {p: (1 << i) for i, p in enumerate(primes)}
    
    # Precompute prime factor bitmask for each number
    factors_mask = [0] * (MAX_N + 1)
    for n in range(2, MAX_N + 1):
        mask = 0
        x = n
        while x > 1:
            p = spf[x]
            mask |= prime_to_bit[p]
            while x % p == 0:
                x //= p
        factors_mask[n] = mask

    # Group numbers by their Smallest Prime Factor
    # candidates[p_idx] = list of n such that spf(n) == primes[p_idx]
    candidates = [[] for _ in range(num_primes)]
    for n in range(2, MAX_N + 1):
        p = spf[n]
        idx = primes.index(p) # This is slow, use a dict
        # Better: create a dict prime_to_idx
        # Let's rebuild prime_to_idx
        pass

    prime_to_idx = {p: i for i, p in enumerate(primes)}
    candidates = [[] for _ in range(num_primes)]
    for n in range(2, MAX_N + 1):
        p = spf[n]
        idx = prime_to_idx[p]
        candidates[idx].append(n)
    
    # Sort candidates in descending order to find high sums early
    for i in range(num_primes):
        candidates[i].sort(reverse=True)

    # Precompute upper bound helper:
    # For a given used_mask, what is the max value we can add for each remaining prime?
    # This is dynamic, so we compute in the B&B loop.
    # To speed up, we can precompute for each prime p, the list of (n, mask) pairs.
    
    best_sum = 1 # Always include 1
    best_set = {1}
    
    def upper_bound(idx, used_mask, current_sum):
        """
        Calculate a valid upper bound for the sum.
        Assumes we can pick the best available number for each remaining prime independently.
        """
        bound = current_sum
        for j in range(idx, num_primes):
            p = primes[j]
            # Find max n in candidates[j] (and potentially others? No, restricted to spf(n)==p)
            # Wait, the bound should consider ALL numbers not yet used, not just those with spf=p.
            # But in our restricted search, we only pick n with spf(n)==p at step p.
            # However, a number n with spf(n)=q (q>p) could be picked at step q.
            # So for prime p, we can only pick numbers from candidates[j].
            # Is this a valid upper bound?
            # Yes, because any valid solution will assign each number n to its spf(n).
            # So the contribution of prime p to the sum is at most the max n in candidates[j] that is compatible with used_mask.
            # Note: A number n in candidates[j] might use primes q > p.
            # The bound ignores that using n blocks q.
            # So it is an overestimate, which is correct for an upper bound.
            
            max_val = 0
            for n in candidates[j]:
                if (factors_mask[n] & used_mask) == 0:
                    if n > max_val:
                        max_val = n
                    else:
                        break # Since sorted descending, if n <= max_val, rest are smaller
            bound += max_val
            
            # Time check inside bound calculation? 
            # It's called frequently. Keep it fast.
        return bound

    def branch_and_bound(idx, used_mask, current_sum, current_set):
        nonlocal best_sum, best_set
        
        if time.time() > deadline:
            return

        # Prune if bound is not better
        # Optimization: Calculate bound only if it might be useful?
        # Or just calculate it.
        ub = upper_bound(idx, used_mask, current_sum)
        if ub <= best_sum:
            return

        if idx == num_primes:
            if current_sum > best_sum:
                best_sum = current_sum
                best_set = set(current_set)
            return

        p_idx = idx
        p = primes[p_idx]
        
        # Option 1: Skip this prime (no number assigned to it as spf)
        branch_and_bound(idx + 1, used_mask, current_sum, current_set)
        
        # Option 2: Pick a number n from candidates[p_idx]
        for n in candidates[p_idx]:
            if (factors_mask[n] & used_mask) == 0:
                new_mask = used_mask | factors_mask[n]
                new_sum = current_sum + n
                # Prune before recursing
                # We can check bound here too, but the recursive call does it.
                # To save time, check if new_sum + (simple bound) > best_sum
                # Simple bound: new_sum + sum of all remaining max candidates
                # Let's just recurse.
                
                branch_and_bound(idx + 1, new_mask, new_sum, current_set + [n])
                
                # If we found a very good solution, we might break early?
                # No, we need to explore all.
                
    # Start B&B
    # Initial state: idx=0, used_mask=0, current_sum=1, current_set=[1]
    branch_and_bound(0, 0, 1, [1])

    # Fallback / Refinement with Local Search if B&B didn't finish or to ensure robustness
    # Given the tightness of B&B with spf restriction, it should be optimal.
    # But let's add a quick local search on the result to be safe.
    
    result_list = list(best_set)
    
    # Local Search
    improved = True
    while improved and time.time() < deadline:
        improved = False
        S_list = list(best_set)
        
        # 1-for-1 swaps
        for a in S_list:
            if a == 1:
                continue
            for b in range(256, a, -1):
                if b in best_set:
                    continue
                if all(gcd(b, x) == 1 for x in best_set if x != a):
                    if b > a:
                        best_set = (best_set - {a}) | {b}
                        best_sum = best_sum - a + b
                        improved = True
                        break
            if improved:
                break
        
        # Add numbers
        if time.time() < deadline:
            for b in range(256, 1, -1):
                if b in best_set:
                    continue
                if all(gcd(b, x) == 1 for x in best_set):
                    best_set.add(b)
                    best_sum += b
                    improved = True
                    break

    return [int(x) for x in sorted(best_set)]
