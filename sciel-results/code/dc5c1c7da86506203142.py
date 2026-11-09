"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math
import numpy as np
import heapq

def construct():
    start_time = time.time()
    DEADLINE = 85.0  # Safety margin before 90s kill
    
    N = 123457
    
    def generate_smooth_numbers(N, primes):
        """Generate all numbers <= N with prime factors in 'primes'."""
        smooth = set([1])
        # Use a priority queue to generate in sorted order to avoid duplicates efficiently?
        # Or just simple nested loops / iterative multiplication.
        # Since N is small, a set-based BFS/DFS is fine.
        # Start with 1.
        # For each new number, multiply by each prime if <= N.
        # To avoid duplicates, we can just use a set and a queue.
        
        queue = [1]
        seen = set([1])
        while queue:
            current = queue.pop(0)
            for p in primes:
                nxt = current * p
                if nxt <= N and nxt not in seen:
                    seen.add(nxt)
                    queue.append(nxt)
        return seen

    # Try to find a decomposition where both numbers are k-smooth for small k.
    # If a and b are k-smooth, then P(a) <= k and P(b) <= k, so max(P(a), P(b)) <= k.
    # We want to minimize this value.
    
    primes_list = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97]
    
    # We check increasing sets of primes.
    # Note: 1 is always smooth.
    # If N-1 is k-smooth, then (1, N-1) is a solution with value P(N-1) <= k.
    
    for i in range(len(primes_list)):
        if time.time() > DEADLINE:
            break
            
        current_primes = primes_list[:i+1]
        smooth_numbers = generate_smooth_numbers(N, current_primes)
        
        # Check if any x in smooth_numbers has N-x also in smooth_numbers
        found = False
        for x in smooth_numbers:
            y = N - x
            if y in smooth_numbers:
                # Found a pair where both are smooth with respect to current_primes
                # The max prime factor is at most max(current_primes)
                # Actually, it could be less if the specific numbers don't use the largest prime.
                # But this is a good upper bound.
                # We can return this immediately as it's likely very good.
                # To be precise, we should calculate P(x) and P(y) to know the exact score,
                # but since we are iterating primes in increasing order, the first found
                # will have the smallest possible "smoothness bound".
                # However, a pair found with k=5 might have P(x)=2, P(y)=3, so max=3.
                # A pair found with k=7 might have max=5.
                # So we should check the actual max prime factor to be sure?
                # Or just trust that smaller k implies smaller P?
                # Yes, if x is 5-smooth, P(x) <= 5.
                # If we find a 5-smooth pair, the score is <= 5.
                # If we find a 7-smooth pair, the score is <= 7.
                # So finding a 5-smooth pair is better than or equal to finding a 7-smooth pair.
                # Thus, we can return the first found pair from the smallest k.
                
                # We should verify if there's a pair with even smaller P within the same k?
                # No, we just need *a* pair. The objective is to minimize max(P(a), P(b)).
                # Any pair of k-smooth numbers has max(P(a), P(b)) <= k.
                # So if we find a 5-smooth pair, we have a solution with value <= 5.
                # We can stop here.
                
                # Just to be safe, let's return this pair.
                # We don't strictly need to factor them again if we know they are smooth,
                # but the evaluator will factor them.
                # We just need to return valid integers.
                
                # Check time
                if time.time() > DEADLINE - 1.0:
                    # If close to deadline, return this valid pair immediately
                    return [x, y]
                    
                found = True
                # We can break out of the loop and return
                # But wait, could there be a pair with smaller P within the same k?
                # E.g. k=5, pair (2, 123455) where 123455 is 5-smooth?
                # P(2)=2. P(123455) <= 5. Max is <= 5.
                # If we found (3, 123454), P(3)=3, P(123454)<=5. Max <= 5.
                # The value is determined by the actual P.
                # Since we don't know the exact P without factoring, and factoring is cheap for these numbers,
                # we could factor them. But returning any 5-smooth pair is likely much better than the current 58.
                # Let's just return the first one found.
                
                return [x, y]
        
        if found:
            break

    # Fallback to the sieve method if no smooth pair found (unlikely for small N)
    lpf = np.ones(N + 1, dtype=np.int32)
    for i in range(2, N + 1):
        if lpf[i] == 1:
            lpf[i::i] = i
            
    a = np.arange(1, N, dtype=np.int32)
    b = N - a
    pa = lpf[a]
    pb = lpf[b]
    m = np.maximum(pa, pb)
    idx = np.argmin(m)
    
    best_a = int(a[idx])
    best_b = int(b[idx])
    
    return [best_a, best_b]

# EVOLVE-BLOCK-END
