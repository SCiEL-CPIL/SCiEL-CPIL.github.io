"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from math import isqrt
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = start_time + 135.0  # Safety margin before 150s kill
    
    def gen_powerful(L):
        """Generate all powerful numbers <= L using a^2 * b^3 representation."""
        powerful_set = set()
        powerful_set.add(1)
        max_b = int(L ** (1/3)) + 2
        for b in range(1, max_b + 1):
            b3 = b ** 3
            if b3 > L:
                break
            max_a = isqrt(L // b3)
            for a in range(1, max_a + 1):
                n = a * a * b3
                if n <= L:
                    powerful_set.add(n)
        return sorted(powerful_set)
    
    def count_reps(N, powerful, powerful_set):
        """Count pairs (a,b) with a<=b, a+b=N, both powerful, using two pointers."""
        count = 0
        i = 0
        j = len(powerful) - 1
        while i <= j:
            s = powerful[i] + powerful[j]
            if s == N:
                if powerful[i] <= powerful[j]:
                    count += 1
                i += 1
                j -= 1
            elif s < N:
                i += 1
            else:
                j -= 1
        return count
    
    def find_best_in_range(powerful, powerful_set, N_lo, N_hi, time_limit):
        """Find N in [N_lo, N_hi] with most powerful representations."""
        best_N = N_lo
        best_count = 0
        n_p = len(powerful)
        
        # Candidate N values: sums of consecutive powerful numbers, and strategic picks
        # First pass: sample candidate N values
        candidates = set()
        step = max(1, (N_hi - N_lo) // 2000)
        for N in range(N_lo, N_hi + 1, step):
            candidates.add(N)
        
        # Also add N values that are sums of small powerful numbers (likely to have many reps)
        for i in range(min(50, n_p)):
            for j in range(i, min(50, n_p)):
                s = powerful[i] + powerful[j]
                if N_lo <= s <= N_hi:
                    candidates.add(s)
        
        # Evaluate candidates
        for N in sorted(candidates):
            if time.time() > time_limit:
                break
            c = count_reps(N, powerful, powerful_set)
            if c > best_count:
                best_count = c
                best_N = N
        
        # Refine around best_N: check neighbors
        if best_count > 0:
            for d in range(1, 200):
                for N in [best_N - d, best_N + d]:
                    if N_lo <= N <= N_hi and time.time() < time_limit:
                        c = count_reps(N, powerful, powerful_set)
                        if c > best_count:
                            best_count = c
                            best_N = N
        
        return best_N, best_count
    
    # Multi-phase search with increasing bounds
    phases = [
        (10**6, 2*10**6),
        (5*10**6, 10**7),
        (2*10**7, 5*10**7),
        (5*10**7, 1.5*10**8),
        (1.5*10**8, 4*10**8),
    ]
    
    best_N_global = 2
    best_count_global = 1
    best_pairs_global = [[1, 1]]
    
    for L, N_hi in phases:
        if time.time() > deadline - 25.0:
            break
        
        # Generate powerful numbers up to L
        t0 = time.time()
        powerful = gen_powerful(L)
        powerful_set = set(powerful)
        gen_time = time.time() - t0
        n_p = len(powerful)
        
        # N range: powerful numbers go up to L, so sums go up to 2L
        N_lo = 2
        N_hi = min(N_hi, 2 * L)
        
        # Time budget for this phase
        remaining = deadline - time.time() - 15.0
        if remaining < 5.0:
            break
        
        # Use two-pointer for candidate evaluation
        # Strategy: for each powerful number a, N = a + b where b is also powerful
        # Count all N values that appear as sums, then find the max
        # This is O(P^2) but we can be smarter:
        
        # Count all sums using a different approach:
        # For each i, for each j >= i, N = powerful[i] + powerful[j]
        # But P can be large. Let's use the two-pointer approach on candidate N values.
        
        # Better structural approach: enumerate all sums but only for a subset
        # that's likely to produce high-count N values.
        # N values with many representations tend to be "round" or have structure.
        
        # Phase 1: coarse scan - count sums for all pairs but with early termination
        count = defaultdict(int)
        n_p = len(powerful)
        
        # If P is small enough, do full O(P^2) enumeration
        if n_p * n_p < 50 * 10**6:  # ~50M operations
            for i in range(n_p):
                if time.time() > deadline - 20.0:
                    break
                a = powerful[i]
                for j in range(i, n_p):
                    b = powerful[j]
                    N = a + b
                    if N > 10**18:
                        break
                    count[N] += 1
        else:
            # For large P, use sampling + refinement
            # Sample random pairs to identify candidate N values
            import random
            random.seed(42)
            samples = 2 * 10**6
            for _ in range(samples):
                if time.time() > deadline - 20.0:
                    break
                i = random.randrange(n_p)
                j = random.randrange(i, n_p)
                N = powerful[i] + powerful[j]
                count[N] += 1
            
            # Also do systematic scan of first portion
            limit = min(n_p, 5000)
            for i in range(limit):
                a = powerful[i]
                for j in range(i, limit):
                    N = a + powerful[j]
                    count[N] += 1
        
        if not count:
            continue
        
        # Find best N from counts
        # Get top candidates
        top_candidates = sorted(count.items(), key=lambda x: -x[1])[:200]
        
        # Verify top candidates with exact two-pointer count
        for N, approx_count in top_candidates:
            if time.time() > deadline - 15.0:
                break
            exact = count_reps(N, powerful, powerful_set)
            if exact > best_count_global:
                best_count_global = exact
                best_N_global = N
                # Collect pairs
                pairs = []
                for a in powerful:
                    b = N - a
                    if b >= a and b in powerful_set:
                        pairs.append([a, b])
                best_pairs_global = pairs
        
        # Also check a few more N values around the best found
        if best_count_global > 1:
            for d in range(1, 100):
                for N in [best_N_global - d, best_N_global + d]:
                    if N < 2 or time.time() > deadline - 10.0:
                        continue
                    exact = count_reps(N, powerful, powerful_set)
                    if exact > best_count_global:
                        best_count_global = exact
                        best_N_global = N
                        pairs = []
                        for a in powerful:
                            b = N - a
                            if b >= a and b in powerful_set:
                                pairs.append([a, b])
                        best_pairs_global = pairs
    
    # Final fallback
    if best_count_global < 1:
        return {'N': 2, 'pairs': [[1, 1]]}
    
    return {'N': best_N_global, 'pairs': best_pairs_global}
# EVOLVE-BLOCK-END
