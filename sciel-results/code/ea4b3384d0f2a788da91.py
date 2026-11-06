import itertools
import math
import time
import random

def construct():
    """
    Returns two lists A, B, each containing exactly 3 distinct integers in [1, 128].
    All DISTINCT values in A+B must be pairwise coprime.
    Maximizes the cardinality of A+B.
    """
    t_start = time.time()
    TIME_LIMIT = 85.0  # Safety margin before 90s kill

    max_val = 256
    
    # Sieve of Eratosthenes for quick prime check
    is_prime = [True] * (max_val + 1)
    is_prime[0] = is_prime[1] = False
    for i in range(2, int(math.sqrt(max_val)) + 1):
        if is_prime[i]:
            for j in range(i*i, max_val + 1, i):
                is_prime[j] = False
            
    def is_pairwise_coprime(sums_set):
        """Check if all distinct sums are pairwise coprime."""
        sums_list = list(sums_set)
        n = len(sums_list)
        for i in range(n):
            si = sums_list[i]
            if si == 1:
                continue
            for j in range(i + 1, n):
                sj = sums_list[j]
                if math.gcd(si, sj) != 1:
                    return False
        return True

    best_A = None
    best_B = None
    best_size = 0

    # Strategy 1: Targeted search for prime-sum grids (K_{3,3} in prime-sum graph).
    # If all 9 sums are prime, they are distinct and pairwise coprime.
    # To get odd primes, we can use A=evens, B=odds (or vice versa).
    
    even_candidates = [i for i in range(2, 129, 2)]
    odd_candidates = [i for i in range(1, 129, 2)]
    
    # We want to find A (size 3) and B (size 3) such that all a+b are prime.
    # This is equivalent to finding a K_{3,3} in the bipartite graph where
    # edges exist if sum is prime.
    
    # Optimization: Pre-filter candidates.
    # For a fixed A, B must be such that all b in B form a clique in the
    # intersection of neighborhoods of A's elements.
    
    # Let's iterate A from a smaller set first to find a quick win.
    # If we find size 9, we return immediately.
    
    # Search A in even_candidates, B in odd_candidates
    # To make this efficient, we can iterate over A and check if there are 3 B's.
    
    # Precompute for each a in even_candidates, the set of b in odd_candidates such that a+b is prime.
    # Since we want to maximize search space coverage within time, we'll just do a nested loop
    # but with early termination and maybe smaller ranges first.
    
    # Let's try a structured search:
    # Iterate A combinations from a subset of even_candidates.
    # If we can't find 9, we fall back to general search.
    
    # To speed up, let's restrict the initial search to smaller numbers.
    # If a+b is prime, and a,b are small, the prime is small.
    
    # Let's try all combinations of 3 from even_candidates[0:40] (2..80)
    # and check if there exist 3 from odd_candidates[0:40] (1..79) that work.
    
    # Actually, let's just do the full nested loop but break early if we find 9.
    # C(64, 3) * C(64, 3) is roughly 40k * 40k = 1.6 billion. Too slow.
    
    # Better: For each A, find valid B candidates.
    # valid_B_for_A = intersection of neighbors(a) for a in A.
    # If len(valid_B_for_A) >= 3, we have a solution.
    
    # Let's implement this.
    
    # Precompute neighbors for each a in even_candidates
    # neighbors[a] = set of b in odd_candidates such that a+b is prime
    
    # To save memory/time, we can just compute on the fly or use bitsets.
    # Given 128 elements, bitsets are efficient.
    
    # Let's map odd_candidates to indices 0..63.
    odd_to_idx = {b: i for i, b in enumerate(odd_candidates)}
    
    # neighbors_mask[a] is a bitmask of b's that work with a.
    # Since 64 bits fit in a Python int, we can use bitwise AND.
    
    neighbors_mask = {}
    for a in even_candidates:
        mask = 0
        for b in odd_candidates:
            if is_prime[a + b]:
                mask |= (1 << odd_to_idx[b])
        neighbors_mask[a] = mask
        
    # Now iterate A
    # We can iterate A in a way that prioritizes "central" values or just iterate.
    # Let's iterate all combinations of 3 from even_candidates.
    # There are C(64, 3) = 41664 combinations. This is fast.
    
    for A in itertools.combinations(even_candidates, 3):
        if time.time() - t_start > TIME_LIMIT - 10:
            break
        # Intersection of masks
        mask = neighbors_mask[A[0]] & neighbors_mask[A[1]] & neighbors_mask[A[2]]
        # Count bits
        if mask.bit_count() >= 3:
            # Find 3 indices
            b_indices = []
            m = mask
            while m and len(b_indices) < 3:
                # Find lowest set bit
                lsb = m & -m
                idx = lsb.bit_length() - 1
                b_indices.append(idx)
                m &= ~lsb
            
            B = [odd_candidates[i] for i in b_indices]
            # Verify
            sums = {a + b for a in A for b in B}
            if len(sums) == 9 and all(is_prime[s] for s in sums):
                return [list(A), B]
                
    # Also try A=odds, B=evens
    even_to_idx = {b: i for i, b in enumerate(even_candidates)}
    neighbors_mask_oe = {}
    for a in odd_candidates:
        mask = 0
        for b in even_candidates:
            if is_prime[a + b]:
                mask |= (1 << even_to_idx[b])
        neighbors_mask_oe[a] = mask
        
    for A in itertools.combinations(odd_candidates, 3):
        if time.time() - t_start > TIME_LIMIT - 10:
            break
        mask = neighbors_mask_oe[A[0]] & neighbors_mask_oe[A[1]] & neighbors_mask_oe[A[2]]
        if mask.bit_count() >= 3:
            b_indices = []
            m = mask
            while m and len(b_indices) < 3:
                lsb = m & -m
                idx = lsb.bit_length() - 1
                b_indices.append(idx)
                m &= ~lsb
            
            B = [even_candidates[i] for i in b_indices]
            sums = {a + b for a in A for b in B}
            if len(sums) == 9 and all(is_prime[s] for s in sums):
                return [list(A), B]

    # If we didn't find a prime-sum grid (size 9), we search for pairwise coprime sums (size 8, 7, etc.)
    # Strategy 2: General pairwise coprime check on small mixed sets.
    
    small_range = range(1, 51)
    for A in itertools.combinations(small_range, 3):
        if time.time() - t_start > TIME_LIMIT - 5:
            break
        for B in itertools.combinations(small_range, 3):
            sums = {a + b for a in A for b in B}
            if is_pairwise_coprime(sums):
                if len(sums) > best_size:
                    best_size = len(sums)
                    best_A = list(A)
                    best_B = list(B)
                    if best_size == 9:
                        return [best_A, best_B]

    # Strategy 3: Random search with larger range if time permits.
    random.seed(42)
    max_iters = 300000
    for i in range(max_iters):
        if time.time() - t_start > TIME_LIMIT - 2:
            break
        A = list(random.sample(range(1, 129), 3))
        B = list(random.sample(range(1, 129), 3))
        sums = {a + b for a in A for b in B}
        if is_pairwise_coprime(sums):
            if len(sums) > best_size:
                best_size = len(sums)
                best_A = A
                best_B = B
                if best_size == 9:
                    return [best_A, best_B]

    if best_A is None:
        return [[1, 7, 13], [4, 10, 16]]
        
    return [best_A, best_B]
