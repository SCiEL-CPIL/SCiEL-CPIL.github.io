"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import sys

_START_TIME = time.time()
_DEADLINE = _START_TIME + 55.0  # Safety margin for 60s limit

def construct():
    """
    Returns a list of distinct integers in 2..40 such that all 2^k subset products are distinct.
    Maximizes the number of selected integers.
    
    Strategy:
    1. Model the problem using prime exponent vectors. Two subset products are distinct iff their exponent vectors are distinct.
    2. The condition is that all 2^k subset sums of the selected exponent vectors are distinct.
    3. Use a complete branch-and-bound search to find the maximum set size.
    4. Order candidates by 'simplicity' (fewer prime factors, smaller norm) to find valid large sets quickly.
    5. Prune branches where the maximum possible remaining size cannot exceed the current best.
    6. Since the 12 primes in 2..40 form a valid set of size 12, and it is hard to exceed 12,
       finding size 12 allows early termination if we assume 12 is optimal or close to it.
       Actually, we should search for > 12 if possible.
    """
    
    candidates = list(range(2, 41))
    
    # 1. Precompute prime exponent vectors for each candidate
    # Primes up to 40: 2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]
    num_primes = len(primes)
    
    def get_vector(n):
        v = [0] * num_primes
        for i, p in enumerate(primes):
            while n % p == 0:
                v[i] += 1
                n //= p
        return tuple(v)
        
    vectors = {x: get_vector(x) for x in candidates}
    
    # 2. Sort candidates by a heuristic that favors "simple" vectors.
    # Simple vectors (primes) have a single 1 and rest 0s.
    # They are less likely to cause collisions because they only increment one dimension.
    # Composite numbers increment multiple dimensions or a single dimension by >1.
    # Incrementing by >1 is "dangerous" because it might skip over a sum?
    # Actually, the condition is that all subset sums are distinct.
    # This is equivalent to the vectors being linearly independent over Z with coefficients 0/1?
    # No, it's a stronger condition: the set must be "sum-distinct" or "Sidon-like" in the group sense.
    
    # Heuristic: Sort by number of non-zero components (support size), then by norm.
    # Primes have support size 1.
    # Squares of primes have support size 1, norm 2.
    # Products of two primes have support size 2.
    
    def complexity(v):
        support = sum(1 for x in v if x > 0)
        norm = sum(v)
        return (support, norm)
        
    sorted_candidates = sorted(candidates, key=lambda x: complexity(vectors[x]))
    
    best_size = 0
    best_set = []
    
    # 3. Branch and Bound
    # We want to maximize the size.
    # Upper bound: number of remaining candidates + current size.
    
    def backtrack(start_idx, current_set, current_sums, current_size):
        nonlocal best_set, best_size
        
        if time.time() > _DEADLINE:
            return
            
        if current_size > best_size:
            best_size = current_size
            best_set = list(current_set)
            
        # Pruning:
        # If we can't beat the best size, return.
        remaining = len(sorted_candidates) - start_idx
        if current_size + remaining <= best_size:
            return
            
        # Optional: If we found a set of size 12 (all primes), and we suspect 12 is max,
        # we could stop. But let's try to find 13.
        
        for i in range(start_idx, len(sorted_candidates)):
            if time.time() > _DEADLINE:
                return
                
            x = sorted_candidates[i]
            v = vectors[x]
            
            # Check collision
            can_add_flag = True
            # Optimization: Check if v + s is in current_sums for any s in current_sums.
            # This is O(2^k * 12) per check.
            for s in current_sums:
                nv = tuple(s[j] + v[j] for j in range(num_primes))
                if nv in current_sums:
                    can_add_flag = False
                    break
                    
            if can_add_flag:
                # Add
                new_sums = set(current_sums)
                for s in current_sums:
                    nv = tuple(s[j] + v[j] for j in range(num_primes))
                    new_sums.add(nv)
                    
                current_set.append(x)
                backtrack(i + 1, current_set, new_sums, current_size + 1)
                current_set.pop()
                
    initial_sums = {tuple([0]*num_primes)}
    backtrack(0, [], initial_sums, 0)
    
    # Ensure types are correct
    return [int(x) for x in best_set]
# EVOLVE-BLOCK-END
