"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from math import gcd
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = start_time + 55  # 55s margin out of 90s
    
    # Fallback known valid set of 16
    fallback_set = [2, 4, 16, 3, 9, 5, 25, 7, 11, 13, 17, 19, 23, 29, 31, 37]
    
    # Candidates are 2..40
    candidates = list(range(2, 41))
    n = len(candidates)  # 39 candidates
    
    # Step 1: Compute prime exponent vectors for each candidate
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]
    prime_index = {p: i for i, p in enumerate(primes)}
    
    def prime_exponent_vector(x):
        """Return tuple of exponents for each prime in primes list."""
        vec = [0] * len(primes)
        for i, p in enumerate(primes):
            while x % p == 0:
                vec[i] += 1
                x //= p
        return tuple(vec)
    
    # Precompute exponent vectors
    exp_vectors = [prime_exponent_vector(x) for x in candidates]
    
    # Step 2: Build conflict graph
    # Two numbers conflict if their prime exponent vectors are linearly dependent
    # over the rationals (i.e., one is a rational multiple of the other)
    # This means they are powers of the same "base" number
    
    def are_linearly_dependent(v1, v2):
        """Check if two exponent vectors are linearly dependent over Q."""
        # Find first non-zero component in v1
        i1 = -1
        for i in range(len(v1)):
            if v1[i] != 0:
                i1 = i
                break
        if i1 == -1:
            return False  # v1 is zero vector
            
        # Find first non-zero component in v2
        i2 = -1
        for i in range(len(v2)):
            if v2[i] != 0:
                i2 = i
                break
        if i2 == -1:
            return False  # v2 is zero vector
            
        # Check if v2[i2] * v1[i] == v1[i2] * v2[i] for all i
        for i in range(len(v1)):
            if v2[i2] * v1[i] != v1[i2] * v2[i]:
                return False
        return True
    
    # Build adjacency list for conflict graph
    # adj[i] = set of j such that i and j conflict
    adj = [set() for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            if are_linearly_dependent(exp_vectors[i], exp_vectors[j]):
                adj[i].add(j)
                adj[j].add(i)
    
    # Step 3: Branch-and-bound for maximum independent set
    # An independent set in the conflict graph is a set of numbers where no two
    # have linearly dependent exponent vectors, which guarantees distinct subset products
    
    def max_independent_set(adj, n, deadline):
        """Find maximum independent set using branch and bound."""
        best_set = set()
        best_size = 0
        
        # Use bitmasks for efficiency
        # available_mask: bitmask of candidates that can still be added
        # current_set: set of indices in current independent set
        
        def can_add(idx, current_set, available_mask):
            """Check if idx can be added to current_set (no conflicts)."""
            # idx must be in available_mask
            if not (available_mask & (1 << idx)):
                return False
            # idx must not conflict with any in current_set
            for j in current_set:
                if j in adj[idx]:
                    return False
            return True
        
        def update_available(idx, available_mask):
            """Update available_mask after adding idx (remove idx and its neighbors)."""
            new_mask = available_mask & ~(1 << idx)
            for j in adj[idx]:
                new_mask &= ~(1 << j)
            return new_mask
        
        def branch_bound(available_mask, current_set, depth):
            nonlocal best_set, best_size
            
            if time.time() > deadline:
                return False
            
            # Pruning: if remaining possible + current size <= best_size, stop
            remaining_count = bin(available_mask).count('1')
            if len(current_set) + remaining_count <= best_size:
                return False
            
            # If no more candidates available, update best if needed
            if available_mask == 0:
                if len(current_set) > best_size:
                    best_size = len(current_set)
                    best_set = set(current_set)
                return True
            
            # Heuristic: try candidates in order of fewest conflicts first
            # to find good solutions early
            candidates_list = []
            for i in range(n):
                if available_mask & (1 << i):
                    # Count conflicts with current_set
                    conflict_count = sum(1 for j in current_set if j in adj[i])
                    candidates_list.append((conflict_count, i))
            
            # Sort by fewest conflicts first
            candidates_list.sort()
            
            for _, idx in candidates_list:
                if time.time() > deadline:
                    return False
                
                # Try adding idx
                new_available = update_available(idx, available_mask)
                new_current = current_set | {idx}
                
                # Recurse
                branch_bound(new_available, new_current, depth + 1)
                
                # Try not adding idx
                # Remove idx from available
                new_available_no = available_mask & ~(1 << idx)
                branch_bound(new_available_no, current_set, depth + 1)
            
            return True
        
        # Initialize with all candidates available
        initial_mask = (1 << n) - 1
        branch_bound(initial_mask, set(), 0)
        
        return list(best_set)
    
    # Run the branch and bound
    result_indices = max_independent_set(adj, n, deadline)
    result_set = [candidates[i] for i in result_indices]
    
    # Verify the result
    def is_valid_set(nums):
        """Verify all subset products are distinct."""
        if len(nums) != len(set(nums)):
            return False
        if any(not (2 <= x <= 40) for x in nums):
            return False
        
        # Generate all subset products
        products = set()
        k = len(nums)
        for mask in range(1 << k):
            prod = 1
            for i in range(k):
                if mask & (1 << i):
                    prod *= nums[i]
            if prod in products:
                return False
            products.add(prod)
        return True
    
    if is_valid_set(result_set) and len(result_set) > len(fallback_set):
        return result_set
    elif is_valid_set(fallback_set):
        return fallback_set
    else:
        # Should not happen, but return something valid
        return [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]

# EVOLVE-BLOCK-END
