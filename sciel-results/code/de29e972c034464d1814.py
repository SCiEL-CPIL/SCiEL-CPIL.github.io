"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
from math import gcd
from itertools import combinations, product

def construct():
    """Search for A, B with 3 distinct integers in [1,128] maximizing distinct pairwise-coprime sums."""
    
    def check_pairwise_coprime(nums):
        """Check if all distinct values in nums are pairwise coprime."""
        distinct = list(set(nums))
        n = len(distinct)
        for i in range(n):
            for j in range(i + 1, n):
                if gcd(distinct[i], distinct[j]) != 1:
                    return False
        return True
    
    def get_distinct_sum_count(A, B):
        """Get count of distinct sums that are pairwise coprime."""
        sums = [a + b for a in A for b in B]
        distinct_sums = list(set(sums))
        if check_pairwise_coprime(distinct_sums):
            return len(distinct_sums)
        return 0
    
    # Strategy: Use a structured search based on number theory insights
    # Key insight: To maximize distinct pairwise-coprime sums, we want:
    # 1. Sums to be spread out (maximize distinct count)
    # 2. Sums to share no common factors (pairwise coprime)
    
    # Let's try a systematic approach:
    # - Use small numbers for one set to keep sums manageable
    # - Use numbers with different prime factors to ensure coprimality
    
    # Precompute gcd matrix for quick lookup
    max_val = 256  # Max possible sum is 128+128=256
    
    # First, let's try a more intelligent search
    # Focus on sets where elements have different prime factorizations
    
    # Try sets with elements that are powers of different primes or have distinct prime factors
    # For example: A = {1, 2, 3}, B = {4, 8, 16} -> sums: 5, 9, 17, 6, 10, 18, 7, 11, 19
    # Check: 5,9,17,6,10,18,7,11,19 - not all pairwise coprime (gcd(9,6)=3)
    
    # Better: use numbers that are relatively prime to each other
    # Let's try a grid search with smart pruning
    
    best_count = 0
    best_A = None
    best_B = None
    
    # Try all combinations of 3 distinct values from 1 to 30 for A
    # Then for each A, try combinations for B
    # This is C(30,3)^2 = 4060^2 ≈ 16.5 million - might be too slow
    
    # Let's be smarter: limit to specific patterns
    
    # Pattern 1: A has small numbers, B has larger numbers
    # Pattern 2: Both have well-spaced numbers
    
    # Let's try a two-phase approach:
    # Phase 1: Try small ranges with full search
    # Phase 2: Expand if needed
    
    # Phase 1: A from 1-15, B from 1-15
    for A in combinations(range(1, 16), 3):
        for B in combinations(range(1, 16), 3):
            count = get_distinct_sum_count(list(A), list(B))
            if count > best_count:
                best_count = count
                best_A = list(A)
                best_B = list(B)
    
    # Phase 2: Try larger ranges with more targeted search
    # Use the best found so far as a starting point for local search
    
    import random
    random.seed(42)
    
    # Local search from best solution
    for iteration in range(100000):
        # Pick a random set to modify
        if random.random() < 0.5:
            A = best_A[:]
            idx = random.randint(0, 2)
            A[idx] = random.randint(1, 128)
            B = best_B
        else:
            A = best_A
            B = best_B[:]
            idx = random.randint(0, 2)
            B[idx] = random.randint(1, 128)
        
        # Ensure distinct elements
        if len(set(A)) != 3 or len(set(B)) != 3:
            continue
        if min(A) < 1 or max(A) > 128 or min(B) < 1 or max(B) > 128:
            continue
        
        count = get_distinct_sum_count(A, B)
        if count > best_count:
            best_count = count
            best_A = A[:]
            best_B = B[:]
    
    return [best_A, best_B]
# EVOLVE-BLOCK-END
