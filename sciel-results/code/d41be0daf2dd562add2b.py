"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import itertools
from math import gcd
from functools import reduce

def _pairwise_coprime(nums):
    """Check if all distinct values in nums are pairwise coprime."""
    distinct = list(set(nums))
    n = len(distinct)
    for i in range(n):
        for j in range(i + 1, n):
            if gcd(distinct[i], distinct[j]) != 1:
                return False
    return True

def construct():
    best_A = None
    best_B = None
    best_count = 0
    
    # Search space: integers in [1, 128]
    # We'll do a structured search
    # First try to find solutions with high cardinality
    
    # Generate candidate sets
    # To make this tractable, we'll sample intelligently
    # Use primes and prime-related numbers for the summands
    
    # Strategy: iterate over A from combinations, for each A iterate over B
    # But limit the search space intelligently
    
    # Let's try a randomized approach with many iterations
    import random
    random.seed(42)
    
    range_vals = list(range(1, 129))
    
    # First, do a targeted search with smaller ranges to find good candidates
    # Then expand
    
    # Try systematic search with A in [1, 40] and B in [1, 40] first
    small_range = list(range(1, 41))
    
    for A in itertools.combinations(small_range, 3):
        A = list(A)
        for B in itertools.combinations(small_range, 3):
            B = list(B)
            sums = [a + b for a in A for b in B]
            distinct_sums = set(sums)
            count = len(distinct_sums)
            if count > best_count:
                if _pairwise_coprime(distinct_sums):
                    best_count = count
                    best_A = A
                    best_B = B
                    if best_count == 9:
                        return [best_A, best_B]
    
    # If not found in small range, try larger
    if best_count < 9:
        # Try with some random sampling in larger range
        for _ in range(50000):
            A = sorted(random.sample(range_vals, 3))
            B = sorted(random.sample(range_vals, 3))
            sums = [a + b for a in A for b in B]
            distinct_sums = set(sums)
            count = len(distinct_sums)
            if count > best_count:
                if _pairwise_coprime(distinct_sums):
                    best_count = count
                    best_A = A
                    best_B = B
                    if best_count == 9:
                        return [best_A, best_B]
    
    if best_A is None:
        # Fallback to known solution
        return [[1, 7, 13], [4, 10, 16]]
    
    return [best_A, best_B]
# EVOLVE-BLOCK-END
