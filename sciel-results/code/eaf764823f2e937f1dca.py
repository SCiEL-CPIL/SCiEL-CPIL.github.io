"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
from math import gcd
from itertools import combinations

# EVOLVE-BLOCK-START
def construct():
    deadline = time.time() + 85  # safety margin before 90s kill
    
    best_score = 0
    best_A = [1, 7, 13]
    best_B = [4, 10, 16]
    
    # Precompute coprime pairs for numbers up to 256 (max sum = 128+128)
    # Actually max sum is 128+128=256, min is 1+1=2
    max_val = 256
    
    def all_pairwise_coprime(values):
        """Check if all distinct values are pairwise coprime."""
        vals = list(set(values))
        for i in range(len(vals)):
            for j in range(i+1, len(vals)):
                if gcd(vals[i], vals[j]) != 1:
                    return False
        return True
    
    # Search over all possible 3-element sets A and B
    # To make this tractable, we'll be smart about the search
    
    # Strategy: iterate over all possible A sets, then for each A,
    # iterate over possible B sets and check the sumset condition
    
    # There are C(128,3) ≈ 345k possible sets for A and same for B
    # That's too many to do naively. Let's be smarter.
    
    # Key insight: the sumset A+B has at most 9 elements.
    # We need all distinct sums to be pairwise coprime.
    
    # Let's try a more structured search:
    # For each candidate A, generate all possible sumsets with each B,
    # and track the best.
    
    # Actually, let's just do a direct search but with early termination.
    # C(128,3) = 345,440 for each set. Total pairs would be ~1.2e11 which is too many.
    
    # Better approach: fix A and search B, or use random sampling.
    
    # Let's try: for each A (from a subset), try many B's.
    # Or: random sampling with a time budget.
    
    import random
    random.seed(42)
    
    # First, do a systematic search over smaller ranges to find good candidates
    # Then refine.
    
    # Let's try all A where elements are from a restricted set,
    # and all B similarly, but that's still large.
    
    # Alternative: enumerate all 3-element subsets of [1,128] for A,
    # and for each, enumerate all 3-element subsets for B.
    # That's C(128,3)^2 ≈ 1.2e11 - way too many.
    
    # Let's use a smarter approach:
    # 1. Generate candidate sumsets that are pairwise coprime
    # 2. For each candidate sumset, check if it can be expressed as A+B
    #    with |A|=|B|=3
    
    # Actually, let's just do random search with a time budget.
    
    def evaluate(A, B):
        sums = [a + b for a in A for b in B]
        distinct = list(set(sums))
        if all_pairwise_coprime(distinct):
            return len(distinct)
        return 0
    
    # Random search
    start_time = time.time()
    attempts = 0
    while time.time() < deadline:
        # Random A
        A = sorted(random.sample(range(1, 129), 3))
        B = sorted(random.sample(range(1, 129), 3))
        
        score = evaluate(A, B)
        if score > best_score:
            best_score = score
            best_A = A
            best_B = B
            if best_score == 9:  # maximum possible
                break
        attempts += 1
    
    return [best_A, best_B]
# EVOLVE-BLOCK-END
