"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import itertools
from fractions import Fraction

def _check_dissociated(subset):
    """Check if all reciprocal subset sums are distinct."""
    n = len(subset)
    # There are 2^n subset sums. We need all to be distinct.
    # Use Fraction for exact arithmetic.
    sums = set()
    # Iterate over all subsets
    # For efficiency, we can use bit manipulation or itertools
    # Since n is at most ~15-20, 2^n is manageable for small n.
    # For n=10, 1024 sums. For n=15, 32768 sums.
    
    # Optimization: if n is large, this might be slow. 
    # But we only call this for candidates.
    
    # We can prune early if we find a collision.
    # Also, note that if we have a collision, it means there exist two distinct subsets A and B
    # such that sum_{i in A} 1/x_i == sum_{j in B} 1/x_j.
    # This is equivalent to sum_{i in A \ B} 1/x_i == sum_{j in B \ A} 1/x_j.
    
    # Let's just compute all subset sums.
    # To speed up, we can sort and use a recursive generation with pruning? 
    # No, just brute force for small n.
    
    # Actually, for n up to 12, 2^12 = 4096, which is very fast.
    # For n=15, 32768, still fast.
    # For n=20, 1 million, might take a second or two.
    
    # We'll use a set of Fractions.
    # To make it faster, we can use a common denominator?
    # LCM of 1..22 is large.
    
    # Let's just use Fraction.
    
    # We can generate subset sums recursively to allow early termination?
    # No, set insertion is O(1) average.
    
    # Let's try a different approach: 
    # If the set is dissociated, then no non-empty subset has sum 0? 
    # No, that's for signed sums.
    
    # Standard definition: A set S is dissociated if all subset sums are distinct.
    # This is equivalent to: the only solution to sum_{i} c_i / x_i = 0 with c_i in {-1, 0, 1}
    # is the trivial one? No.
    # If sum_{A} 1/x_i = sum_{B} 1/x_j, then sum_{A\B} 1/x_i = sum_{B\A} 1/x_j.
    # Let C = A\B and D = B\A. Then sum_{C} 1/x_i - sum_{D} 1/x_j = 0.
    # So we are looking for a non-trivial linear combination with coefficients in {-1, 0, 1}
    # that sums to 0.
    
    # So we can just check all 3^n combinations? No, 2^n is better.
    
    # Let's stick to 2^n subset sums.
    
    # Optimization: 
    # We can sort the fractions? No.
    
    # Let's just do it.
    subset_sums = set()
    # Use iterative bit masking
    for mask in range(1 << n):
        s = Fraction(0)
        # Only iterate over set bits?
        # For n<=20, this is fine.
        for i in range(n):
            if mask & (1 << i):
                s += Fraction(1, subset[i])
        
        if s in subset_sums:
            return False
        subset_sums.add(s)
        
    return True

def construct():
    # We want to maximize cardinality of a subset of [1, 22].
    # The current seed has 9 elements.
    # Can we have 10? 11? 12?
    
    # Strategy: 
    # 1. Try to find a large set greedily or via search.
    # 2. Start with a known good set and try to add elements.
    # 3. Or, search for a set of size k from largest k down to 1.
    
    # Since the search space is C(22, k), for k=10 it's 646646.
    # For each candidate, checking takes 2^k operations.
    # 646646 * 1024 ~ 660 million operations. Too slow in Python.
    
    # We need a smarter search.
    
    # Observation: 
    # If we pick large numbers, the reciprocal sums are smaller and more likely to be distinct?
    # Actually, if we pick powers of 2, they are dissociated. 
    # 1, 2, 4, 8, 16 are in [1, 22]. That's 5 elements.
    # 1, 2, 4, 8, 16 -> 5 elements.
    # Can we add 3? 
    # {1, 2, 3, 4, 8, 16}?
    # 1/1 = 1. 1/2+1/2? No.
    # 1/3+1/6? No 6.
    # Let's check {1, 2, 3, 4, 8, 16}.
    # Sums:
    # 1/1 = 1
    # 1/2 = 0.5
    # 1/3 = 0.333
    # 1/4 = 0.25
    # 1/8 = 0.125
    # 1/16 = 0.0625
    # 1/2+1/4 = 0.75
    # 1/4+1/8 = 0.375
    # 1/8+1/16 = 0.1875
    # 1/2+1/8 = 0.625
    # 1/4+1/16 = 0.3125
    # 1/8+1/3? 0.4583
    # 1/2+1/3 = 0.8333
    # 1/4+1/3 = 0.5833
    # 1/8+1/3 = 0.4583
    # 1/16+1/3 = 0.3958
    # 1/2+1/4+1/8 = 0.875
    # 1/4+1/8+1/16 = 0.4375
    # 1/8+1/16+1/3? 
    # It seems likely to be dissociated.
    
    # Let's try to build a larger set.
    # A known result: The maximum size of a dissociated subset of {1, ..., n} is roughly log2(n) * something?
    # Actually, for reciprocals, the set {1, 2, 4, ..., 2^k} works.
    # In [1, 22], we have 1, 2, 4, 8, 16. (5 elements)
    # Can we add more?
    # Try adding 3. {1, 2, 3, 4, 8, 16}. 
    # Let's check if this is dissociated.
    
    # Let's try to find the maximum size by searching for subsets of size k.
    # We'll start with k=12 and go down.
    # But C(22, 12) is huge.
    
    # Better approach: Greedy with backtracking?
    # Or, use the fact that if we sort the numbers, we can check incrementally.
    
    # Let's try to construct a set of size 10 or 11.
    # Candidate: {1, 2, 3, 5, 7, 11, 13, 17, 19, 21}?
    # Let's test a few promising sets.
    
    # Heuristic: Pick numbers that are "far apart" in terms of their reciprocal values?
    # Or pick primes?
    
    # Let's try a randomized search with local optimization.
    # Start with a random set of size k, check if dissociated.
    # If not, try to swap an element.
    
    # But we need to be fast.
    
    # Let's try a specific set:
    # {1, 2, 3, 5, 7, 11, 13, 17, 19, 21}
    # Size 10.
    
    # Let's check if we can do 11.
    # {1, 2, 3, 5, 7, 11, 13, 17, 19, 21, 23}? 23 > 22.
    # {1, 2, 3, 5, 7, 11, 13, 17, 19, 20, 21}?
    
    # Let's write a function to check a specific set.
    
    def is_dissociated_fast(subset):
        # Use a more efficient check?
        # For now, use the slow one for verification.
        return _check_dissociated(subset)
    
    # Let's try some candidate sets of size 10 and 11.
    
    # Candidate 1: The seed set plus 21
    seed = [1, 2, 3, 5, 7, 11, 13, 17, 19]
    cand1 = seed + [21]
    if is_dissociated_fast(cand1):
        return cand1
    
    # Candidate 2: Seed set plus 20
    cand2 = seed + [20]
    if is_dissociated_fast(cand2):
        return cand2
    
    # Candidate 3: Seed set plus 4
    cand3 = seed + [4]
    if is_dissociated_fast(cand3):
        return cand3
    
    # Candidate 4: Seed set plus 6
    cand4 = seed + [6]
    if is_dissociated_fast(cand4):
        return cand4
    
    # Candidate 5: Seed set plus 8
    cand5 = seed + [8]
    if is_dissociated_fast(cand5):
        return cand5
    
    # Candidate 6: Seed set plus 9
    cand6 = seed + [9]
    if is_dissociated_fast(cand6):
        return cand6
    
    # Candidate 7: Seed set plus 10
    cand7 = seed + [10]
    if is_dissociated_fast(cand7):
        return cand7
    
    # Candidate 8: Seed set plus 12
    cand8 = seed + [12]
    if is_dissociated_fast(cand8):
        return cand8
    
    # Candidate 9: Seed set plus 14
    cand9 = seed + [14]
    if is_dissociated_fast(cand9):
        return cand9
    
    # Candidate 10: Seed set plus 15
    cand10 = seed + [15]
    if is_dissociated_fast(cand10):
        return cand10
    
    # Candidate 11: Seed set plus 16
    cand11 = seed + [16]
    if is_dissociated_fast(cand11):
        return cand11
    
    # Candidate 12: Seed set plus 18
    cand12 = seed + [18]
    if is_dissociated_fast(cand12):
        return cand12
    
    # If none of these work, try to find a different set of size 10.
    # Let's try a greedy approach: 
    # Start with empty set, try to add elements from 1 to 22.
    # But order matters.
    
    # Let's try to find a set of size 10 by searching.
    # We'll use a simple heuristic: 
    # Try to add elements in descending order of value?
    
    # Let's try a random search for a set of size 10.
    import random
    random.seed(42)
    
    for _ in range(1000):
        cand = random.sample(range(1, 23), 10)
        if is_dissociated_fast(cand):
            return cand
    
    # If we didn't find a size 10 set, return the seed set.
    return seed

# EVOLVE-BLOCK-END
