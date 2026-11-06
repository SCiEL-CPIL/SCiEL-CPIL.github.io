"""Search for optimal pairwise-coprime sumset construction."""
import itertools
import math
import time
import random

def construct():
    # Safety margin: evaluator kills at 90s, so we aim to finish by 50s.
    deadline = time.time() + 50.0
    
    # Precompute gcd for speed? math.gcd is C-implemented and fast.
    
    def check_pairwise_coprime(vals):
        """Check if all values in vals are pairwise coprime."""
        n = len(vals)
        for i in range(n):
            vi = vals[i]
            for j in range(i + 1, n):
                if math.gcd(vi, vals[j]) != 1:
                    return False
        return True

    def evaluate_pair(A, B):
        """Returns (count, is_valid) for the pair A, B."""
        sums = set()
        for a in A:
            for b in B:
                sums.add(a + b)
        
        count = len(sums)
        if count < 2:
            return count, False
        
        # Only check coprime if count is potentially useful
        # We want to maximize count, so we check validity for any count > current best
        vals = sorted(sums)
        if check_pairwise_coprime(vals):
            return count, True
        return count, False

    best_count = 0
    best_A = [1, 7, 13]
    best_B = [4, 10, 16]
    
    # Initialize with a known valid pair if possible, or just 0
    # Let's quickly check the seed from the prompt description
    # A=[1,7,13], B=[4,10,16] -> Sums: 5,11,17, 11,17,23, 17,23,29
    # Distinct: {5, 11, 17, 23, 29}
    # GCDs: 
    # gcd(5,11)=1, gcd(5,17)=1, gcd(5,23)=1, gcd(5,29)=1
    # gcd(11,17)=1, gcd(11,23)=1, gcd(11,29)=1
    # gcd(17,23)=1, gcd(17,29)=1
    # gcd(23,29)=1
    # All pairwise coprime. Count = 5.
    best_count = 5
    best_A = [1, 7, 13]
    best_B = [4, 10, 16]

    # Phase 1: Exhaustive search on a small range [1, 20]
    # C(20, 3) = 1140 combinations.
    # Pairs = 1140 * 1140 = 1,299,600.
    # This should be fast enough in Python.
    
    small_range = range(1, 21)
    sets_small = list(itertools.combinations(small_range, 3))
    
    for A in sets_small:
        if time.time() > deadline:
            break
        for B in sets_small:
            count, valid = evaluate_pair(A, B)
            if valid and count > best_count:
                best_count = count
                best_A = list(A)
                best_B = list(B)
                if best_count == 9:
                    return [best_A, best_B]
    
    # Phase 2: If we didn't find 9, or want to explore larger numbers, do random search.
    # We might find a 9-count set with larger numbers, or maybe 8-count if 9 is impossible?
    # 9 is the max. If we found 9, we returned.
    # If we found < 9, we try to find 9 with random search on larger range.
    
    if best_count < 9:
        random.seed(42)
        # Try random pairs from [1, 128]
        # Limit iterations to ensure we don't timeout
        max_iter = 200000
        for _ in range(max_iter):
            if time.time() > deadline - 2.0: # Stop 2s before deadline
                break
            
            A = random.sample(range(1, 129), 3)
            B = random.sample(range(1, 129), 3)
            A.sort()
            B.sort()
            
            count, valid = evaluate_pair(A, B)
            if valid and count > best_count:
                best_count = count
                best_A = A
                best_B = B
                if best_count == 9:
                    return [best_A, best_B]

    return [best_A, best_B]
