"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import math
import time
import random
from itertools import permutations, combinations

def _gcd(a, b):
    while b:
        a, b = b, a % b
    return a

def _pairwise_coprime(vals):
    """Check if all distinct values are pairwise coprime."""
    distinct = list(set(vals))
    n = len(distinct)
    for i in range(n):
        for j in range(i + 1, n):
            if _gcd(distinct[i], distinct[j]) != 1:
                return False
    return True

def _evaluate(A, B):
    """Return the number of distinct pairwise-coprime sums, or 0 if not all coprime."""
    sums = set()
    for a in A:
        for b in B:
            sums.add(a + b)
    distinct = list(sums)
    if not distinct:
        return 0
    # Check pairwise coprime
    for i in range(len(distinct)):
        for j in range(i + 1, len(distinct)):
            if _gcd(distinct[i], distinct[j]) != 1:
                return 0
    return len(distinct)

def construct():
    start_time = time.time()
    deadline = start_time + 85  # 5s safety margin before 90s limit
    
    # We aim for 9 distinct pairwise coprime sums.
    # A and B must contain 3 distinct integers from [1, 128].
    
    # Known good solution from previous runs or manual search?
    # Let's try to find one systematically.
    
    # Strategy:
    # 1. Iterate over candidate A sets.
    # 2. For each A, try to find B such that all 9 sums are distinct and pairwise coprime.
    
    # To speed up, we can precompute primes or just use GCD.
    # Since 9 is small, GCD is fast.
    
    # We will use a randomized restart approach with a strong local search / construction.
    
    best_score = 0
    best_A = [1, 2, 3]
    best_B = [1, 2, 3]
    
    # Pre-compute all pairs of numbers in [1, 128] that are coprime? No, sums are up to 256.
    
    # Let's try a structured search first.
    # Fix A = [1, 2, 3] ? No, sums will be close, likely sharing factors.
    # Let's try A with large gaps to ensure sums are distinct.
    
    # Random search with smart initialization
    random.seed(42)
    
    # We'll try 10,000 random A sets, and for each, try to construct B greedily or via backtracking.
    
    for _ in range(100000):
        if time.time() > deadline:
            break
        
        # Generate random A
        A = sorted(random.sample(range(1, 129), 3))
        
        # Try to find B
        # We need b1, b2, b3 such that:
        # a1+b1, a1+b2, a1+b3, a2+b1, a2+b2, a2+b3, a3+b1, a3+b2, a3+b3
        # are all distinct and pairwise coprime.
        
        # Let's try to pick B elements one by one.
        # This is a constraint satisfaction problem.
        
        # Heuristic: Pick b1 randomly.
        # Then b2 must make a1+b2, a2+b2, a3+b2 coprime to existing sums and each other.
        # Then b3 similarly.
        
        found_B = None
        
        for b1 in random.sample(range(1, 129), 50):
            if time.time() > deadline:
                break
            
            sums1 = [A[0] + b1, A[1] + b1, A[2] + b1]
            # Check if sums1 are pairwise coprime
            if not _pairwise_coprime(sums1):
                continue
            
            for b2 in random.sample(range(1, 129), 50):
                if b2 == b1:
                    continue
                if time.time() > deadline:
                    break
                
                sums2 = [A[0] + b2, A[1] + b2, A[2] + b2]
                # Check if sums2 are pairwise coprime
                if not _pairwise_coprime(sums2):
                    continue
                
                # Check if sums2 are coprime to sums1
                all_sums_12 = sums1 + sums2
                if not _pairwise_coprime(all_sums_12):
                    continue
                
                for b3 in random.sample(range(1, 129), 50):
                    if b3 == b1 or b3 == b2:
                        continue
                    if time.time() > deadline:
                        break
                    
                    sums3 = [A[0] + b3, A[1] + b3, A[2] + b3]
                    # Check if sums3 are pairwise coprime
                    if not _pairwise_coprime(sums3):
                        continue
                    
                    # Check if sums3 are coprime to sums1 and sums2
                    all_sums_123 = all_sums_12 + sums3
                    if not _pairwise_coprime(all_sums_123):
                        continue
                    
                    # Check distinctness
                    if len(set(all_sums_123)) == 9:
                        found_B = [b1, b2, b3]
                        break
                if found_B:
                    break
            if found_B:
                break
        
        if found_B:
            score = _evaluate(A, found_B)
            if score > best_score:
                best_score = score
                best_A = A
                best_B = found_B
                if best_score == 9:
                    break
    
    # If we found 9, return it.
    # If not, try a local search on the best found so far.
    
    if best_score < 9 and time.time() < deadline:
        # Local search: swap elements in A or B to improve score
        # This is complex, so we rely on the random search being robust enough.
        # But let's add a simple hill climbing step.
        
        A_cur = list(best_A)
        B_cur = list(best_B)
        score_cur = best_score
        
        improved = True
        while improved and time.time() < deadline:
            improved = False
            # Try swapping each element in A
            for i in range(3):
                for new_val in range(1, 129):
                    if new_val in A_cur:
                        continue
                    old_val = A_cur[i]
                    A_cur[i] = new_val
                    new_score = _evaluate(A_cur, B_cur)
                    if new_score > score_cur:
                        score_cur = new_score
                        improved = True
                    else:
                        A_cur[i] = old_val
            
            # Try swapping each element in B
            for i in range(3):
                for new_val in range(1, 129):
                    if new_val in B_cur:
                        continue
                    old_val = B_cur[i]
                    B_cur[i] = new_val
                    new_score = _evaluate(A_cur, B_cur)
                    if new_score > score_cur:
                        score_cur = new_score
                        improved = True
                    else:
                        B_cur[i] = old_val
            
            if score_cur > best_score:
                best_score = score_cur
                best_A = A_cur[:]
                best_B = B_cur[:]
                
            if best_score == 9:
                break
    
    return [best_A, best_B]
# EVOLVE-BLOCK-END
