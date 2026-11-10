"""Search-based program: find (A,B) with 3 distinct ints in [1,128] maximizing |A+B| with pairwise-coprime distinct sums."""
import time
import random
from math import gcd
from itertools import combinations, product

# EVOLVE-BLOCK-START
def construct():
    deadline = time.time() + 80  # safety margin under 90s wall-clock
    random.seed(42)

    # Precompute coprime table for values in [2, 256]
    max_val = 256
    coprime = [[False] * (max_val + 1) for _ in range(max_val + 1)]
    for i in range(2, max_val + 1):
        for j in range(i, max_val + 1):
            if gcd(i, j) == 1:
                coprime[i][j] = True
                coprime[j][i] = True
    
    def distinct_sums(A, B):
        return {a + b for a in A for b in B}

    def is_pairwise_coprime(vals):
        lst = list(vals)
        n = len(lst)
        for i in range(n):
            for j in range(i + 1, n):
                if not coprime[lst[i]][lst[j]]:
                    return False
        return True

    def score_of(A, B):
        s = distinct_sums(A, B)
        if len(s) < 2:
            return -1
        if not is_pairwise_coprime(s):
            return -1
        return len(s)

    best_score = -1
    best_pair = ([1, 7, 13], [4, 10, 16]) # Seed fallback

    # Strategy:
    # 1. Try to find 9 distinct pairwise coprime sums.
    # 2. Use random search with the fast coprime check.
    
    for iteration in range(1000000):
        if time.time() > deadline:
            break
        
        # Generate random A and B
        A = random.sample(range(1, 129), 3)
        B = random.sample(range(1, 129), 3)
        
        s = distinct_sums(A, B)
        if len(s) < 9:
            continue
        
        if is_pairwise_coprime(s):
            best_score = 9
            best_pair = (A, B)
            break
            
    # If not found, try structured search with spread-out A
    if best_score < 9:
        patterns = [
            [1, 2, 4], [1, 3, 7], [1, 5, 11], [1, 10, 21], 
            [1, 20, 41], [1, 30, 61], [1, 40, 81], [1, 50, 101]
        ]
        for A in patterns:
            if time.time() > deadline:
                break
            for _ in range(50000):
                if time.time() > deadline:
                    break
                B = random.sample(range(1, 129), 3)
                s = distinct_sums(A, B)
                if len(s) == 9 and is_pairwise_coprime(s):
                    best_score = 9
                    best_pair = (A, B)
                    break
            if best_score == 9:
                break
                
    # Fallback to best found or seed
    if best_score < 0:
        final_A, final_B = [1, 7, 13], [4, 10, 16]
    else:
        final_A, final_B = best_pair
        
    return [final_A, final_B]
# EVOLVE-BLOCK-END
