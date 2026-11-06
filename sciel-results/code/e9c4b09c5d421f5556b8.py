import math
from itertools import combinations

def construct():
    # Precompute primes up to 256 (max sum 128+128=256)
    is_prime = [True] * 257
    is_prime[0] = is_prime[1] = False
    for i in range(2, int(256**0.5) + 1):
        if is_prime[i]:
            for j in range(i*i, 257, i):
                is_prime[j] = False

    # Helper to check if a list of numbers is pairwise coprime
    def is_pairwise_coprime(nums):
        n = len(nums)
        for i in range(n):
            for j in range(i + 1, n):
                if math.gcd(nums[i], nums[j]) != 1:
                    return False
        return True

    best_score = 0
    best_A = [1, 2, 3]
    best_B = [4, 5, 6]
    
    # Candidate B sets: Small primes and small integers
    # The directive suggests fixing B to small primes.
    # Let's include a range of small B sets.
    small_primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]
    candidate_Bs = []
    for b1 in small_primes:
        for b2 in small_primes:
            if b2 > b1:
                for b3 in small_primes:
                    if b3 > b2:
                        candidate_Bs.append([b1, b2, b3])
                        
    # Also try some non-prime small B's just in case, e.g., [1,2,3]
    candidate_Bs.append([1, 2, 3])
    candidate_Bs.append([1, 2, 4])
    candidate_Bs.append([1, 3, 5])
    candidate_Bs.append([2, 4, 6])
    
    # To ensure we don't miss anything, we could also do a quick random search
    # but the directive asks for the specific approach.
    # We will iterate through A for each B.
    
    # Optimization: Pre-filter A?
    # If we want sums to be coprime, they should ideally be primes.
    # If a+b is composite, it shares factors with other sums.
    # So we prefer A such that a+b is prime for all b in B.
    
    # Let's iterate A from 1 to 128
    A_range = range(1, 129)
    
    # We can break early if we find score 9
    found_nine = False
    
    for B in candidate_Bs:
        # For each B, iterate through combinations of A
        # To save time, we can iterate A1, A2, A3
        # But combinations is fine.
        for A in combinations(A_range, 3):
            A = list(A)
            sums = []
            valid = True
            for a in A:
                for b in B:
                    s = a + b
                    if s > 256:
                        valid = False
                        break
                    sums.append(s)
                if not valid:
                    break
            
            if not valid:
                continue
                
            distinct_sums = list(set(sums))
            score = len(distinct_sums)
            
            if score <= best_score:
                continue
                
            if is_pairwise_coprime(distinct_sums):
                best_score = score
                best_A = A
                best_B = B
                if best_score == 9:
                    return [best_A, best_B]
                    
    # Fallback: Random search if structured didn't find high score
    # (Unlikely needed if 9 exists, but good for robustness)
    import random
    for _ in range(100000):
        A = list(random.sample(range(1, 129), 3))
        B = list(random.sample(range(1, 129), 3))
        sums = [a + b for a in A for b in B]
        distinct_sums = list(set(sums))
        score = len(distinct_sums)
        if score <= best_score:
            continue
        if is_pairwise_coprime(distinct_sums):
            best_score = score
            best_A = A
            best_B = B
            if best_score == 9:
                return [best_A, best_B]

    return [list(best_A), list(best_B)]
