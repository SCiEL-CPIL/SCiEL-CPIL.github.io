import time
import math
from itertools import combinations, product

def construct():
    start_time = time.time()
    deadline = start_time + 85

    # Precompute prime factor masks for numbers 1..256
    # Primes up to 256
    primes = []
    is_prime = [True] * 257
    is_prime[0] = is_prime[1] = False
    for i in range(2, 257):
        if is_prime[i]:
            primes.append(i)
            for j in range(i*i, 257, i):
                is_prime[j] = False
    
    # Map prime to bit index
    prime_to_idx = {p: i for i, p in enumerate(primes)}
    
    pf_mask = [0] * 257
    for n in range(1, 257):
        mask = 0
        for p in primes:
            if n % p == 0:
                mask |= (1 << prime_to_idx[p])
        pf_mask[n] = mask

    # Check if two numbers are coprime
    def is_coprime(a, b):
        return (pf_mask[a] & pf_mask[b]) == 0

    # Check if a list of numbers is pairwise coprime
    def is_pairwise_coprime(nums):
        for i in range(len(nums)):
            for j in range(i+1, len(nums)):
                if not is_coprime(nums[i], nums[j]):
                    return False
        return True

    # Check a specific A, B pair
    def check(A, B):
        sums = []
        for a in A:
            for b in B:
                sums.append(a + b)
        
        # Check distinct
        if len(set(sums)) != 9:
            return 0
            
        # Check pairwise coprime using masks
        if not is_pairwise_coprime(sums):
            return 0
        return 9

    # Best known solution
    best_A = [1, 7, 13]
    best_B = [4, 10, 16]
    best_score = check(best_A, best_B)
    if best_score == 9:
        return [best_A, best_B]

    # Strategy: 
    # 1. Try to find 9 distinct pairwise coprime sums.
    # 2. Use a smarter search: 
    #    - Pick A first (smaller search space if we restrict A).
    #    - For each A, pick B such that A+B are distinct and coprime.
    #    
    # Optimization: 
    # We can precompute for each pair (a, b) if a+b is coprime to all other potential sums?
    # That's hard because the "other potential sums" depend on the full set.
    
    # Alternative: Backtracking on B given A.
    # For a fixed A = [a1, a2, a3], we want to pick B = [b1, b2, b3] from candidates.
    # The sums are S = {a1+b1, a1+b2, a1+b3, a2+b1, ..., a3+b3}.
    # We need S to be pairwise coprime and distinct.
    
    # Let's iterate A from a restricted set of "good" candidates.
    # "Good" candidates for A: numbers that don't introduce too many small prime factors?
    # Actually, 1 is very good because gcd(1+x, y) is often 1 if y is prime or coprime to 1+x.
    
    # Let's try a heuristic: 
    # Iterate A from combinations of [1..50] (or so).
    # For each A, iterate B from combinations of [1..128].
    # But 50C3 = 19600. 128C3 = 347k. Product is ~6e9. Too slow.
    
    # We need to prune B selection.
    # For fixed A, let's pick b1, b2, b3 one by one.
    # When picking b_k, we can check if the new sums introduced by b_k are coprime to existing sums.
    # Existing sums depend on previous b's.
    
    # Let's implement a backtracking search for B given A.
    
    def find_B_for_A(A, candidates, deadline):
        # candidates: list of integers to choose from for B
        # We want to pick 3 distinct integers from candidates.
        
        # Precompute sums for a given B element b:
        # sums_b = [a + b for a in A]
        
        # We will pick b1, then b2, then b3.
        # State: current_B, current_sums (list of sums so far), current_mask_union?
        # Actually, we just need to check pairwise coprimality of the 9 sums.
        
        # To speed up, we can precompute for each b in candidates:
        # b_sums[b] = [a+b for a in A]
        
        # Let's try a few B candidates.
        # We can order candidates by some heuristic?
        
        # Iterate b1 in candidates
        for b1 in candidates:
            if time.time() > deadline:
                return None
            sums1 = [a + b1 for a in A]
            if len(set(sums1)) != 3:
                continue
            if not is_pairwise_coprime(sums1):
                continue
            
            # Iterate b2 in candidates (b2 != b1)
            for b2 in candidates:
                if b2 == b1:
                    continue
                if time.time() > deadline:
                    return None
                sums2 = [a + b2 for a in A]
                if len(set(sums2)) != 3:
                    continue
                # Check sums2 against sums1
                # All 6 sums must be pairwise coprime
                combined1 = sums1 + sums2
                if len(set(combined1)) != 6:
                    continue
                if not is_pairwise_coprime(combined1):
                    continue
                
                # Iterate b3 in candidates (b3 != b1, b3 != b2)
                for b3 in candidates:
                    if b3 == b1 or b3 == b2:
                        continue
                    if time.time() > deadline:
                        return None
                    sums3 = [a + b3 for a in A]
                    if len(set(sums3)) != 3:
                        continue
                    combined2 = combined1 + sums3
                    if len(set(combined2)) != 9:
                        continue
                    if not is_pairwise_coprime(combined2):
                        continue
                    
                    return [b1, b2, b3]
        return None

    # Search A
    # Let's restrict A to a smaller set to make the inner loop faster?
    # Or just iterate A from 1..60?
    # 60C3 = 34220.
    # For each A, the inner loop is O(N^3) where N is size of candidates for B.
    # If we use B candidates from 1..128, N=128. 128^3 = 2M.
    # 34220 * 2M is too slow.
    
    # We need to prune B candidates for each A.
    # Idea: Filter B candidates based on A.
    # For a fixed A, a candidate b is "bad" if a+b shares a factor with some "likely" other sum?
    # Hard to predict.
    
    # Alternative:
    # Iterate B first? Same complexity.
    
    # Let's try a different approach:
    # Randomized search with local improvement?
    # Or just a smarter deterministic search.
    
    # Let's try restricting A and B to smaller ranges first, then expand.
    # Range 1..30: 30C3 = 4060. 4060 * (30^3=27000) = 1e8. Might be okay in Python if optimized?
    # 1e8 operations in Python is ~10-20s.
    
    # Let's try range 1..40.
    
    for limit in [30, 40, 50, 60, 80, 128]:
        if time.time() > deadline:
            break
        candidates = list(range(1, limit + 1))
        
        # We can pre-filter candidates for B based on A?
        # Let's just run the backtracking for each A.
        
        for A in combinations(candidates, 3):
            if time.time() > deadline:
                break
            A = list(A)
            
            # Quick check: if A has duplicates (no, combinations ensures distinct)
            
            # Try to find B
            B = find_B_for_A(A, candidates, deadline)
            if B:
                score = check(A, B)
                if score > best_score:
                    best_score = score
                    best_A = A
                    best_B = B
                    if best_score == 9:
                        return [best_A, best_B]
                        
    return [best_A, best_B]
