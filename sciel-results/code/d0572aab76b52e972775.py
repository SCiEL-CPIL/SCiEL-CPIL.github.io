import math
import time
import random
from itertools import combinations, permutations, product

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Leave 5s margin for 90s kill
    
    def is_pairwise_coprime(nums):
        unique_nums = list(set(nums))
        n = len(unique_nums)
        for i in range(n):
            for j in range(i + 1, n):
                if math.gcd(unique_nums[i], unique_nums[j]) != 1:
                    return False
        return True

    def get_score(A, B):
        sums = [a + b for a in A for b in B]
        distinct_sums = list(set(sums))
        if not is_pairwise_coprime(distinct_sums):
            return 0
        return len(distinct_sums)

    # Seed solution
    best_A = [1, 7, 13]
    best_B = [4, 10, 16]
    best_score = get_score(best_A, best_B)
    
    if best_score < 5:
        # Fallback seed
        best_A = [1, 2, 3]
        best_B = [1, 2, 3]
        best_score = get_score(best_A, best_B)

    # Strategy: 
    # 1. Exhaustive search over small ranges to find good structures.
    # 2. Random restarts with local search.
    # 3. Focus on finding 9 distinct coprime sums.

    # Precompute gcd table for speed? Maybe just use math.gcd.
    
    # Phase 1: Structured Search
    # We want sums s_ij = a_i + b_j to be distinct and pairwise coprime.
    # Let's try to find sets A and B where the sums are primes or prime powers.
    
    # Heuristic: If we pick A and B such that a_i + b_j are all distinct,
    # we need the differences a_i - a_k to not equal b_l - b_m for i!=k, l!=m
    # in a way that causes collisions? No, a_i + b_j = a_k + b_l implies a_i - a_k = b_l - b_j.
    # So we need the set of differences in A to be disjoint from the set of differences in B (excluding 0).
    
    # Let's generate candidate sets with "independent" differences.
    
    # Phase 2: Random Search with Local Improvement
    # We will perform many random restarts.
    # To be efficient, we first check if all 9 sums are distinct.
    # Then we check if they are pairwise coprime.
    
    def local_search(A, B, max_steps=100):
        cur_A = A[:]
        cur_B = B[:]
        cur_score = get_score(cur_A, cur_B)
        
        if cur_score == 0:
            return None, None, 0
            
        for _ in range(max_steps):
            if time.time() > deadline:
                break
            if cur_score == 9:
                return cur_A, cur_B, cur_score
                
            # Try to improve by changing one element
            improved = False
            # Randomly choose which set to mutate
            if random.random() < 0.5:
                idx = random.randint(0, 2)
                old_val = cur_A[idx]
                # Try a random new value
                for _ in range(10):
                    new_val = random.randint(1, 128)
                    if new_val in cur_A and new_val != old_val:
                        continue
                    cur_A[idx] = new_val
                    if len(set(cur_A)) != 3:
                        cur_A[idx] = old_val
                        continue
                        
                    new_score = get_score(cur_A, cur_B)
                    if new_score > cur_score:
                        cur_score = new_score
                        improved = True
                        break
                    else:
                        cur_A[idx] = old_val
            else:
                idx = random.randint(0, 2)
                old_val = cur_B[idx]
                for _ in range(10):
                    new_val = random.randint(1, 128)
                    if new_val in cur_B and new_val != old_val:
                        continue
                    cur_B[idx] = new_val
                    if len(set(cur_B)) != 3:
                        cur_B[idx] = old_val
                        continue
                        
                    new_score = get_score(cur_A, cur_B)
                    if new_score > cur_score:
                        cur_score = new_score
                        improved = True
                        break
                    else:
                        cur_B[idx] = old_val
                        
            if not improved:
                # If no improvement in this round, maybe break or continue?
                # Let's continue a few more rounds to escape local minima?
                # Or just break to save time.
                # Given the small space, breaking is fine.
                break
                
        return cur_A, cur_B, cur_score

    # Main Loop
    # We'll do a mix of random restarts and targeted searches.
    
    # Targeted Search: 
    # Pick 3 small primes for A? Or just random small numbers.
    # The key is that sums should be coprime.
    
    # Let's try to find a solution with 9 distinct sums first.
    # If we can find 9 distinct sums that are pairwise coprime, we win.
    
    # Heuristic for distinct sums:
    # Let A = {a, a+d1, a+d2} and B = {b, b+e1, b+e2}
    # Sums are a+b, a+b+e1, a+b+e2, a+d1+b, a+d1+b+e1, a+d1+b+e2, a+d2+b, a+d2+b+e1, a+d2+b+e2
    # These are distinct if {0, d1, d2} + {0, e1, e2} has 9 distinct elements.
    # This is true if d1 != d2, e1 != e2, and d1 != e1, d1 != e2, d2 != e1, d2 != e2 (and negatives).
    # Basically, the difference sets must be "independent".
    
    # Let's iterate over small difference sets.
    # d1, d2 in 1..20
    # e1, e2 in 1..20
    
    found_good_structure = False
    
    # Precompute valid difference pairs
    valid_diffs_A = []
    valid_diffs_B = []
    
    # We can just try random small A and B and check distinctness quickly.
    
    # Random Restart Loop
    restarts = 0
    while time.time() < deadline and restarts < 20000:
        restarts += 1
        
        # Generate random A and B
        # Bias towards small numbers to keep sums small (easier to be coprime? maybe not)
        # Actually, large numbers are harder to be coprime? No, gcd is often 1 for random large numbers.
        # But small numbers have more small prime factors.
        
        # Let's try uniform random in [1, 128]
        A = random.sample(range(1, 129), 3)
        B = random.sample(range(1, 129), 3)
        
        # Quick check: are all 9 sums distinct?
        sums = [a + b for a in A for b in B]
        if len(set(sums)) < 9:
            continue
            
        # Check pairwise coprime
        distinct_sums = list(set(sums))
        if not is_pairwise_coprime(distinct_sums):
            continue
            
        # If we get here, we have 9 distinct pairwise coprime sums!
        score = 9
        if score > best_score:
            best_score = score
            best_A = A[:]
            best_B = B[:]
            break  # Can't do better than 9
        
    # If we didn't find 9, let's try to find 8 or 7 via local search from valid (but not perfect) candidates.
    # Or just continue random search for anything with score > best_score.
    
    if best_score < 9:
        # Do local search from random valid candidates (score > 0)
        attempts = 0
        while time.time() < deadline and attempts < 1000:
            attempts += 1
            
            A = random.sample(range(1, 129), 3)
            B = random.sample(range(1, 129), 3)
            
            score = get_score(A, B)
            if score == 0:
                continue
                
            # Local search
            new_A, new_B, new_score = local_search(A, B, max_steps=50)
            
            if new_score > best_score:
                best_score = new_score
                best_A = new_A
                best_B = new_B
                
                if best_score == 9:
                    break

    return [best_A, best_B]
