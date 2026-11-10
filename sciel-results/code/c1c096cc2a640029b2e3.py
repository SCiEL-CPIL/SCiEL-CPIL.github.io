"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from math import gcd
from itertools import combinations

def are_pairwise_coprime(nums):
    unique = list(set(nums))
    for i in range(len(unique)):
        for j in range(i + 1, len(unique)):
            if gcd(unique[i], unique[j]) != 1:
                return False
    return True

def construct():
    start_time = time.time()
    # Keep a safety margin before the 90s kill
    deadline = start_time + 85 

    # Default fallback (from seed/inspiration)
    best_A = [1, 7, 13]
    best_B = [4, 10, 16]
    best_count = 5
    best_sums = [1+4, 1+10, 1+16, 7+4, 7+10, 7+16, 13+4, 13+10, 13+16]
    best_sums_distinct = list(set(best_sums))
    if are_pairwise_coprime(best_sums_distinct):
        best_count = len(best_sums_distinct)

    # Precompute primes up to 256 (max sum 128+128)
    def is_prime(n):
        if n < 2: return False
        if n == 2: return True
        if n % 2 == 0: return False
        i = 3
        while i * i <= n:
            if n % i == 0: return False
            i += 2
        return True

    primes_set = set(x for x in range(2, 257) if is_prime(x))
    
    # To get a score of 9 (max), we need 9 distinct pairwise coprime sums.
    # If all sums are distinct primes, they are pairwise coprime.
    # Strategy: Find A, B such that all a_i + b_j are prime.
    
    # We iterate over possible A values.
    # To make the search efficient, we can fix a1, a2 and find valid b's, then filter for a3.
    # However, simply iterating all triples (a1, a2, a3) and (b1, b2, b3) is O(128^6) which is too slow.
    # 
    # Better approach:
    # 1. Iterate a1, a2, a3.
    # 2. For a given A, find all b in [1, 128] such that a1+b, a2+b, a3+b are all prime.
    # 3. If there are at least 3 such b's, pick 3 and check if the 9 sums are distinct.
    #    Since they are primes, if they are distinct, they are pairwise coprime.
    
    # Optimization: Precompute for each b, which a's make a+b prime?
    # Actually, let's just iterate A and filter B.
    # Number of A triples is C(128, 3) ~ 340k. For each A, checking 128 B's is ~43M ops.
    # This might be tight in 85s in Python.
    # Let's restrict A to a smaller range first, or use a smarter filter.
    
    # Smarter filter:
    # For a fixed b, a must be such that a+b is prime.
    # Let's iterate B first? Same complexity.
    
    # Let's try a hybrid: Iterate a1, a2. Compute valid_b_12.
    # Then for each a3, check if any b in valid_b_12 makes a3+b prime.
    # If we find 3 such b's, we have a candidate.
    
    # Let's limit the search space to increase speed.
    # Primes are dense enough that small numbers should work.
    
    found_9 = False
    
    # Phase 1: Search for 9 distinct prime sums
    # We will iterate a1, a2, a3 but break early if possible.
    # To speed up, we can precompute `prime_diffs[b]` = set of a such that a+b is prime?
    # No, simpler:
    
    # Precompute valid_b_for_pair[a1][a2] -> set of b
    # This is C(128, 2) * 128 ~ 8k * 128 ~ 1M ops. Fast.
    valid_b_pairs = {}
    for a1 in range(1, 129):
        for a2 in range(a1 + 1, 129):
            valid_b = []
            for b in range(1, 129):
                if (a1 + b) in primes_set and (a2 + b) in primes_set:
                    valid_b.append(b)
            valid_b_pairs[(a1, a2)] = valid_b
            
    # Now iterate a1, a2, a3
    # We want to find A, B such that all 9 sums are prime.
    # This implies that for the chosen B = {b1, b2, b3}, 
    # a1+b_i, a2+b_i, a3+b_i are prime for all i.
    # So b_i must be in valid_b_pairs[(a1, a2)] AND valid_b_pairs[(a1, a3)] AND valid_b_pairs[(a2, a3)].
    
    # Let's iterate a1, a2, a3
    for a1 in range(1, 100): # Limit range for speed, expand if needed
        if time.time() > deadline - 5:
            break
        for a2 in range(a1 + 1, 100):
            if time.time() > deadline - 5:
                break
            vb_12 = valid_b_pairs.get((a1, a2), [])
            if len(vb_12) < 3:
                continue
                
            for a3 in range(a2 + 1, 100):
                if time.time() > deadline - 5:
                    break
                
                # b must be in vb_12, and a1+b prime, a3+b prime, a2+b prime (already in vb_12)
                # So b must be in vb_12 AND (a1+b) in primes AND (a3+b) in primes
                # Actually vb_12 ensures a1+b and a2+b are prime.
                # We just need to check a3+b.
                
                valid_b_123 = []
                for b in vb_12:
                    if (a3 + b) in primes_set:
                        valid_b_123.append(b)
                
                if len(valid_b_123) < 3:
                    continue
                
                # We have at least 3 b's. Let's pick 3.
                # We want the 9 sums to be distinct.
                # Sums are a_i + b_j.
                # If we pick b1, b2, b3 from valid_b_123, all 9 sums are prime.
                # We just need to ensure they are distinct.
                
                # Try the first 3 valid b's? Or search for a triple that yields distinct sums?
                # Usually, distinct b's and distinct a's with prime sums will yield distinct sums
                # unless there's a specific collision like a1+b1 = a2+b2.
                # Collision: a1 - a2 = b2 - b1.
                
                # Let's just take the first 3 and check. If collision, try next combo?
                # For now, let's just check if ANY 3 b's yield 9 distinct sums.
                # Since len(valid_b_123) might be small, we can check all combos if it's small.
                
                if len(valid_b_123) > 20:
                    # Too many, just sample or take first 3
                    b_candidates = valid_b_123[:3]
                else:
                    b_candidates = valid_b_123
                    
                # Check combinations of b_candidates
                found = False
                for b1, b2, b3 in combinations(b_candidates, 3):
                    B = [b1, b2, b3]
                    A = [a1, a2, a3]
                    sums = [A[i] + B[j] for i in range(3) for j in range(3)]
                    if len(set(sums)) == 9:
                        # All distinct primes -> pairwise coprime
                        best_A = A
                        best_B = B
                        best_count = 9
                        found_9 = True
                        found = True
                        break
                if found:
                    break
            if found_9:
                break
        if found_9:
            break
            
    if found_9:
        return [best_A, best_B]

    # Phase 2: If no 9-prime solution found, search for general pairwise coprime sums
    # This is harder. We can use random search with a time limit.
    import random
    rng = random.Random(42)
    
    # Precompute "good" numbers: primes and prime powers?
    # Just use random search for now, checking pairwise coprime.
    
    attempts = 0
    while time.time() < deadline and attempts < 100000:
        A = sorted(rng.sample(range(1, 129), 3))
        B = sorted(rng.sample(range(1, 129), 3))
        
        sums = [A[i] + B[j] for i in range(3) for j in range(3)]
        distinct_sums = set(sums)
        count = len(distinct_sums)
        
        if count > best_count:
            if are_pairwise_coprime(distinct_sums):
                best_count = count
                best_A = A[:]
                best_B = B[:]
                if best_count == 9:
                    return [best_A, best_B]
        attempts += 1

    return [best_A, best_B]
# EVOLVE-BLOCK-END
