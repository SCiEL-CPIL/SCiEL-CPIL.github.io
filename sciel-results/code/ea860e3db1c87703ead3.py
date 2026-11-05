import itertools
import time
import math

# Deadline guard
_START_TIME = time.time()
_DEADLINE = _START_TIME + 85.0

# Precompute primes up to 256 (max sum 128+128)
_MAX = 256
_is_prime = [False] * (_MAX + 1)
_is_prime[2] = True
for i in range(3, _MAX + 1, 2):
    is_p = True
    for j in range(3, int(i**0.5) + 1, 2):
        if i % j == 0:
            is_p = False
            break
    if is_p:
        _is_prime[i] = True

# List of odd and even numbers in [1, 128]
_odds = [x for x in range(1, 129) if x % 2 == 1]
_evens = [x for x in range(2, 129) if x % 2 == 0]

# Precompute for each odd a, the set of even b such that a+b is prime
# valid_b_for_a[a] = set of even b in [2, 128] such that a+b is prime
_valid_b_for_a = {}
for a in _odds:
    valid_bs = set()
    for b in _evens:
        s = a + b
        if s <= _MAX and _is_prime[s]:
            valid_bs.add(b)
    _valid_b_for_a[a] = valid_bs

def construct():
    best_score = 0
    best_A = [1, 3, 5]
    best_B = [2, 4, 6]
    
    # Fallback check for the initial best (though unlikely to be optimal)
    # We will update best_score and best_A/B as we find better solutions.
    
    # Strategy:
    # 1. Iterate over all combinations of 3 distinct odds A.
    # 2. For each A, compute the intersection of valid_b_for_a for all a in A.
    # 3. If the intersection size is >= 3, iterate over combinations of 3 distinct evens B from the intersection.
    # 4. Check if all 9 sums a+b are distinct.
    # 5. If distinct, since they are all primes (by construction of valid_b), they are pairwise coprime.
    # 6. Update best if score > best_score.
    
    # To speed up, we can iterate over A in a specific order or limit the search if we find a perfect score (9).
    # Max possible score is 9.
    
    # Optimization: Sort odds by the size of their valid_b set in ascending order? 
    # Actually, we want A such that the intersection is large.
    # Smaller odds might have more valid b's? Not necessarily.
    # Let's just iterate. There are C(64, 3) = 41664 combinations of A.
    # For each A, intersection calculation is fast (set intersection).
    # If intersection size >= 3, we check C(k, 3) combinations of B.
    
    # We can break early if we find a solution with score 9.
    
    for A in itertools.combinations(_odds, 3):
        if time.time() > _DEADLINE:
            break
            
        # Compute intersection of valid b's
        common_bs = _valid_b_for_a[A[0]] & _valid_b_for_a[A[1]] & _valid_b_for_a[A[2]]
        
        if len(common_bs) < 3:
            continue
            
        # Sort common_bs for consistent iteration (optional, but good for debugging)
        common_bs_list = sorted(common_bs)
        
        # Iterate over combinations of 3 distinct evens from common_bs
        # We want to find any B such that all 9 sums are distinct.
        # Since sums are primes, distinctness is the only requirement for pairwise coprimality.
        
        found_9 = False
        for B in itertools.combinations(common_bs_list, 3):
            sums = set()
            distinct = True
            for a in A:
                for b in B:
                    s = a + b
                    if s in sums:
                        distinct = False
                        break
                    sums.add(s)
                if not distinct:
                    break
            
            if distinct:
                # All 9 sums are distinct primes, so they are pairwise coprime.
                best_score = 9
                best_A = list(A)
                best_B = list(B)
                found_9 = True
                break
        
        if found_9:
            break
        # If we don't find 9, we might still want to check for score < 9?
        # The task says "Maximize the cardinality". 9 is the maximum possible.
        # If we don't find 9, we should keep searching for the next best?
        # However, finding 9 is highly likely.
        # If we don't find 9 in the first few A's, we might need to check for score 8, 7, etc.
        # But let's assume 9 is findable. If not, we can add logic for lower scores later.
        # For now, we break if we find 9.
        
        # If we don't break, we continue searching for other A's that might yield 9.
        # If we exhaust all A's without finding 9, we return the best found so far.
        # But wait, the loop above only updates best if we find 9.
        # We need to track the best score found, which might be less than 9 if 9 is not found.
        
        # Let's refine:
        # Inside the B loop, if distinct is True, score is 9.
        # If we don't find 9 for this A, we can check for score 8?
        # Score 8 means one pair of sums is equal.
        # This is complex. Let's first see if 9 is found.
        
        # If we don't find 9, we can try to find the best possible score for this A.
        # But given the density of primes, 9 is very likely.
        
        # If we finish the loop without breaking (no 9 found), we might want to do a second pass for score < 9.
        # But for now, let's assume 9 is the target.
        
        # Actually, let's just track the best score found in the entire search.
        # If we find 9, we return immediately.
        
        # If we don't find 9, we need to evaluate scores < 9.
        # Let's add a fallback mechanism if 9 is not found.
        
    # If we didn't find a score of 9, we should try to find the best score less than 9.
    # However, the loop above breaks if 9 is found.
    # If 9 is not found, we need to re-evaluate or do a more thorough search for lower scores.
    # But given the constraints, 9 is almost certainly achievable.
    
    # Let's add a check: if best_score < 9, do a randomized search or a more exhaustive check for lower scores.
    if best_score < 9:
        # This block is likely not executed if 9 is found.
        # For safety, we can do a quick randomized search to see if we can find a better score.
        import random
        rng = random.Random(42)
        for _ in range(10000):
            if time.time() > _DEADLINE:
                break
            A = rng.sample(_odds, 3)
            B = rng.sample(_evens, 3)
            sums = [a + b for a in A for b in B]
            distinct_sums = set(sums)
            if len(distinct_sums) <= best_score:
                continue
            # Check pairwise coprime
            ds = list(distinct_sums)
            valid = True
            for i in range(len(ds)):
                for j in range(i+1, len(ds)):
                    if math.gcd(ds[i], ds[j]) != 1:
                        valid = False
                        break
                if not valid:
                    break
            if valid:
                best_score = len(distinct_sums)
                best_A = A
                best_B = B
                if best_score == 9:
                    break
    
    return [best_A, best_B]

# Run construct to verify (optional, but good for local testing)
if __name__ == "__main__":
    res = construct()
    print(res)
