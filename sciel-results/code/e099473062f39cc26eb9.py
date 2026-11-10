import time
from math import gcd
from itertools import combinations

def construct():
    deadline = time.time() + 85  # safety margin under 90s wall-clock
    
    def distinct_sums(A, B):
        return {a + b for a in A for b in B}
    
    def is_pairwise_coprime(vals):
        lst = list(vals)
        n = len(lst)
        for i in range(n):
            for j in range(i + 1, n):
                if gcd(lst[i], lst[j]) != 1:
                    return False
        return True
    
    def score_of(A, B):
        s = distinct_sums(A, B)
        if len(s) < 2:
            return -1
        if not is_pairwise_coprime(s):
            return -1
        return len(s)
    
    # Start with known valid solution
    best_A = [1, 7, 13]
    best_B = [4, 10, 16]
    best_score = score_of(best_A, best_B)
    
    # Precompute small primes and 1 for potential sum targets
    def is_prime(n):
        if n < 2:
            return False
        if n == 2:
            return True
        if n % 2 == 0:
            return False
        for i in range(3, int(n**0.5) + 1, 2):
            if n % i == 0:
                return False
        return True
    
    # Generate all primes in [2, 256] (max possible sum is 128+128=256)
    primes_in_range = [p for p in range(2, 257) if is_prime(p)]
    
    # Strategy: Use backtracking to find A, B such that all 9 sums are distinct primes
    # This guarantees pairwise coprimality and maximum cardinality 9
    
    # Key insight: For A = {a1, a2, a3}, B = {b1, b2, b3} with a1<a2<a3, b1<b2<b3
    # The 9 sums form a structured pattern:
    # a1+b1 < a1+b2 < a1+b3
    # a2+b1 < a2+b2 < a2+b3
    # a3+b1 < a3+b2 < a3+b3
    # with a1+b1 < a2+b1 < a3+b1 and a1+b1 < a1+b2 < a1+b3
    
    # Try all combinations of 9 primes as target sums, then solve for A, B
    # But this is too many combinations. Instead, use a smarter approach.
    
    # Approach: Enumerate possible (a1, b1, a3, b3) tuples, then try to find a2, b2
    # such that all 9 sums are distinct primes
    
    # Precompute a set for fast prime lookup
    prime_set = set(primes_in_range)
    
    # Try different strategies for finding A and B
    
    # Strategy 1: Fix the differences
    # Let da = a2 - a1, db = b2 - b1
    # Then:
    # a1+b1 = p1 (prime)
    # a1+b2 = a1 + b1 + db = p1 + db (prime)
    # a2+b1 = a1 + da + b1 = p1 + da (prime)
    # a2+b2 = p1 + da + db (prime)
    # a3+b1 = a1 + 2*da + b1 = p1 + 2*da (prime)
    # a3+b2 = p1 + 2*da + db (prime)
    # a1+b3 = a1 + b1 + 2*db = p1 + 2*db (prime)
    # a2+b3 = p1 + da + 2*db (prime)
    # a3+b3 = p1 + 2*da + 2*db (prime)
    
    # So we need: p1, p1+da, p1+db, p1+da+db, p1+2*da, p1+2*da+db, p1+2*db, p1+da+2*db, p1+2*da+2*db
    # all to be distinct primes
    
    # Iterate over possible p1, da, db
    found_solution = False
    
    for p1 in primes_in_range:
        if time.time() > deadline - 5:
            break
        # p1 is a1+b1, so a1 >= 1, b1 >= 1, thus p1 >= 2
        # Also a1 <= 128, b1 <= 128, so p1 <= 256
        
        # da = a2 - a1, db = b2 - b1
        # a2 = a1 + da, a3 = a1 + 2*da
        # b2 = b1 + db, b3 = b1 + 2*db
        # All must be in [1, 128]
        # So 1 <= a1 <= 128, 1 <= a1+da <= 128, 1 <= a1+2*da <= 128
        # Similarly for b
        
        # da and db must be positive (since a1<a2<a3, b1<b2<b3)
        # Max da: a1+2*da <= 128, a1 >= 1, so 2*da <= 127, da <= 63
        # Max db: similarly db <= 63
        
        for da in range(1, 64):
            if time.time() > deadline - 3:
                break
            for db in range(1, 64):
                if time.time() > deadline - 2:
                    break
                
                # Check all 9 sums are primes
                sums = [
                    p1,
                    p1 + da,
                    p1 + db,
                    p1 + da + db,
                    p1 + 2*da,
                    p1 + 2*da + db,
                    p1 + 2*db,
                    p1 + da + 2*db,
                    p1 + 2*da + 2*db
                ]
                
                # All must be distinct primes in [2, 256]
                if len(set(sums)) != 9:
                    continue
                if not all(s in prime_set for s in sums):
                    continue
                
                # Now find a1, b1 such that:
                # a1 + b1 = p1
                # 1 <= a1 <= 128
                # 1 <= a1 + da <= 128 => 1-da <= a1 <= 128-da
                # 1 <= a1 + 2*da <= 128 => 1-2*da <= a1 <= 128-2*da
                # Similarly for b1: b1 = p1 - a1
                # 1 <= b1 <= 128 => 1 <= p1-a1 <= 128 => p1-128 <= a1 <= p1-1
                # 1 <= b1+db <= 128 => 1-db <= p1-a1 <= 128-db => p1+db-128 <= a1 <= p1+db-1
                # 1 <= b1+2*db <= 128 => p1+2*db-128 <= a1 <= p1+2*db-1
                
                a1_min = max(1, 1 - da, 1 - 2*da, p1 - 128, p1 + db - 128, p1 + 2*db - 128)
                a1_max = min(128, 128 - da, 128 - 2*da, p1 - 1, p1 + db - 1, p1 + 2*db - 1)
                
                if a1_min > a1_max:
                    continue
                
                # Try a1 values
                for a1 in range(a1_min, a1_max + 1):
                    b1 = p1 - a1
                    A = [a1, a1 + da, a1 + 2*da]
                    B = [b1, b1 + db, b1 + 2*db]
                    
                    # Verify all in [1, 128] and distinct
                    if all(1 <= v <= 128 for v in A) and all(1 <= v <= 128 for v in B):
                        if len(set(A)) == 3 and len(set(B)) == 3:
                            sc = score_of(A, B)
                            if sc == 9:
                                return A, B
                            elif sc > best_score:
                                best_score = sc
                                best_A = A
                                best_B = B
                            found_solution = True
                            break
                if found_solution:
                    break
            if found_solution:
                break
        if found_solution:
            break
    
    # If we found a solution with score 9, we're done
    if best_score == 9:
        return best_A, best_B
    
    # Fallback: broader search if target-based didn't find 9
    import random
    random.seed(42)
    
    # Smarter random: bias towards structured sets
    while time.time() < deadline - 2:
        # Try arithmetic progressions or near-arithmetic
        step_a = random.randint(5, 30)
        step_b = random.randint(3, 20)
        a1 = random.randint(1, 128 - 2*step_a)
        b1 = random.randint(1, 128 - 2*step_b)
        
        A = [a1, a1 + step_a, a1 + 2*step_a]
        B = [b1, b1 + step_b, b1 + 2*step_b]
        
        if all(1 <= v <= 128 for v in A) and all(1 <= v <= 128 for v in B):
            sc = score_of(A, B)
            if sc > best_score:
                best_score = sc
                best_A = A
                best_B = B
                if best_score == 9:
                    break
        
        # Also pure random
        A = sorted(random.sample(range(1, 129), 3))
        B = sorted(random.sample(range(1, 129), 3))
        sc = score_of(A, B)
        if sc > best_score:
            best_score = sc
            best_A = A
            best_B = B
            if best_score == 9:
                break
    
    return best_A, best_B
