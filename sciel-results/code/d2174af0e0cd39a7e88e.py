import time
from math import gcd
from itertools import combinations, permutations

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
    
    # Precompute small primes
    small_primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97]
    
    # Generate candidate target sum sets (9 pairwise coprime values)
    # Focus on sets where 1 is included (makes pairwise coprime easier)
    # and use small primes to keep sums in range
    target_sums_list = []
    
    # Option 1: {1} + 8 small primes (most promising)
    for combo in combinations(small_primes[:15], 8):
        target = [1] + list(combo)
        target.sort()
        target_sums_list.append(target)
    
    # Option 2: 9 small primes without 1
    for combo in combinations(small_primes[:12], 9):
        target = list(combo)
        target.sort()
        target_sums_list.append(target)
    
    # Sort targets by "easiness" - prefer smaller max sum (easier to fit in [2,256])
    target_sums_list.sort(key=lambda t: t[-1])
    
    # For each target sum set, try to find A and B using permutation approach
    # The 9 sums form a 3x3 grid: S[i][j] = A[i] + B[j]
    # If we assign target values to grid positions, we can solve for A and B
    # A[i] = S[i][j] - B[j] for any j, must be consistent
    # B[j] = S[i][j] - A[i] for any i, must be consistent
    
    # Key insight: if we fix A[0]=0 (by shifting), then B[j] = S[0][j]
    # and A[i] = S[i][0] - B[0] = S[i][0] - S[0][0]
    # Then check S[i][j] == A[i] + B[j] for all i,j
    
    # Actually, let's use: A[i] = S[i][0] - B[0], B[j] = S[0][j] - A[0]
    # This gives: S[i][j] should = (S[i][0] - B[0]) + (S[0][j] - A[0])
    # = S[i][0] + S[0][j] - A[0] - B[0]
    # So the condition is: S[i][j] + A[0] + B[0] = S[i][0] + S[0][j]
    # Let c = A[0] + B[0] = S[0][0]
    # Then: S[i][j] = S[i][0] + S[0][j] - S[0][0]
    
    # This means the grid must be "rank-1 additive": S[i][j] = row_i + col_j
    # where row_i = S[i][0] - S[0][0] + A[0] and col_j = S[0][j] - A[0]
    # Actually simpler: S[i][j] - S[i][0] - S[0][j] + S[0][0] = 0 for all i,j
    
    for target in target_sums_list:
        if time.time() > deadline - 5:
            break
        if len(target) != 9:
            continue
        
        # Try all 9! permutations of target values into the 3x3 grid
        # But that's 362880 per target... too many.
        # Instead, use the additive structure constraint.
        # S[i][j] = S[i][0] + S[0][j] - S[0][0]
        # So once we pick S[0][0], S[0][1], S[0][2], S[1][0], S[2][0],
        # the rest are determined: S[i][j] = S[i][0] + S[0][j] - S[0][0]
        
        # Pick 5 values: S[0][0], S[0][1], S[0][2], S[1][0], S[2][0]
        # These must be 5 distinct values from target
        # Then S[1][1] = S[1][0] + S[0][1] - S[0][0]
        # S[1][2] = S[1][0] + S[0][2] - S[0][0]
        # S[2][1] = S[2][0] + S[0][1] - S[0][0]
        # S[2][2] = S[2][0] + S[0][2] - S[0][0]
        # All 9 must be distinct and equal to target
        
        t_set = set(target)
        
        # Choose S[0][0] from target
        for s00 in target:
            if time.time() > deadline - 5:
                break
            # Choose S[0][1] from target (different from s00)
            for s01 in target:
                if s01 == s00:
                    continue
                # Choose S[0][2] from target (different from s00, s01)
                for s02 in target:
                    if s02 == s00 or s02 == s01:
                        continue
                    # Choose S[1][0] from target (different from s00, s01, s02)
                    for s10 in target:
                        if s10 == s00 or s10 == s01 or s10 == s02:
                            continue
                        # Choose S[2][0] from target (different from s00, s01, s02, s10)
                        for s20 in target:
                            if s20 in (s00, s01, s02, s10):
                                continue
                            
                            # Compute the remaining 4 values
                            s11 = s10 + s01 - s00
                            s12 = s10 + s02 - s00
                            s21 = s20 + s01 - s00
                            s22 = s20 + s02 - s00
                            
                            # Check all 9 are in target and distinct
                            all_nine = [s00, s01, s02, s10, s20, s11, s12, s21, s22]
                            if len(set(all_nine)) != 9:
                                continue
                            if not all(v in t_set for v in all_nine):
                                continue
                            
                            # Now solve for A and B
                            # A[0] + B[0] = s00
                            # A[0] + B[1] = s01
                            # A[0] + B[2] = s02
                            # A[1] + B[0] = s10
                            # A[2] + B[0] = s20
                            
                            # Let A[0] = a, then B[0] = s00 - a
                            # B[1] = s01 - a, B[2] = s02 - a
                            # A[1] = s10 - B[0] = s10 - s00 + a
                            # A[2] = s20 - B[0] = s20 - s00 + a
                            
                            # Need all in [1, 128] and distinct
                            # A = [a, s10-s00+a, s20-s00+a]
                            # B = [s00-a, s01-a, s02-a]
                            
                            # a must satisfy:
                            # 1 <= a <= 128
                            # 1 <= s10-s00+a <= 128 => 1-(s10-s00) <= a <= 128-(s10-s00)
                            # 1 <= s20-s00+a <= 128 => 1-(s20-s00) <= a <= 128-(s20-s00)
                            # 1 <= s00-a <= 128 => s00-128 <= a <= s00-1
                            # 1 <= s01-a <= 128 => s01-128 <= a <= s01-1
                            # 1 <= s02-a <= 128 => s02-128 <= a <= s02-1
                            
                            a_min = max(1, 1-(s10-s00), 1-(s20-s00), s00-128, s01-128, s02-128)
                            a_max = min(128, 128-(s10-s00), 128-(s20-s00), s00-1, s01-1, s02-1)
                            
                            if a_min > a_max:
                                continue
                            
                            # Try values of a in range
                            # Also need A elements distinct and B elements distinct
                            for a in range(a_min, a_max + 1):
                                A = [a, s10 - s00 + a, s20 - s00 + a]
                                B = [s00 - a, s01 - a, s02 - a]
                                
                                if len(set(A)) != 3 or len(set(B)) != 3:
                                    continue
                                if not all(1 <= v <= 128 for v in A):
                                    continue
                                if not all(1 <= v <= 128 for v in B):
                                    continue
                                
                                A_sorted = sorted(A)
                                B_sorted = sorted(B)
                                sc = score_of(A_sorted, B_sorted)
                                if sc >= 9:
                                    return A_sorted, B_sorted
                                if sc > best_score:
                                    best_score = sc
                                    best_A = A_sorted
                                    best_B = B_sorted
    
    # Fallback: broader search if target-based didn't find 9
    if best_score < 9:
        import random
        random.seed(42)
        
        while time.time() < deadline - 2:
            # Try arithmetic progressions
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
            
            # Pure random
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
