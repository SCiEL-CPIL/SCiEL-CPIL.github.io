"""Structural change: ILP-based search for pairwise-coprime sumsets."""
import random
import math
import time

_LIMIT = 128
_DEADLINE = 85.0  # safety margin below the 90s wall-clock kill


def _valid(A, B):
    """Return (distinct_sum_count, distinct_sum_set) for a candidate, or (0, None) if invalid."""
    A = sorted(A)
    B = sorted(B)
    if len(set(A)) != 3 or len(set(B)) != 3:
        return 0, None
    for x in A:
        if not (1 <= x <= _LIMIT):
            return 0, None
    for y in B:
        if not (1 <= y <= _LIMIT):
            return 0, None
    sums = {a + b for a in A for b in B}
    lst = sorted(sums)
    for i in range(len(lst)):
        for j in range(i + 1, len(lst)):
            if math.gcd(lst[i], lst[j]) != 1:
                return 0, None
    return len(lst), sums


def _ilp_solve(start_time, deadline):
    """
    Use ILP to find optimal A, B. Variables:
    - x[i] for i in 1..128: 1 if i in A
    - y[j] for j in 1..128: 1 if j in B  
    - z[s] for s in 2..256: 1 if s is in sumset
    Constraints:
    - sum(x) = 3, sum(y) = 3
    - z[s] >= x[i] + y[j] - 1 for all i,j with i+j=s
    - z[s] <= sum over (i,j) with i+j=s of (x[i] + y[j] - 1) ... actually z[s] = OR
    - For incompatible pairs (s1,s2) with gcd(s1,s2)>1: z[s1] + z[s2] <= 1
    Maximize sum(z)
    """
    try:
        from scipy.optimize import linprog
    except ImportError:
        return None, None
    
    # Actually, linprog is for LP, not ILP. We need PuLP or similar.
    # Let's use a different approach: branch and bound with LP relaxation.
    # Or just use a smarter combinatorial search.
    
    # Alternative structural approach: enumerate sumsets of size k (starting from 9 down)
    # and check decomposability.
    
    # For a sumset S = A+B with |A|=|B|=3, S must satisfy:
    # - |S| <= 9
    # - S can be "factored" into two 3-element sets
    # 
    # A necessary condition: for each s in S, s = a+b for some a in A, b in B
    # This is equivalent to: S is the Minkowski sum of two 3-element sets.
    #
    # Characterization: If S = A+B, then for any a1,a2 in A, the set {s - a1 : s in S} 
    # and {s - a2 : s in S} differ by a1-a2, and both must be subsets of B... 
    # This is complex. Let's use a direct approach.
    
    # Better structural approach: Fix A, then the sumset is determined by B.
    # For a fixed A, find B that maximizes |A+B| with pairwise coprime constraint.
    # 
    # For fixed A = {a1, a2, a3}, and B = {b1, b2, b3}, the sums are:
    # a1+b1, a1+b2, a1+b3, a2+b1, a2+b2, a2+b3, a3+b1, a3+b2, a3+b3
    # 
    # We can iterate over A (combinations of 3 from 1..128 = ~350k) and for each,
    # search over B more efficiently using the coprime constraint.
    
    # Even better: use the fact that if we fix A, then for each b in B,
    # the three sums a1+b, a2+b, a3+b must all be coprime with each other
    # AND with all other sums.
    
    # Let me try: enumerate all 3-subsets of [1,128] for A (sorted), and for each,
    # use a greedy/backtracking approach to find B.
    
    # But 128 choose 3 = 350,976. For each, we need to search B which is also 350,976.
    # That's 123 billion total. Too many.
    
    # Pruning: If A has two elements with gcd(a1-a2, something) > 1, that might help.
    
    # Actually, let's try a different structural approach:
    # 1. Precompute for each pair (a, b) the set of "forbidden" sums
    # 2. Use a maximum independent set approach on the compatibility graph
    
    # Compatibility graph: nodes are potential sums (2..256), edges connect compatible pairs (gcd=1)
    # We want a clique of size k that can be decomposed as A+B.
    
    # This is still complex. Let me try a practical approach:
    # Systematic search with strong pruning, but smarter than random.
    
    # Key insight: if all sums are odd, they can only share odd prime factors.
    # If we use small odd primes as "witnesses", we can constrain the search.
    
    # Let's try: fix the sumset structure first.
    # For |A+B| = 9, all 9 sums must be distinct and pairwise coprime.
    # The 9 sums form a 3x3 grid: s[i][j] = a[i] + b[j]
    # Property: s[i][j] + s[i'][j'] = s[i][j'] + s[i'][j] for all i,i',j,j'
    # This is a strong constraint!
    
    # So: s[0][0] + s[1][1] = s[0][1] + s[1][0]
    # And similar for other pairs.
    
    # This means the sumset has a very specific structure.
    
    # Let me enumerate: choose 4 sums that form a "corner" of the grid, then derive the rest.
    # s00, s01, s10, s11 with s00+s11 = s01+s10.
    # Then s02 = s00 + (s01-s00) + (s02-s01)... no, that's not right.
    
    # Actually: s[i][j] = a[i] + b[j]
    # s[0][0] = a0+b0
    # s[0][1] = a0+b1 => b1 = s[0][1] - a0
    # s[1][0] = a1+b0 => a1 = s[1][0] - b0
    # s[1][1] = a1+b1 = (s[1][0]-b0) + (s[0][1]-a0) = s[1][0] + s[0][1] - a0 - b0 = s[1][0] + s[0][1] - s[0][0]
    # So s[1][1] = s[1][0] + s[0][1] - s[0][0]
    
    # Similarly: s[2][0] = a2+b0, s[0][2] = a0+b2
    # s[2][2] = s[2][0] + s[0][2] - s[0][0]
    # s[1][2] = a1+b2 = (s[1][0]-b0) + (s[0][2]-a0) = s[1][0] + s[0][2] - s[0][0]
    # s[2][1] = a2+b1 = (s[2][0]-b0) + (s[0][1]-a0) = s[2][0] + s[0][1] - s[0][0]
    
    # So given s[0][0], s[0][1], s[0][2], s[1][0], s[2][0], we can compute all other sums:
    # s[1][1] = s[1][0] + s[0][1] - s[0][0]
    # s[1][2] = s[1][0] + s[0][2] - s[0][0]
    # s[2][1] = s[2][0] + s[0][1] - s[0][0]
    # s[2][2] = s[2][0] + s[0][2] - s[0][0]
    
    # Then: a0 = s[0][0] - b0, but we need to recover a and b.
    # a0 = s[0][0] - b0, a1 = s[1][0] - b0, a2 = s[2][0] - b0
    # b0 = s[0][0] - a0, b1 = s[0][1] - a0, b2 = s[0][2] - a0
    
    # So we need: a0, a1, a2, b0, b1, b2 all in [1, 128] and distinct within each set.
    
    # This gives us a 5-parameter search: s[0][0], s[0][1], s[0][2], s[1][0], s[2][0]
    # with the constraint that all derived sums are in [2, 256] and all a,b in [1,128].
    
    # Number of combinations: 255^5 ≈ 10^11. Still too many.
    
    # But with strong pruning (pairwise coprime check at each step), it might work.
    
    # Let me implement this structural search.
    
    best_score = 0
    best_A = None
    best_B = None
    
    # Enumerate s00, s01, s02, s10, s20
    # s00 in [2, 256], s01 in [2, 256], etc.
    # Pruning: s01 != s00, s02 != s00, s02 != s01 (for distinct sums in row 0)
    # s10 != s00, s20 != s00, s20 != s10 (for distinct sums in col 0)
    # Derived sums must be in [2, 256]
    # All 9 sums must be pairwise coprime
    # a0 = s00 - b0, but b0 is free... 
    
    # Actually, we have freedom in choosing b0 (or equivalently a0).
    # Given the 5 sums, b0 can range such that:
    # a0 = s00 - b0 in [1,128] => b0 in [s00-128, s00-1]
    # a1 = s10 - b0 in [1,128] => b0 in [s10-128, s10-1]
    # a2 = s20 - b0 in [1,128] => b0 in [s20-128, s20-1]
    # b1 = s01 - a0 = s01 - s00 + b0 in [1,128] => b0 in [s00-s01+1, s00-s01+128]
    # b2 = s02 - a0 = s02 - s00 + b0 in [1,128] => b0 in [s00-s02+1, s00-s02+128]
    
    # So b0 must be in the intersection of these 5 intervals.
    
    # This is getting complex. Let me just implement a clean version.
    
    def search_structural(deadline):
        nonlocal best_score, best_A, best_B
        
        # Enumerate the 5 free parameters
        for s00 in range(2, 257):
            if time.time() > deadline:
                return
            for s01 in range(2, 257):
                if time.time() > deadline:
                    return
                if s01 == s00:
                    continue
                if math.gcd(s00, s01) != 1:
                    continue
                for s02 in range(2, 257):
                    if time.time() > deadline:
                        return
                    if s02 == s00 or s02 == s01:
                        continue
                    if math.gcd(s00, s02) != 1 or math.gcd(s01, s02) != 1:
                        continue
                    for s10 in range(2, 257):
                        if time.time() > deadline:
                            return
                        if s10 == s00:
                            continue
                        if math.gcd(s00, s10) != 1:
                            continue
                        # s11 = s10 + s01 - s00
                        s11 = s10 + s01 - s00
                        if s11 < 2 or s11 > 256:
                            continue
                        if s11 == s00 or s11 == s01 or s11 == s02 or s11 == s10:
                            continue
                        if (math.gcd(s11, s00) != 1 or math.gcd(s11, s01) != 1 or 
                            math.gcd(s11, s02) != 1 or math.gcd(s11, s10) != 1):
                            continue
                        for s20 in range(2, 257):
                            if time.time() > deadline:
                                return
                            if s20 == s00 or s20 == s10:
                                continue
                            if math.gcd(s20, s00) != 1 or math.gcd(s20, s10) != 1:
                                continue
                            # s21 = s20 + s01 - s00
                            s21 = s20 + s01 - s00
                            if s21 < 2 or s21 > 256:
                                continue
                            if s21 == s00 or s21 == s01 or s21 == s02 or s21 == s10 or s21 == s11:
                                continue
                            if (math.gcd(s21, s00) != 1 or math.gcd(s21, s01) != 1 or 
                                math.gcd(s21, s02) != 1 or math.gcd(s21, s10) != 1 or 
                                math.gcd(s21, s11) != 1):
                                continue
                            # s22 = s20 + s02 - s00
                            s22 = s20 + s02 - s00
                            if s22 < 2 or s22 > 256:
                                continue
                            if s22 == s00 or s22 == s01 or s22 == s02 or s22 == s10 or s22 == s11 or s22 == s21:
                                continue
                            if (math.gcd(s22, s00) != 1 or math.gcd(s22, s01) != 1 or 
                                math.gcd(s22, s02) != 1 or math.gcd(s22, s10) != 1 or 
                                math.gcd(s22, s11) != 1 or math.gcd(s22, s21) != 1):
                                continue
                            # s12 = s10 + s02 - s00
                            s12 = s10 + s02 - s00
                            if s12 < 2 or s12 > 256:
                                continue
                            if s12 == s00 or s12 == s01 or s12 == s02 or s12 == s10 or s12 == s11 or s12 == s21 or s12 == s22:
                                continue
                            if (math.gcd(s12, s00) != 1 or math.gcd(s12, s01) != 1 or 
                                math.gcd(s12, s02) != 1 or math.gcd(s12, s10) != 1 or 
                                math.gcd(s12, s11) != 1 or math.gcd(s12, s21) != 1 or 
                                math.gcd(s12, s22) != 1):
                                continue
                            
                            # All 9 sums are distinct and pairwise coprime!
                            # Now check if we can find valid a,b in [1,128]
                            # b0 in [max(s00-128, s10-128, s20-128, s00-s01+1, s00-s02+1), 
                            #          min(s00-1, s10-1, s20-1, s00-s01+128, s00-s02+128)]
                            lo = max(s00 - 128, s10 - 128, s20 - 128, s00 - s01 + 1, s00 - s02 + 1)
                            hi = min(s00 - 1, s10 - 1, s20 - 1, s00 - s01 + 128, s00 - s02 + 128)
                            
                            if lo > hi:
                                continue
                            
                            for b0 in range(lo, hi + 1):
                                if time.time() > deadline:
                                    return
                                a0 = s00 - b0
                                a1 = s10 - b0
                                a2 = s20 - b0
                                b1 = s01 - a0
                                b2 = s02 - a0
                                
                                # Check distinctness
                                A = [a0, a1, a2]
                                B = [b0, b1, b2]
                                if len(set(A)) != 3 or len(set(B)) != 3:
                                    continue
                                if any(x < 1 or x > 128 for x in A + B):
                                    continue
                                
                                # Verify
                                score, _ = _valid(A, B)
                                if score > best_score:
                                    best_score = score
                                    best_A = sorted(A)
                                    best_B = sorted(B)
                                    if best_score >= 9:
                                        return
    
    search_structural(deadline)
    return best_score, best_A, best_B


def construct():
    start = time.time()
    
    # Valid fallback: A={4, 6, 10}, B={1, 3, 7} -> Sums {5, 7, 9, 11, 13, 17}
    best_A = [4, 6, 10]
    best_B = [1, 3, 7]
    best_score, _ = _valid(best_A, best_B)
    
    # Structural search
    deadline = start + _DEADLINE
    score, A_found, B_found = _ilp_solve(start, deadline)
    
    if score > best_score and A_found is not None and B_found is not None:
        best_score = score
        best_A = A_found
        best_B = B_found
    
    # If still have time, do quick local search
    if time.time() < deadline - 5 and best_score < 9:
        rng = random.Random(20240517)
        evens = list(range(2, _LIMIT + 1, 2))
        odds = list(range(1, _LIMIT + 1, 2))
        
        def consider(A, B):
            nonlocal best_A, best_B, best_score
            s, _ = _valid(A, B)
            if s > best_score:
                best_score = s
                best_A = sorted(A)
                best_B = sorted(B)
        
        for _ in range(5000):
            if time.time() > deadline - 2:
                break
            A = rng.sample(evens, 3)
            B = rng.sample(odds, 3)
            consider(A, B)
            if best_score >= 9:
                break
        
        improved = True
        while improved and time.time() < deadline - 1:
            improved = False
            A = list(best_A)
            B = list(best_B)
            
            for pos in range(3):
                if time.time() > deadline - 1:
                    break
                for p in evens:
                    if p in A:
                        continue
                    old = A[pos]
                    A[pos] = p
                    s, _ = _valid(A, B)
                    if s > best_score:
                        best_score = s
                        best_A = sorted(A)
                        improved = True
                    else:
                        A[pos] = old
            
            for pos in range(3):
                if time.time() > deadline - 1:
                    break
                for p in odds:
                    if p in B:
                        continue
                    old = B[pos]
                    B[pos] = p
                    s, _ = _valid(A, B)
                    if s > best_score:
                        best_score = s
                        best_B = sorted(B)
                        improved = True
                    else:
                        B[pos] = old
    
    return [best_A, best_B]
