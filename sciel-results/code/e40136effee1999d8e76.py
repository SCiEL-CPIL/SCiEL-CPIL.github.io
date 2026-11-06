"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from math import gcd
from itertools import combinations

def _pairwise_coprime(vals):
    """Check if all distinct values in vals are pairwise coprime."""
    distinct = list(set(vals))
    for i in range(len(distinct)):
        for j in range(i+1, len(distinct)):
            if gcd(distinct[i], distinct[j]) != 1:
                return False
    return True

def _evaluate(A, B):
    """Return (cardinality, valid) for sumset A+B."""
    sums = set()
    for a in A:
        for b in B:
            sums.add(a + b)
    if not _pairwise_coprime(list(sums)):
        return (0, False)
    return (len(sums), True)

def _find_best():
    """Systematic search for A, B maximizing |A+B| with pairwise coprime sums."""
    deadline = time.time() + 75  # 75s safety margin
    
    # Strategy: enumerate candidate "sum pools" - sets of pairwise coprime numbers
    # that can each be written as a+b with a,b in [1,128].
    # Then find A (3 distinct in [1,128]) and B (3 distinct in [1,128]) 
    # such that A+B ⊆ pool and |A+B| is maximized.
    
    # For 3x3 = 9 sums, we want 9 distinct pairwise-coprime values.
    # Each sum s = a+b where a,b ∈ [1,128], so s ∈ [2, 256].
    
    # Key insight: if we pick A and B carefully, we can get up to 9 distinct sums.
    # Let's search over structured families.
    
    best_score = 0
    best_result = ([1, 7, 13], [4, 10, 16])
    
    # Precompute: for each pair (a,b) with a,b in [1,128], the sum a+b
    # We need A+B sums to be pairwise coprime.
    
    # Approach 1: Try A = {x, x+d1, x+d2}, B = {y, y+d1', y+d2'}
    # The 9 sums form a structured set.
    
    # Approach 2: Direct search over small structured families
    # Use primes as targets: pick 9 pairwise coprime numbers in [2,256]
    # that can be partitioned into a 3x3 grid of sums.
    
    # Let's try: A = {a1, a2, a3}, B = {b1, b2, b3}
    # Sums: a_i + b_j for i,j in {1,2,3}
    # We need all 9 distinct and pairwise coprime.
    
    # Efficient search: fix A, then find B such that all 9 sums are pairwise coprime.
    # For a given A, the sums with B are: {a_i + b_j : i,j}
    # This is equivalent to: for each b_j, the set {a_i + b_j} must be coprime 
    # to all other sums.
    
    # Let's do a more targeted search:
    # Try A values from a curated list, and for each, search B.
    
    # Candidate A values: focus on small numbers to keep sums small and prime-friendly
    # Sums range from min(A)+min(B) to max(A)+max(B)
    
    # For 9 pairwise coprime sums, we ideally want them to be primes or prime powers.
    # In [2, 256], there are 54 primes.
    
    # Strategy: enumerate A from a reasonable set, then for each A, 
    # enumerate B and check.
    
    # To make this tractable, use the following observation:
    # If A = {a, a+d, a+2d} (arithmetic progression) and B = {b, b+e, b+2e},
    # then sums form a 3x3 grid with structure.
    
    # But more generally, let's just do a smart enumeration.
    
    # Precompute all valid (A, B) pairs is 128^6 which is too many.
    # Instead: fix A (3 distinct from [1,128]), then for B, we need 3 distinct values
    # from [1,128] such that all 9 sums are pairwise coprime.
    
    # Number of A choices: C(128,3) ≈ 340,000. For each, checking all B is C(128,3) ≈ 340,000.
    # Total: ~1.16 × 10^11 — too slow.
    
    # Better: for a given A, the constraint on B is that for each pair of sums 
    # (a_i + b_j, a_k + b_l), gcd must be 1.
    # This is a strong constraint. Let's use it to prune.
    
    # Alternative: generate candidate sum sets first.
    # Pick 9 pairwise coprime numbers in [2,256]. Then find A, B such that A+B = that set.
    
    # A+B = {a_i + b_j} is a "Minkowski sum" structure. For 3x3, the 9 sums have 
    # additive structure: s_ij = a_i + b_j, so s_ij + s_kl = s_il + s_kj.
    
    # This means: for any 2x2 submatrix, s_ij + s_kl = s_il + s_kj.
    # So we need 9 pairwise coprime numbers in [2,256] that form a 3x3 additive grid.
    
    # Let's enumerate 3x3 additive grids of pairwise coprime numbers.
    # A 3x3 additive grid is determined by a1, b1, d_a1, d_a2, d_b1, d_b2 where:
    #   row i: a1 + (i-1)*? No, A doesn't need to be AP.
    # Actually: A = {a1, a2, a3}, B = {b1, b2, b3}
    # Sums: s_ij = a_i + b_j
    # The grid constraint: s_ij + s_kl = s_il + s_kj for all i,k,j,l.
    
    # So we can parameterize: pick a1, a2, a3 (distinct, in [1,128]) and b1 (in [1,128]).
    # Then s_11 = a1+b1, s_12 = a1+b2, s_13 = a1+b3
    # s_21 = a2+b1, s_22 = a2+b2, s_23 = a2+b3
    # s_31 = a3+b1, s_32 = a3+b2, s_33 = a3+b3
    
    # Given a1,a2,a3,b1: 
    #   b2 = s_12 - a1, b3 = s_13 - a1
    #   s_22 = a2 + b2 = a2 + s_12 - a1
    #   s_23 = a2 + b3 = a2 + s_13 - a1
    #   s_32 = a3 + b2 = a3 + s_12 - a1
    #   s_33 = a3 + b3 = a3 + s_13 - a1
    
    # So the 9 sums are determined by (a1, a2, a3, b1, s_12, s_13) where:
    #   s_11 = a1 + b1
    #   s_12, s_13 are free (but b2 = s_12-a1, b3 = s_13-a1 must be in [1,128] and distinct from b1)
    #   s_21 = a2 + b1
    #   s_31 = a3 + b1
    #   s_22 = a2 + s_12 - a1
    #   s_23 = a2 + s_13 - a1
    #   s_32 = a3 + s_12 - a1
    #   s_33 = a3 + s_13 - a1
    
    # Constraints:
    # 1. a1, a2, a3 distinct in [1,128]
    # 2. b1 in [1,128], b2 = s_12-a1 in [1,128], b3 = s_13-a1 in [1,128]
    # 3. b1, b2, b3 distinct
    # 4. All 9 sums in [2, 256] (automatically if a_i, b_j in [1,128])
    # 5. All 9 sums distinct
    # 6. All 9 sums pairwise coprime
    
    # Free variables: a1, a2, a3, b1, s_12, s_13
    # Ranges: a_i ∈ [1,128], b1 ∈ [1,128], s_12 ∈ [a1+1, a1+128], s_13 ∈ [a1+1, a1+128]
    
    # This is still large. Let's be smarter.
    
    # For 9 pairwise coprime sums, they should mostly be primes.
    # Let's try to find configurations where the 9 sums are all primes.
    
    # Key pruning: if any sum is even and > 2, it's composite (unless it IS 2).
    # So at most one sum can be even, and it must be 2.
    # Since sums are a_i + b_j ≥ 2, and we need 9 distinct sums, 
    # at most one can be 2, and the other 8 must be odd primes.
    # For 8 odd primes, all sums must be odd, meaning a_i + b_j is odd for 8 of 9 pairs.
    # a_i + b_j is odd iff a_i and b_j have different parity.
    # If all 9 sums are odd: impossible (9 is odd, can't have all different parity pairs... 
    #   actually: if A has k evens and (3-k) odds, B has m evens and (3-m) odds,
    #   then number of odd sums = k*(3-m) + (3-k)*m)
    #   For all 9 odd: k*(3-m) + (3-k)*m = 9
    #   k*3 - km + 3m - km = 9 → 3k + 3m - 2km = 9
    #   If k=0: 3m=9→m=3. If k=3: 9+3m-6m=9→-3m=0→m=0.
    #   So either A all odd & B all even, or A all even & B all odd.
    
    # So: either A ⊆ odd, B ⊆ even, or A ⊆ even, B ⊆ odd.
    # This dramatically reduces the search space!
    
    # Case 1: A all odd, B all even. Sums all odd. Need 9 distinct odd pairwise-coprime sums.
    # Case 2: A all even, B all odd. Same thing.
    
    # Let's search Case 1: A = {a1,a2,a3} all odd in [1,128], B = {b1,b2,b3} all even in [2,128].
    
    # For each such A, B: check if all 9 sums are pairwise coprime.
    # Number of choices: C(64,3) * C(64,3) ≈ 41,664 * 41,664 ≈ 1.73 × 10^9. Still too many.
    
    # Need more pruning. Use the pairwise coprime constraint incrementally.
    
    # For fixed A, as we build B one element at a time, check constraints.
    
    # Let's do: fix A (3 odd numbers). For b1 (even): sums are a1+b1, a2+b1, a3+b1.
    # These 3 must be pairwise coprime.
    # For b2 (even, ≠ b1): sums a1+b2, a2+b2, a3+b2 must be pairwise coprime 
    # AND coprime to the first 3.
    # For b3: same with all 6 previous.
    
    # This is much more tractable with pruning.
    
    def check_three_coprime(x, y, z):
        return gcd(x,y) == 1 and gcd(x,z) == 1 and gcd(y,z) == 1
    
    def check_all_coprime(vals):
        for i in range(len(vals)):
            for j in range(i+1, len(vals)):
                if gcd(vals[i], vals[j]) != 1:
                    return False
        return True
    
    best = 0
    best_A = None
    best_B = None
    
    # Odd numbers in [1,128]
    odds = list(range(1, 129, 2))  # 64 values
    evens = list(range(2, 129, 2))  # 64 values
    
    # Enumerate A from odds
    # For efficiency, start with promising A values (smaller, more "prime-friendly")
    
    # Precompute for speed
    for idx_a in range(len(odds)):
        a1 = odds[idx_a]
        for idx_a2 in range(idx_a + 1, len(odds)):
            a2 = odds[idx_a2]
            for idx_a3 in range(idx_a2 + 1, len(odds)):
                a3 = odds[idx_a3]
                A = [a1, a2, a3]
                
                # For each pair (b1, b2) from evens, check if {a_i+b1, a_i+b2} are all coprime
                # Then try b3.
                
                # First, for each even b, compute the 3 sums and check they're pairwise coprime
                # Precompute valid b's for this A
                valid_bs = []
                for b in evens:
                    s1 = a1 + b
                    s2 = a2 + b
                    s3 = a3 + b
                    if s1 > 256 or s2 > 256 or s3 > 256:
                        continue
                    if check_three_coprime(s1, s2, s3):
                        valid_bs.append((b, s1, s2, s3))
                
                # Now find 3 distinct b's from valid_bs such that all 9 sums are pairwise coprime
                # For each pair (b1, b2), check cross-coprimality
                n_valid = len(valid_bs)
                for i1 in range(n_valid):
                    b1, s11, s12, s13 = valid_bs[i1]
                    for i2 in range(i1 + 1, n_valid):
                        b2, s21, s22, s23 = valid_bs[i2]
                        # Check cross-coprimality between {s11,s12,s13} and {s21,s22,s23}
                        ok = True
                        for x in (s11, s12, s13):
                            for y in (s21, s22, s23):
                                if gcd(x, y) != 1:
                                    ok = False
                                    break
                            if not ok:
                                break
                        if not ok:
                            continue
                        
                        # Now try b3
                        for i3 in range(i2 + 1, n_valid):
                            b3, s31, s32, s33 = valid_bs[i3]
                            # Check cross-coprimality with first two groups
                            ok = True
                            for x in (s11, s12, s13, s21, s22, s23):
                                for y in (s31, s32, s33):
                                    if gcd(x, y) != 1:
                                        ok = False
                                        break
                                if not ok:
                                    break
                            if not ok:
                                continue
                            
                            # All 9 sums are pairwise coprime!
                            # Check they're all distinct
                            all_sums = [s11, s12, s13, s21, s22, s23, s31, s32, s33]
                            if len(set(all_sums)) == 9:
                                score = 9
                                if score > best:
                                    best = score
                                    best_A = A[:]
                                    best_B = [b1, b2, b3]
                                
                                if time.time() > deadline:
                                    break
                        if time.time() > deadline:
                            break
                    if time.time() > deadline:
                        break
                if time.time() > deadline:
                    break
            if time.time() > deadline:
                break
        if time.time() > deadline:
            break
    
    if best_A is not None and best > 0:
        return best_A, best_B
    
    # Fallback: return the seed
    return [1, 7, 13], [4, 10, 16]

def construct():
    """Return two lists A,B, each containing exactly 3 distinct integers in [1,128]."""
    A, B = _find_best()
    # Verify constraints
    assert len(A) == 3 and len(B) == 3
    assert len(set(A)) == 3 and len(set(B)) == 3
    assert all(1 <= x <= 128 for x in A)
    assert all(1 <= x <= 128 for x in B)
    sums = set(a + b for a in A for b in B)
    assert _pairwise_coprime(list(sums))
    return A, B
# EVOLVE-BLOCK-END

