"""Rewritten program: bipartite/graph search with prime-mask pruning.

We must return two lists A,B of 3 distinct integers in [1,128] such that all
DISTINCT values in the sumset A+B are pairwise coprime, maximising |A+B|.

Key structural insight (parity):
  Pairwise-coprime => at most ONE even number in the whole sumset.
  For a 3x3 additive grid a_i+b_j the number of even sums is
      k*m + (3-k)*(3-m)   (k=#even in A, m=#even in B)
  This equals 1 for NO (k,m) in [0,3]^2, so we must have ZERO even sums.
  That forces (A all odd & B all even) or (A all even & B all odd).
  Hence every sum is ODD and factor 2 never appears; we only need to manage
  odd prime factors.  This halves the search space with no loss.

Search:
  * Precompute a prime-factor bitmask for every integer 2..256 (odd sums only
    ever hit odd primes, but we build for all for simplicity).
  * Enumerate all valid columns: a column is (a, b1, b2, b3) where a is one
    element of A and b1,b2,b3 are the three elements of B.  The three sums
    a+b1, a+b2, a+b3 must be pairwise coprime.
  * For each triple of distinct columns (a1, a2, a3) with the same B, check
    that all 9 sums are pairwise coprime (using bitmask intersection).
  * Track the maximum number of distinct sums found.

The maximum possible distinct sums is 9 (when all 9 sums are different).
"""
import time
from itertools import combinations
from math import gcd

def construct():
    deadline = time.time() + 80  # safety margin before 90s kill

    # --- Precompute prime factor masks ---
    # We only need odd primes since all sums will be odd.
    # But build masks for all numbers up to 256 for generality.
    max_val = 256
    # Collect all primes up to max_val
    primes = []
    is_prime = [True] * (max_val + 1)
    is_prime[0] = is_prime[1] = False
    for i in range(2, max_val + 1):
        if is_prime[i]:
            primes.append(i)
            for j in range(i * i, max_val + 1, i):
                is_prime[j] = False
    
    # Assign each prime an index
    prime_index = {p: i for i, p in enumerate(primes)}
    num_primes = len(primes)
    
    # Build bitmask for each number
    masks = [0] * (max_val + 1)
    for n in range(2, max_val + 1):
        m = n
        mask = 0
        for p in primes:
            if p * p > m:
                break
            if m % p == 0:
                mask |= (1 << prime_index[p])
                while m % p == 0:
                    m //= p
        if m > 1:
            mask |= (1 << prime_index[m])
        masks[n] = mask
    
    # --- Enumerate valid B triples (all even or all odd) ---
    # Case 1: A all odd, B all even
    # Case 2: A all even, B all odd
    
    # For efficiency, let's work with the "column" approach:
    # A column is (a, b) where a is from A and b is from B.
    # Actually, let's think differently.
    
    # Let's enumerate all valid pairs (a, B_triple) where B_triple is a set of 3 distinct values.
    # For a fixed B = {b1, b2, b3}, a value a is "valid" if a+b1, a+b2, a+b3 are pairwise coprime.
    
    # Precompute for each possible B triple, which a values are valid.
    # But that's too many B triples. Let's instead enumerate columns.
    
    # A column is (a, b) - no wait, let's think of it as:
    # For each pair (a, b), the sum s = a+b has a mask.
    # We need 3 a's and 3 b's such that all 9 sums are pairwise coprime.
    
    # Strategy: Enumerate all possible (a1, a2, a3) triples (distinct, same parity)
    # and for each, find B triples that work.
    
    # Actually, let's use the column approach more carefully.
    # A "column" here means: fix one element a from A. The three sums a+b1, a+b2, a+b3
    # must be pairwise coprime. So for a given a and B, the column is valid if
    # the three sums are pairwise coprime.
    
    # Let's precompute for each a (1..128) and each pair (b1, b2) whether a+b1 and a+b2 are coprime.
    # Then for a triple (b1, b2, b3), a is valid if all three pairs are coprime.
    
    # Better: For each a, precompute which b values are "compatible" with a in the sense that
    # for any two b's, the sums are coprime.
    
    # Let's just do a direct search with pruning.
    
    # Generate all odd and even numbers in [1, 128]
    odds = list(range(1, 129, 2))
    evens = list(range(2, 129, 2))
    
    best_count = 0
    best_A = None
    best_B = None
    
    # Case 1: A odd, B even
    # For each triple of distinct odds for A, check if there exists a triple of distinct evens for B
    # such that all 9 sums are pairwise coprime.
    
    # To speed up: for a given A triple, precompute the set of "forbidden" b values.
    # A b is forbidden if there exists a in A such that... no, it's more complex.
    # Two sums a+b and a'+b' conflict if gcd(a+b, a'+b') > 1.
    
    # Let's just enumerate B triples for each A triple, but with pruning.
    
    # For a given A = {a1, a2, a3}, a B value b is "potentially valid" if for all a in A,
    # a+b is odd (which is guaranteed) and... we need the full 3x3 to be pairwise coprime.
    
    # Let's precompute compatibility: for each pair (a, b), the mask of a+b.
    # Then for A and B, check all 9 pairs.
    
    # Optimization: for a fixed A triple, iterate over B triples but prune early.
    
    # Precompute masks for all a+b where a in odds, b in evens (or vice versa)
    # Since a is odd and b is even, a+b is odd, so mask only involves odd primes.
    
    # Let's precompute a 2D array: mask_odd_even[a][b] for a odd, b even
    # And mask_even_odd[a][b] for a even, b odd
    
    # Actually, let's just compute on the fly with the masks array.
    
    def check_pairwise_coprime_9sums(A, B, masks):
        """Check if all 9 sums a+b for a in A, b in B are pairwise coprime."""
        sums = []
        for a in A:
            for b in B:
                s = a + b
                sums.append(s)
        
        # Check all pairs
        n = len(sums)
        for i in range(n):
            for j in range(i + 1, n):
                if masks[sums[i]] & masks[sums[j]]:
                    return False
        return True
    
    def count_distinct(A, B):
        s = set()
        for a in A:
            for b in B:
                s.add(a + b)
        return len(s)
    
    # Case 1: A odd, B even
    # Enumerate A triples from odds
    # For each, enumerate B triples from evens with pruning
    
    # To make this faster, let's precompute for each a (odd) and b (even) the mask
    # and also precompute which b's are "compatible" with a given set of a's.
    
    # For a fixed A triple {a1, a2, a3}, a b is "admissible" if:
    #   gcd(a1+b, a2+b) == 1, gcd(a1+b, a3+b) == 1, gcd(a2+b, a3+b) == 1
    # This is necessary but not sufficient (we also need cross-pairs between different b's).
    
    # Let's precompute for each pair of a's (a1, a2) which b's satisfy gcd(a1+b, a2+b)==1.
    
    # This is getting complex. Let's just do the direct search but with good pruning.
    
    # Precompute: for each a in odds, for each b in evens, the mask of a+b
    # Store as a dict or 2D list
    
    # Since odds and evens are indexed, let's create lookup tables
    odd_to_idx = {v: i for i, v in enumerate(odds)}
    even_to_idx = {v: i for i, v in enumerate(evens)}
    
    n_odds = len(odds)
    n_evens = len(evens)
    
    # mask_oe[i][j] = masks[odds[i] + evens[j]]
    mask_oe = [[0] * n_evens for _ in range(n_odds)]
    for i in range(n_odds):
        a = odds[i]
        for j in range(n_evens):
            b = evens[j]
            mask_oe[i][j] = masks[a + b]
    
    # mask_eo[i][j] = masks[evens[i] + odds[j]]
    mask_eo = [[0] * n_odds for _ in range(n_evens)]
    for i in range(n_evens):
        a = evens[i]
        for j in range(n_odds):
            b = odds[j]
            mask_eo[i][j] = masks[a + b]
    
    # Case 1: A from odds, B from evens
    # For each triple of distinct odd indices (i1, i2, i3), find B triples
    # that work.
    
    # Pruning: for a given A triple, precompute for each b which a's it's compatible with
    # in the sense that the sums are pairwise coprime within the same b.
    
    # Actually, let's just iterate. The space is C(64,3) * C(64,3) which is large,
    # but with early pruning it should be manageable.
    
    # Better approach: for each A triple, find all "admissible" b's (where the three
    # sums with the three a's are pairwise coprime), then check triples of admissible b's.
    
    for i1 in range(n_odds):
        if time.time() > deadline - 5:
            break
        a1 = odds[i1]
        for i2 in range(i1 + 1, n_odds):
            a2 = odds[i2]
            for i3 in range(i2 + 1, n_odds):
                a3 = odds[i3]
                
                # Find admissible b's
                admissible = []
                for j in range(n_evens):
                    b = evens[j]
                    m1 = mask_oe[i1][j]
                    m2 = mask_oe[i2][j]
                    m3 = mask_oe[i3][j]
                    if (m1 & m2) == 0 and (m1 & m3) == 0 and (m2 & m3) == 0:
                        admissible.append(j)
                
                if len(admissible) < 3:
                    continue
                
                # Now check triples of admissible b's
                # For each triple (j1, j2, j3), check all cross-pairs
                for j1_idx in range(len(admissible)):
                    if time.time() > deadline - 5:
                        break
                    j1 = admissible[j1_idx]
                    for j2_idx in range(j1_idx + 1, len(admissible)):
                        j2 = admissible[j2_idx]
                        for j3_idx in range(j2_idx + 1, len(admissible)):
                            j3 = admissible[j3_idx]
                            
                            # Check cross-pairs: a_i + b_j for different i,j
                            # We already know same-column pairs are coprime.
                            # Need to check: (a1+b1, a2+b2), (a1+b1, a3+b3), 
                            # (a1+b2, a2+b1), (a1+b2, a3+b2), (a1+b3, a2+b3),
                            # (a2+b1, a3+b1), (a2+b3, a3+b2), (a1+b3, a2+b1)... 
                            # Actually all 9 choose 2 = 36 pairs, but 9 are same-column (3 per column).
                            # So 27 cross-pairs to check.
                            
                            # Let's just check all pairs for simplicity, with early exit
                            valid = True
                            # Same-column already checked, so check cross
                            # a1+b1 vs a2+b1: m1_j1 & m2_j1
                            # a1+b1 vs a3+b1: m1_j1 & m3_j1
                            # a2+b1 vs a3+b1: m2_j1 & m3_j1
                            # etc.
                            
                            # Check all 36 pairs
                            pairs_to_check = [
                                (i1, j1, i2, j1), (i1, j1, i3, j1),
                                (i2, j1, i3, j1),
                                (i1, j2, i2, j2), (i1, j2, i3, j2),
                                (i2, j2, i3, j2),
                                (i1, j3, i2, j3), (i1, j3, i3, j3),
                                (i2, j3, i3, j3),
                                # Cross-column
                                (i1, j1, i2, j2), (i1, j1, i3, j2),
                                (i1, j1, i2, j3), (i1, j1, i3, j3),
                                (i2, j1, i1, j2), (i2, j1, i3, j2),
                                (i2, j1, i1, j3), (i2, j1, i3, j3),
                                (i3, j1, i1, j2), (i3, j1, i2, j2),
                                (i3, j1, i1, j3), (i3, j1, i2, j3),
                                (i1, j2, i2, j1), (i1, j2, i3, j1),
                                (i1, j2, i2, j3), (i1, j2, i3, j3),
                                (i2, j2, i1, j1), (i2, j2, i3, j1),
                                (i2, j2, i1, j3), (i2, j2, i3, j3),
                                (i3, j2, i1, j1), (i3, j2, i2, j1),
                                (i3, j2, i1, j3), (i3, j2, i2, j3),
                                (i1, j3, i2, j1), (i1, j3, i3, j1),
                                (i1, j3, i2, j2), (i1, j3, i3, j2),
                                (i2, j3, i1, j1), (i2, j3, i3, j1),
                                (i2, j3, i1, j2), (i2, j3, i3, j2),
                                (i3, j3, i1, j1), (i3, j3, i2, j1),
                                (i3, j3, i1, j2), (i3, j3, i2, j2),
                            ]
                            
                            # This is too many. Let's just check all 36 pairs directly.
                            # Actually, let's just compute all 9 sums and check all pairs.
                            sums_list = [
                                a1 + evens[j1], a1 + evens[j2], a1 + evens[j3],
                                a2 + evens[j1], a2 + evens[j2], a2 + evens[j3],
                                a3 + evens[j1], a3 + evens[j2], a3 + evens[j3],
                            ]
                            
                            for p in range(9):
                                if not valid:
                                    break
                                for q in range(p + 1, 9):
                                    if masks[sums_list[p]] & masks[sums_list[q]]:
                                        valid = False
                                        break
                            
                            if valid:
                                A = [a1, a2, a3]
                                B = [evens[j1], evens[j2], evens[j3]]
                                cnt = count_distinct(A, B)
                                if cnt > best_count:
                                    best_count = cnt
                                    best_A = A
                                    best_B = B
                                    if best_count == 9:
                                        return [best_A, best_B]
    
    # Case 2: A even, B odd
    for i1 in range(n_evens):
        if time.time() > deadline - 5:
            break
        a1 = evens[i1]
        for i2 in range(i1 + 1, n_evens):
            a2 = evens[i2]
            for i3 in range(i2 + 1, n_evens):
                a3 = evens[i3]
                
                admissible = []
                for j in range(n_odds):
                    b = odds[j]
                    m1 = mask_eo[i1][j]
                    m2 = mask_eo[i2][j]
                    m3 = mask_eo[i3][j]
                    if (m1 & m2) == 0 and (m1 & m3) == 0 and (m2 & m3) == 0:
                        admissible.append(j)
                
                if len(admissible) < 3:
                    continue
                
                for j1_idx in range(len(admissible)):
                    if time.time() > deadline - 5:
                        break
                    j1 = admissible[j1_idx]
                    for j2_idx in range(j1_idx + 1, len(admissible)):
                        j2 = admissible[j2_idx]
                        for j3_idx in range(j2_idx + 1, len(admissible)):
                            j3 = admissible[j3_idx]
                            
                            sums_list = [
                                a1 + odds[j1], a1 + odds[j2], a1 + odds[j3],
                                a2 + odds[j1], a2 + odds[j2], a2 + odds[j3],
                                a3 + odds[j1], a3 + odds[j2], a3 + odds[j3],
                            ]
                            
                            valid = True
                            for p in range(9):
                                if not valid:
                                    break
                                for q in range(p + 1, 9):
                                    if masks[sums_list[p]] & masks[sums_list[q]]:
                                        valid = False
                                        break
                            
                            if valid:
                                A = [a1, a2, a3]
                                B = [odds[j1], odds[j2], odds[j3]]
                                cnt = count_distinct(A, B)
                                if cnt > best_count:
                                    best_count = cnt
                                    best_A = A
                                    best_B = B
                                    if best_count == 9:
                                        return [best_A, best_B]
    
    # Return best found, or a default if nothing found
    if best_A is not None:
        return [best_A, best_B]
    else:
        # Fallback: return the seed solution
        return [[1, 7, 13], [4, 10, 16]]
