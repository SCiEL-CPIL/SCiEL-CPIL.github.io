import time
from math import gcd

def construct():
    deadline = time.time() + 85  # 5s safety margin
    
    max_val = 256
    # Sieve for primes up to 256
    is_prime = [True] * (max_val + 1)
    is_prime[0] = is_prime[1] = False
    for i in range(2, int(max_val**0.5) + 1):
        if is_prime[i]:
            for j in range(i*i, max_val + 1, i):
                is_prime[j] = False
    primes = [i for i in range(2, max_val + 1) if is_prime[i]]
    prime_index = {p: i for i, p in enumerate(primes)}
    
    # Precompute masks
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
        
    odds = list(range(1, 129, 2))
    evens = list(range(2, 129, 2))
    
    n_odds = len(odds)
    n_evens = len(evens)
    
    # Precompute masks for sums
    # mask_oe[i][j] = masks[odds[i] + evens[j]]
    mask_oe = [[0] * n_evens for _ in range(n_odds)]
    for i in range(n_odds):
        a = odds[i]
        for j in range(n_evens):
            mask_oe[i][j] = masks[a + evens[j]]
            
    mask_eo = [[0] * n_odds for _ in range(n_evens)]
    for i in range(n_evens):
        a = evens[i]
        for j in range(n_odds):
            mask_eo[i][j] = masks[a + odds[j]]

    def is_pairwise_coprime_9(m1, m2, m3, m4, m5, m6, m7, m8, m9):
        # Check if all 9 masks are pairwise disjoint
        # Equivalent to: popcount(OR of all) == sum of popcounts
        # This is faster than 36 ANDs
        or_all = m1 | m2 | m3 | m4 | m5 | m6 | m7 | m8 | m9
        sum_pop = m1.bit_count() + m2.bit_count() + m3.bit_count() + m4.bit_count() + \
                  m5.bit_count() + m6.bit_count() + m7.bit_count() + m8.bit_count() + m9.bit_count()
        return or_all.bit_count() == sum_pop

    best_count = 0
    best_A = None
    best_B = None
    
    # Helper to check and update best
    def try_update(A, B):
        nonlocal best_count, best_A, best_B
        sums = [a + b for a in A for b in B]
        distinct = len(set(sums))
        if distinct > best_count:
            best_count = distinct
            best_A = A
            best_B = B
            return best_count == 9
        return False

    # Case 1: A odd, B even
    # We iterate A triples. To speed up, we can check if a B triple is valid.
    # For a fixed A, we find admissible B's.
    
    for i1 in range(n_odds):
        if time.time() > deadline:
            break
        a1 = odds[i1]
        for i2 in range(i1 + 1, n_odds):
            a2 = odds[i2]
            for i3 in range(i2 + 1, n_odds):
                a3 = odds[i3]
                
                # Find admissible B indices
                # A B index j is admissible if the 3 sums (a1+b_j, a2+b_j, a3+b_j) are pairwise coprime
                admissible = []
                for j in range(n_evens):
                    m1 = mask_oe[i1][j]
                    m2 = mask_oe[i2][j]
                    m3 = mask_oe[i3][j]
                    # Check pairwise disjoint for these 3
                    if (m1 & m2) == 0 and (m1 & m3) == 0 and (m2 & m3) == 0:
                        admissible.append(j)
                        
                if len(admissible) < 3:
                    continue
                    
                # Check triples of admissible B's
                # We can prune B triples by checking cross-coprimality early
                L = len(admissible)
                for j1_idx in range(L):
                    if time.time() > deadline:
                        break
                    j1 = admissible[j1_idx]
                    for j2_idx in range(j1_idx + 1, L):
                        j2 = admissible[j2_idx]
                        # Precompute masks for column 1 and 2
                        # c1 = [m(a1,b1), m(a2,b1), m(a3,b1)]
                        # c2 = [m(a1,b2), m(a2,b2), m(a3,b2)]
                        # We need all cross pairs to be disjoint.
                        # And we need to find a j3 such that c3 is disjoint from c1 and c2.
                        
                        m11 = mask_oe[i1][j1]; m21 = mask_oe[i2][j1]; m31 = mask_oe[i3][j1]
                        m12 = mask_oe[i1][j2]; m22 = mask_oe[i2][j2]; m32 = mask_oe[i3][j2]
                        
                        # Check cross pairs between col 1 and col 2
                        # 9 pairs
                        if (m11 & m12) or (m11 & m22) or (m11 & m32) or \
                           (m21 & m12) or (m21 & m22) or (m21 & m32) or \
                           (m31 & m12) or (m31 & m22) or (m31 & m32):
                            continue
                            
                        # Now find j3
                        # j3 must be disjoint from col 1 and col 2
                        # And col 3 must be internally disjoint (already guaranteed by admissible list? 
                        # Yes, admissible list ensures internal disjointness for any a-triple and b)
                        
                        for j3_idx in range(j2_idx + 1, L):
                            j3 = admissible[j3_idx]
                            m13 = mask_oe[i1][j3]; m23 = mask_oe[i2][j3]; m33 = mask_oe[i3][j3]
                            
                            # Check cross pairs between col 3 and col 1
                            if (m13 & m11) or (m13 & m21) or (m13 & m31) or \
                               (m23 & m11) or (m23 & m21) or (m23 & m31) or \
                               (m33 & m11) or (m33 & m21) or (m33 & m31):
                                continue
                                
                            # Check cross pairs between col 3 and col 2
                            if (m13 & m12) or (m13 & m22) or (m13 & m32) or \
                               (m23 & m12) or (m23 & m22) or (m23 & m32) or \
                               (m33 & m12) or (m33 & m22) or (m33 & m32):
                                continue
                                
                            # All 9 sums are pairwise coprime
                            A = [a1, a2, a3]
                            B = [evens[j1], evens[j2], evens[j3]]
                            if try_update(A, B):
                                return [A, B]
                                
    # Case 2: A even, B odd
    for i1 in range(n_evens):
        if time.time() > deadline:
            break
        a1 = evens[i1]
        for i2 in range(i1 + 1, n_evens):
            a2 = evens[i2]
            for i3 in range(i2 + 1, n_evens):
                a3 = evens[i3]
                
                admissible = []
                for j in range(n_odds):
                    m1 = mask_eo[i1][j]
                    m2 = mask_eo[i2][j]
                    m3 = mask_eo[i3][j]
                    if (m1 & m2) == 0 and (m1 & m3) == 0 and (m2 & m3) == 0:
                        admissible.append(j)
                        
                if len(admissible) < 3:
                    continue
                    
                L = len(admissible)
                for j1_idx in range(L):
                    if time.time() > deadline:
                        break
                    j1 = admissible[j1_idx]
                    for j2_idx in range(j1_idx + 1, L):
                        j2 = admissible[j2_idx]
                        
                        m11 = mask_eo[i1][j1]; m21 = mask_eo[i2][j1]; m31 = mask_eo[i3][j1]
                        m12 = mask_eo[i1][j2]; m22 = mask_eo[i2][j2]; m32 = mask_eo[i3][j2]
                        
                        if (m11 & m12) or (m11 & m22) or (m11 & m32) or \
                           (m21 & m12) or (m21 & m22) or (m21 & m32) or \
                           (m31 & m12) or (m31 & m22) or (m31 & m32):
                            continue
                            
                        for j3_idx in range(j2_idx + 1, L):
                            j3 = admissible[j3_idx]
                            m13 = mask_eo[i1][j3]; m23 = mask_eo[i2][j3]; m33 = mask_eo[i3][j3]
                            
                            if (m13 & m11) or (m13 & m21) or (m13 & m31) or \
                               (m23 & m11) or (m23 & m21) or (m23 & m31) or \
                               (m33 & m11) or (m33 & m21) or (m33 & m31):
                                continue
                                
                            if (m13 & m12) or (m13 & m22) or (m13 & m32) or \
                               (m23 & m12) or (m23 & m22) or (m23 & m32) or \
                               (m33 & m12) or (m33 & m22) or (m33 & m32):
                                continue
                                
                            A = [a1, a2, a3]
                            B = [odds[j1], odds[j2], odds[j3]]
                            if try_update(A, B):
                                return [A, B]
                                
    if best_A is not None:
        return [best_A, best_B]
    else:
        return [[1, 7, 13], [4, 10, 16]]
