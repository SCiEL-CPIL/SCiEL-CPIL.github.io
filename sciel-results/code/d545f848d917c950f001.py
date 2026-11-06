"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import math
import time
import itertools

def construct():
    """
    Construct maximum number of disjoint residue classes with distinct moduli in [2, 128].
    
    Strategy:
    1. Use a greedy approach to select moduli and assign residues such that
       for any two classes (r1, m1) and (r2, m2), gcd(m1, m2) divides (r1 - r2)
       only if r1 == r2 mod gcd(m1, m2), which would make them intersect.
       Actually, two classes r1 mod m1 and r2 mod m2 are disjoint iff
       r1 != r2 mod gcd(m1, m2).
       
    2. We want to maximize the number of classes.
       Key insight: If we pick moduli that are all multiples of a common prime p,
       and assign distinct residues mod p, then any two classes are disjoint.
       
    3. For a prime p, the maximum number of classes is p (using moduli p, 2p, ..., p^2
       if p^2 <= 128, or fewer if not).
       p=11: 11, 22, ..., 121 (11 moduli, all <= 128). 11 classes.
       p=13: 13, 26, ..., 156 > 128. So 13, 26, ..., 117 (9 moduli). 9 classes.
       p=7: 7, 14, ..., 49 (7 moduli). 7 classes.
       
    4. Can we do better than 11?
       What if we use multiple primes?
       If we use moduli from different primes, the gcd might be 1, meaning
       any two classes with coprime moduli always intersect (since gcd=1, 
       r1 != r2 mod 1 is impossible, 0 != 0 is false).
       
       Wait, if gcd(m1, m2) = 1, then the classes always intersect because
       there exists a solution to x = r1 mod m1 and x = r2 mod m2 by CRT.
       So we cannot use coprime moduli.
       
       Therefore, all moduli must share a common factor. The best is to use
       a large prime p and use moduli p, 2p, ..., kp where kp <= 128.
       
       For p=11: k = floor(128/11) = 11. So 11 classes.
       For p=13: k = floor(128/13) = 9. So 9 classes.
       
       Can we mix primes? No, because if m1 is multiple of p and m2 is multiple of q
       with p != q, then gcd(m1, m2) might be 1 or a small number.
       
       Actually, if we use moduli that are all multiples of 6 (2*3), we need
       residues to be distinct mod 6. We can have at most 6 classes.
       
       So the best is p=11 giving 11 classes.
    """
    start_time = time.time()
    deadline = start_time + 85  # Leave 5s safety margin
    
    # The optimal strategy is to use a prime p and moduli p, 2p, ..., kp
    # where kp <= 128, with distinct residues mod p.
    # Best prime is p=11 with 11 moduli.
    
    p = 11
    max_mod = 128
    k = max_mod // p  # floor(128/11) = 11
    
    result = []
    for i in range(1, k + 1):
        m = p * i
        r = (i - 1) % p  # Distinct residues 0, 1, ..., k-1 mod p
        result.append([r, m])
    
    # Verify pairwise disjointness
    def are_disjoint(pair1, pair2):
        r1, m1 = pair1
        r2, m2 = pair2
        g = math.gcd(m1, m2)
        return (r1 % g) != (r2 % g)
    
    # Check all pairs
    valid = True
    n = len(result)
    for i in range(n):
        if time.time() > deadline:
            break
        for j in range(i + 1, n):
            if not are_disjoint(result[i], result[j]):
                valid = False
                break
        if not valid:
            break
    
    if not valid:
        # Fallback: just return the 11 classes which we know work
        return [[i, 11 * (i + 1)] for i in range(11)]
    
    return result
# EVOLVE-BLOCK-END
