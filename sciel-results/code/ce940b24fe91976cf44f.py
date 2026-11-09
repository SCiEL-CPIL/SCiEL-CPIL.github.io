"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from math import gcd

def construct():
    deadline = time.time() + 80  # safety margin before 90s kill
    
    def is_prime(n):
        if n < 2:
            return False
        if n < 4:
            return True
        if n % 2 == 0 or n % 3 == 0:
            return False
        i = 5
        while i * i <= n:
            if n % i == 0 or n % (i + 2) == 0:
                return False
            i += 6
        return True
    
    def is_pairwise_coprime(values):
        """Check if all distinct values in the list are pairwise coprime."""
        unique_vals = list(set(values))
        n = len(unique_vals)
        for i in range(n):
            for j in range(i + 1, n):
                if gcd(unique_vals[i], unique_vals[j]) != 1:
                    return False
        return True
    
    def score(A, B):
        """Return the number of distinct values in A+B if they are pairwise coprime, else 0."""
        sums = [a + b for a in A for b in B]
        distinct = list(set(sums))
        if len(distinct) < 2:
            return 0
        if is_pairwise_coprime(distinct):
            return len(distinct)
        return 0
    
    # Precompute primes up to 256 (max sum is 128+128=256)
    primes = set()
    for p in range(2, 257):
        if is_prime(p):
            primes.add(p)
    
    # Strategy: find A, B such that all 9 sums are distinct primes
    # This guarantees pairwise coprime and |A+B| = 9 (maximum)
    
    best_score = 0
    best_A = [1, 7, 13]
    best_B = [4, 10, 16]
    
    # Try to find a configuration with 9 distinct prime sums
    # A = [a1, a2, a3], B = [b1, b2, b3]
    # Sums: a1+b1, a1+b2, a1+b3, a2+b1, a2+b2, a2+b3, a3+b1, a3+b2, a3+b3
    
    # We can fix A and search for B, or use a smarter approach.
    # Let's try small values for A first, then search B.
    
    # Approach: iterate over candidate A triples, then for each, try B triples
    # where all 9 sums are primes.
    
    # To speed up, we can note that if we want all 9 sums to be primes,
    # we need a_i + b_j to be prime for all i,j.
    
    # Let's try a structured search:
    # For A, try values in [1, 50] (smaller range for speed)
    # For B, try values in [1, 128]
    
    # Actually, let's be smarter. If all sums are prime, then:
    # a1+b1, a1+b2, a1+b3 are all prime
    # a2+b1, a2+b2, a2+b3 are all prime
    # a3+b1, a3+b2, a3+b3 are all prime
    #
    # Note that (a2+b1) - (a1+b1) = a2 - a1 must be such that if a1+b1 is prime,
    # a2+b1 is also prime. This is a prime constellation.
    
    # Let's just do a bounded search with the deadline guard.
    
    # Try specific known good candidates first
    # A = [1, 2, 3], B = [2, 3, 4]: sums = 3,4,5,4,5,6,5,6,7 -> distinct: {3,4,5,6,7}
    # 3,4 coprime; 3,5 coprime; 3,6 not coprime (gcd=3). Fail.
    
    # Let's try A = [1, 3, 5], B = [2, 4, 6]: sums = 3,5,7,5,7,9,7,9,11 -> distinct {3,5,7,9,11}
    # 3,9 not coprime. Fail.
    
    # Try to find all-prime sums.
    # A = [1, 2, 4], B = [2, 3, 5]: sums = 3,4,6, 4,5,7, 6,7,9 -> not all prime
    
    # Let me just search.
    # To keep it fast, I'll search A in [1,30] and B in [1,128] with pruning.
    
    found = False
    
    # Search A triples with small values
    for a1 in range(1, 31):
        if time.time() > deadline - 10:
            break
        for a2 in range(a1 + 1, 31):
            if time.time() > deadline - 10:
                break
            for a3 in range(a2 + 1, 31):
                if time.time() > deadline - 10:
                    break
                A = [a1, a2, a3]
                # For each B triple, check if all 9 sums are prime
                for b1 in range(1, 129):
                    if time.time() > deadline - 5:
                        break
                    # Prune: a1+b1 must be prime
                    if not is_prime(a1 + b1):
                        continue
                    for b2 in range(b1 + 1, 129):
                        if time.time() > deadline - 5:
                            break
                        if not is_prime(a1 + b2):
                            continue
                        if not is_prime(a2 + b1):
                            continue
                        for b3 in range(b2 + 1, 129):
                            if time.time() > deadline - 5:
                                break
                            if not is_prime(a1 + b3):
                                continue
                            if not is_prime(a2 + b2):
                                continue
                            if not is_prime(a2 + b3):
                                continue
                            if not is_prime(a3 + b1):
                                continue
                            if not is_prime(a3 + b2):
                                continue
                            if not is_prime(a3 + b3):
                                continue
                            # All 9 sums are prime - check they are distinct
                            B = [b1, b2, b3]
                            sums = [a + b for a in A for b in B]
                            if len(set(sums)) == 9:
                                best_A = A
                                best_B = B
                                best_score = 9
                                found = True
                                break
                        if found:
                            break
                    if found:
                        break
                if found:
                    break
            if found:
                break
        if found:
            break
    
    if found:
        return [best_A, best_B]
    
    # If no 9-prime solution found, fall back to best pairwise coprime with max distinct
    # Search for best score < 9
    best_score = 0
    for a1 in range(1, 41):
        if time.time() > deadline - 5:
            break
        for a2 in range(a1 + 1, 41):
            if time.time() > deadline - 5:
                break
            for a3 in range(a2 + 1, 41):
                if time.time() > deadline - 5:
                    break
                A = [a1, a2, a3]
                for b1 in range(1, 129):
                    if time.time() > deadline - 5:
                        break
                    for b2 in range(b1 + 1, 129):
                        if time.time() > deadline - 5:
                            break
                        for b3 in range(b2 + 1, 129):
                            if time.time() > deadline - 5:
                                break
                            B = [b1, b2, b3]
                            s = score(A, B)
                            if s > best_score:
                                best_score = s
                                best_A = A
                                best_B = B
                            if best_score == 9:
                                return [best_A, best_B]
    
    return [best_A, best_B]
# EVOLVE-BLOCK-END
