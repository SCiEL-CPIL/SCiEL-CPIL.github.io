import time
from math import gcd
from itertools import combinations, permutations
import random

def construct():
    deadline = time.time() + 82  # safety margin under 90s wall-clock

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

    # Generate primes up to 300 (sums of two numbers in [1,128] can be up to 256)
    def sieve(n):
        is_p = [True] * (n + 1)
        is_p[0] = is_p[1] = False
        for i in range(2, int(n**0.5) + 1):
            if is_p[i]:
                for j in range(i*i, n+1, i):
                    is_p[j] = False
        return [i for i in range(2, n+1) if is_p[i]]

    primes = sieve(300)
    prime_set = set(primes)

    # Strategy: Search for 3x3 grid where all 9 sums are prime (hence pairwise coprime)
    # A = [a1, a2, a3], B = [b1, b2, b3]
    # Sums: a_i + b_j must all be prime
    # Let a1=1 (WLOG by shifting? No, we can't shift freely due to bounds, but we can try small a1)
    # Actually, let's iterate over small a1, a2, a3 and b1, b2, b3
    # To be efficient, we can fix a1=1, a2, a3 and check for b1, b2, b3
    # But that's 128^5 which is too big.
    
    # Better: Iterate over pairs (a1, a2) and (b1, b2)
    # Given a1, a2, b1, b2:
    # s11 = a1+b1 (must be prime)
    # s12 = a1+b2 (must be prime)
    # s21 = a2+b1 (must be prime)
    # s22 = a2+b2 (must be prime)
    # If these 4 are prime, we have a 2x2 grid of primes.
    # Now we need a3 and b3 such that:
    # a3+b1, a3+b2, a1+b3, a2+b3, a3+b3 are all prime.
    
    # Let's iterate a1, a2, b1, b2
    # Number of pairs (a1,a2) with a1 != a2 in [1,128] is ~128*127
    # Number of pairs (b1,b2) is ~128*127
    # Total ~ 2M * 2M = 4T, too big.
    
    # We can restrict to small values first.
    # Let's try a1, a2, b1, b2 in [1, 50] first.
    # 50*49 * 50*49 ~ 6M, which is feasible in Python if optimized? Maybe slow.
    # Let's use a smarter approach.
    
    # For a fixed a1, a2, the differences d_a = a2 - a1.
    # For a fixed b1, b2, the differences d_b = b2 - b1.
    # s12 - s11 = d_b
    # s21 - s11 = d_a
    # s22 - s11 = d_a + d_b
    # So we need primes p1, p1+d_b, p1+d_a, p1+d_a+d_b.
    
    # Let's iterate d_a, d_b in a reasonable range, say [-100, 100]
    # And p1 in primes.
    # For each (d_a, d_b, p1), we have a 2x2 prime grid.
    # Then we need to find a3, b3 such that a3+b1, a3+b2, a1+b3, a2+b3, a3+b3 are prime.
    # This is still complex.
    
    # Alternative: Direct search for 9 pairwise coprime sums.
    # We know primes are pairwise coprime.
    # Let's try to find A, B such that all 9 sums are prime.
    
    # We can iterate over A (3 distinct in [1,128]) and for each A, find B.
    # Number of A is C(128,3) ~ 340k. For each A, finding B is hard.
    
    # Let's go back to the 2x2 grid idea but optimize.
    # Iterate a1, a2 in [1, 60], b1 in [1, 60].
    # For each (a1, a2, b1), find b2 such that a1+b1, a1+b2, a2+b1, a2+b2 are prime.
    # Let p1 = a1+b1.
    # a1+b2 = p1 + (b2-b1) = p1 + d_b
    # a2+b1 = p1 + (a2-a1) = p1 + d_a
    # a2+b2 = p1 + d_a + d_b
    # So we need d_b such that p1+d_b is prime, and p1+d_a+d_b is prime.
    # And b2 = b1 + d_b must be in [1,128] and distinct from b1.
    
    # Let's precompute for each p1 (prime), and d_a (diff of a's), the valid d_b's.
    
    # Even simpler: Just brute force with pruning.
    # Iterate a1, a2, a3, b1, b2, b3
    # But we can fix a1=1, a2=2, a3=3? No, sums must be prime.
    # 1+1=2 (prime), 1+2=3 (prime), 1+3=4 (not prime). So B cannot have 3 if A has 1.
    
    # Let's try a structured search.
    # We know that if all sums are prime, they are pairwise coprime.
    # Let's search for A, B where all 9 sums are prime.
    
    # We'll iterate over A from a smaller range first, say [1, 30]
    # And B from [1, 30]
    # If found, great. If not, expand.
    
    # But 30^3 * 30^3 = 729M, too big.
    
    # Let's use the 2x2 grid extension.
    # Iterate a1, a2, b1, b2 such that the 2x2 sums are prime.
    # Then try to extend to 3x3.
    
    # To make it fast, we iterate a1, a2, b1, b2 in [1, 40].
    # 40^4 = 2.56M. For each, check 4 primes. Fast.
    # Then for each valid 2x2, try to find a3, b3.
    # a3, b3 in [1, 128].
    # Check 5 new primes.
    
    # This is feasible.
    
    found = False
    search_limit = 50
    
    # Precompute prime set for fast lookup
    pset = set(primes)
    
    for a1 in range(1, search_limit + 1):
        for a2 in range(1, search_limit + 1):
            if a1 == a2: continue
            for b1 in range(1, search_limit + 1):
                for b2 in range(1, search_limit + 1):
                    if b1 == b2: continue
                    s11 = a1 + b1
                    s12 = a1 + b2
                    s21 = a2 + b1
                    s22 = a2 + b2
                    if s11 in pset and s12 in pset and s21 in pset and s22 in pset:
                        # Found 2x2 grid of primes
                        # Now find a3, b3
                        for a3 in range(1, 129):
                            if a3 == a1 or a3 == a2: continue
                            for b3 in range(1, 129):
                                if b3 == b1 or b3 == b2: continue
                                s13 = a1 + b3
                                s23 = a2 + b3
                                s31 = a3 + b1
                                s32 = a3 + b2
                                s33 = a3 + b3
                                if (s13 in pset and s23 in pset and s31 in pset and 
                                    s32 in pset and s33 in pset):
                                    # Check all 9 are distinct? Not required, but pairwise coprime is.
                                    # Primes are pairwise coprime if distinct.
                                    # If some are equal, they are not coprime (gcd(p,p)=p>1).
                                    # So we need all 9 sums to be DISTINCT primes.
                                    sums = [s11, s12, s13, s21, s22, s23, s31, s32, s33]
                                    if len(set(sums)) == 9:
                                        A = sorted([a1, a2, a3])
                                        B = sorted([b1, b2, b3])
                                        sc = score_of(A, B)
                                        if sc == 9:
                                            return A, B
                                        if sc > best_score:
                                            best_score = sc
                                            best_A = A
                                            best_B = B
                        if time.time() > deadline - 5:
                            break
                if time.time() > deadline - 5:
                    break
            if time.time() > deadline - 5:
                break
        if time.time() > deadline - 5:
            break
        if found:
            break
    
    # If not found, fallback to random search for best score
    if best_score < 9:
        random.seed(42)
        while time.time() < deadline - 2:
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
