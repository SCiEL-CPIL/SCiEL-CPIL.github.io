"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math
from fractions import Fraction

def construct():
    start_time = time.time()
    DEADLINE = 85.0
    TARGET = Fraction(5, 31)
    MAX_DEN = 10_000_000
    MAX_TERMS = 256

    # Known 3-term solution: 1/8 + 1/31 + 1/248 = 5/31
    # Verify: 1/8 + 1/31 + 1/248 = (31 + 8 + 1)/248 = 40/248 = 5/31
    # Largest denominator is 248.
    
    # Let's verify if we can do better than 248.
    # For 3 terms with a <= b <= c:
    # 1/a + 1/b + 1/c = 5/31
    # We need to check if there's a solution with c < 248.
    
    # From the number-theoretic approach:
    # For each a in [7, 18]:
    #   N = 5a - 31, D = 31a
    #   (Nb - D)(Nc - D) = D^2
    #   So for each divisor p of D^2:
    #     b = (p + D) / N, c = (D^2/p + D) / N
    #   We need b, c to be positive integers, distinct, with a <= b <= c.
    
    # Let's exhaustively search for the minimum c over all 3-term solutions.
    
    def find_min_3term():
        """Find the minimum largest denominator c in any 3-term Egyptian fraction of 5/31."""
        best_c = MAX_DEN
        best_sol = None
        
        for a in range(7, 19):
            N = 5 * a - 31
            D = 31 * a
            if N <= 0:
                continue
            
            D2 = D * D
            
            # Enumerate divisors of D^2
            # D = 31a, so D^2 = 961 * a^2
            # We need divisors p of D^2 such that:
            #   (p + D) % N == 0 and (D^2//p + D) % N == 0
            #   b = (p + D) // N >= a
            #   c = (D^2//p + D) // N >= b
            #   a, b, c all distinct
            
            # To be efficient, note that p must satisfy (p + D) ≡ 0 (mod N)
            # So p ≡ -D (mod N)
            # p = k*N - D for some integer k
            # And p must divide D^2
            # Also p > 0, so k*N > D, i.e., k > D/N = 31a/(5a-31)
            
            # For a=7: D/N = 217/4 = 54.25, so k >= 55
            # For a=18: D/N = 558/59 ≈ 9.46, so k >= 10
            
            # Let's just enumerate divisors directly for correctness
            divisors = []
            limit = int(math.isqrt(D2))
            for i in range(1, limit + 1):
                if D2 % i == 0:
                    divisors.append(i)
                    if i != D2 // i:
                        divisors.append(D2 // i)
            
            for p in divisors:
                q = D2 // p
                if (p + D) % N != 0:
                    continue
                if (q + D) % N != 0:
                    continue
                b = (p + D) // N
                c = (q + D) // N
                if b < 1 or c < 1:
                    continue
                if b > c:
                    # Swap and recheck
                    b, c = c, b
                if a > b:
                    continue
                if a == b or a == c or b == c:
                    continue
                if c < best_c:
                    best_c = c
                    best_sol = [a, b, c]
        
        return best_sol, best_c
    
    sol, max_c = find_min_3term()
    
    if sol is None:
        sol = [8, 31, 248]
    
    # Verify the solution
    s = sum(Fraction(1, d) for d in sol)
    if s != TARGET or len(set(sol)) != len(sol) or any(d < 1 or d > MAX_DEN for d in sol) or len(sol) > MAX_TERMS:
        sol = [8, 31, 248]
    
    return sol
# EVOLVE-BLOCK-END
