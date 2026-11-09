import time
from fractions import Fraction
from math import isqrt

TARGET = Fraction(5, 31)
MAX_DENOM = 10_000_000
MAX_TERMS = 256

def _get_divisors(n):
    """Get all divisors of n."""
    if n == 0:
        return []
    divs = set()
    for i in range(1, isqrt(n) + 1):
        if n % i == 0:
            divs.add(i)
            divs.add(n // i)
    return sorted(divs)

def _find_best_3_term():
    """
    Find the 3-term solution with the minimum maximum denominator.
    Returns (max_denom, [a, b, c]) or (inf, None).
    """
    best_max = float('inf')
    best_sol = None
    
    # a < b < c
    # 3/a > 5/31 => a < 18.6 => a <= 18
    # 1/a < 5/31 => a > 6.2 => a >= 7
    for a in range(7, 19):
        rem = TARGET - Fraction(1, a)
        if rem <= 0:
            continue
        
        N = rem.numerator
        D = rem.denominator
        
        # (Nb - D)(Nc - D) = D^2
        # X = Nb - D, Y = Nc - D
        # X * Y = D^2
        # b = (X + D) / N, c = (Y + D) / N
        # X, Y must be divisors of D^2
        # X % N == -D % N
        
        target_mod = (-D) % N
        divs = _get_divisors(D * D)
        
        for X in divs:
            if X % N != target_mod:
                continue
            Y = (D * D) // X
            if Y % N != target_mod:
                continue
            
            b = (X + D) // N
            c = (Y + D) // N
            
            if b <= a or c <= b:
                continue
            
            # Check distinctness and sum
            if a != b and b != c and a != c:
                if Fraction(1, a) + Fraction(1, b) + Fraction(1, c) == TARGET:
                    max_d = max(a, b, c)
                    if max_d < best_max:
                        best_max = max_d
                        best_sol = [a, b, c]
                        
    return best_max, best_sol

def _find_best_4_term():
    """
    Find the 4-term solution with the minimum maximum denominator.
    Returns (max_denom, [a, b, c, d]) or (inf, None).
    """
    best_max = float('inf')
    best_sol = None
    
    # a < b < c < d
    # 4/a > 5/31 => a < 24.8 => a <= 24
    # 1/a < 5/31 => a > 6.2 => a >= 7
    for a in range(7, 25):
        rem_a = TARGET - Fraction(1, a)
        if rem_a <= 0:
            continue
        
        N_a = rem_a.numerator
        D_a = rem_a.denominator
        
        # 1/b + 1/c + 1/d = rem_a
        # 3/b > rem_a => b < 3 / rem_a = 3 * D_a / N_a
        # 1/b < rem_a => b > 1 / rem_a = D_a / N_a
        # Also b > a
        
        b_min = a + 1
        b_max = min(int(3 * D_a / N_a), 10000000) # Upper bound on b
        
        for b in range(b_min, b_max + 1):
            rem_b = rem_a - Fraction(1, b)
            if rem_b <= 0:
                continue
            
            N_b = rem_b.numerator
            D_b = rem_b.denominator
            
            # 1/c + 1/d = rem_b
            # (N_b c - D_b)(N_b d - D_b) = D_b^2
            # X = N_b c - D_b, Y = N_b d - D_b
            # X * Y = D_b^2
            # c = (X + D_b) / N_b, d = (Y + D_b) / N_b
            
            target_mod = (-D_b) % N_b
            divs = _get_divisors(D_b * D_b)
            
            for X in divs:
                if X % N_b != target_mod:
                    continue
                Y = (D_b * D_b) // X
                if Y % N_b != target_mod:
                    continue
                
                c = (X + D_b) // N_b
                d = (Y + D_b) // N_b
                
                if c <= b or d <= c:
                    continue
                
                if a != b and b != c and c != d and a != c and a != d and b != d:
                    if Fraction(1, a) + Fraction(1, b) + Fraction(1, c) + Fraction(1, d) == TARGET:
                        max_d = max(a, b, c, d)
                        if max_d < best_max:
                            best_max = max_d
                            best_sol = [a, b, c, d]
                            
    return best_max, best_sol

def construct():
    start_time = time.time()
    deadline = start_time + 50.0
    
    # Known solution
    known = [8, 31, 248]
    best = known
    best_max = 248
    
    # Search for 3-term solution
    if time.time() < deadline:
        max_3, sol_3 = _find_best_3_term()
        if sol_3 and max_3 < best_max:
            best = sol_3
            best_max = max_3
            
    # Search for 4-term solution
    if time.time() < deadline:
        max_4, sol_4 = _find_best_4_term()
        if sol_4 and max_4 < best_max:
            best = sol_4
            best_max = max_4
            
    # If we have time, we could try 5 terms, but it's computationally expensive.
    # Given the stagnation, a structural change to exhaustive small-term search is better.
    
    return best
