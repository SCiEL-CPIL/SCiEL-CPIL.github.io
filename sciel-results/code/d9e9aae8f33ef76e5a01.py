import time
from fractions import Fraction
from math import isqrt

TARGET = Fraction(5, 31)
MAX_DENOM = 10_000_000
MAX_TERMS = 256

def _get_divisors(n):
    """Generate all divisors of n."""
    if n == 0:
        return []
    divisors = set()
    for i in range(1, isqrt(n) + 1):
        if n % i == 0:
            divisors.add(i)
            divisors.add(n // i)
    return divisors

def _find_solution_for_max(M):
    """
    Try to find a set of distinct denominators in [1, M] that sum to TARGET.
    Returns a list of denominators if found, else None.
    Assumes M is the largest denominator.
    """
    # 2-term: 1/a + 1/M = 5/31
    # 1/a = 5/31 - 1/M = (5M - 31) / (31M)
    # a = 31M / (5M - 31)
    num = 31 * M
    den = 5 * M - 31
    if den > 0 and num % den == 0:
        a = num // den
        if 1 <= a < M:
            return [a, M]
            
    # 3-term: 1/a + 1/b + 1/M = 5/31
    # 1/a + 1/b = 5/31 - 1/M = R
    # a < b < M
    # Iterate a. Since 1/a < R, a > 1/R.
    # Also 2/a > R (since 1/a + 1/b < 2/a and 1/a + 1/b = R), so a < 2/R.
    R_3 = TARGET - Fraction(1, M)
    if R_3 <= 0:
        return None
    
    a_min = int(1 / R_3) + 1
    a_max = min(M - 2, int(2 / R_3))
    
    for a in range(a_min, a_max + 1):
        # R_ab = R_3 - 1/a
        R_ab = R_3 - Fraction(1, a)
        if R_ab <= 0:
            continue
        # 1/b = R_ab => b = 1/R_ab
        # b must be integer
        if R_ab.denominator == 1:
            b = R_ab.numerator
            if b > a and b < M:
                return [a, b, M]
        else:
            # b = 1 / (N/D) = D/N. Must be integer.
            if R_ab.denominator % R_ab.numerator == 0:
                b = R_ab.denominator // R_ab.numerator
                if b > a and b < M:
                    return [a, b, M]

    # 4-term: 1/a + 1/b + 1/c + 1/M = 5/31
    # 1/a + 1/b + 1/c = 5/31 - 1/M = R_4
    R_4 = TARGET - Fraction(1, M)
    if R_4 <= 0:
        return None
        
    a_min = int(1 / R_4) + 1
    a_max = min(M - 3, int(3 / R_4)) # 3/a > R_4 => a < 3/R_4
    
    for a in range(a_min, a_max + 1):
        R_bc = R_4 - Fraction(1, a)
        if R_bc <= 0:
            continue
        
        # 1/b + 1/c = R_bc = N/D
        N = R_bc.numerator
        D = R_bc.denominator
        
        # (Nb - D)(Nc - D) = D^2
        # Let k be a divisor of D^2.
        # Nb - D = k => b = (k + D) / N
        # Nc - D = D^2 / k => c = (D^2/k + D) / N
        
        divs = _get_divisors(D * D)
        
        for k in divs:
            # b = (k + D) / N
            if (k + D) % N != 0:
                continue
            b = (k + D) // N
            
            # c = (D^2/k + D) / N
            if (D * D // k + D) % N != 0:
                continue
            c = (D * D // k + D) // N
            
            # Check distinctness and bounds
            # We want a < b < c < M or any permutation, but we fixed a and M.
            # So we need b, c distinct from a and M, and from each other.
            # And b, c < M.
            if b == a or c == a or b == M or c == M or b == c:
                continue
            if b < 1 or c < 1:
                continue
            if b >= M or c >= M:
                continue
            
            # Verify sum
            if Fraction(1, a) + Fraction(1, b) + Fraction(1, c) + Fraction(1, M) == TARGET:
                return [a, b, c, M]
                
    return None

def construct():
    start_time = time.time()
    deadline = start_time + 50.0 # 50s margin
    
    # Known solution
    known = [8, 31, 248]
    best = known
    best_max = 248
    
    # Check smaller max denominators
    # Lower bound: max denom must be at least 19?
    # 1/19 + 1/20 + 1/21 ~ 0.15. 5/31 ~ 0.161.
    # So M >= 19 is a reasonable lower bound to start checking.
    
    for M in range(247, 18, -1):
        if time.time() > deadline:
            break
        sol = _find_solution_for_max(M)
        if sol is not None:
            best = sol
            best_max = M
            # Since we are iterating M downwards, the first solution found is optimal
            # for that M. We can return immediately as we want global min M.
            return sol
            
    return best
