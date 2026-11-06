"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
from fractions import Fraction
import time

def construct():
    start_time = time.time()
    deadline = start_time + 50  # 50s safety margin to ensure < 60s total
    
    # 1. Greedy to find a baseline solution
    def greedy_solve():
        R = Fraction(1, 1)
        d_prev = 4
        terms = []
        while R > 0:
            # d must be > d_prev and d >= 5
            # d must satisfy 1/d <= R  => d >= ceil(1/R)
            # ceil(1/R) = ceil(D/N) = (D + N - 1) // N
            d_min = max(d_prev + 1, 5, (R.denominator + R.numerator - 1) // R.numerator)
            
            # Pick d_min
            d = d_min
            terms.append(d)
            R -= Fraction(1, d)
            d_prev = d
            
            if len(terms) > 256:
                return None
            if R < 0:
                return None
                
        return terms

    best_terms = greedy_solve()
    if not best_terms:
        # Fallback
        raise RuntimeError("Greedy failed")
        
    best_count = len(best_terms)
    
    # 2. DFS to improve
    # We try to find a solution with count < best_count
    
    def max_sum_frac(d, k):
        """Calculate sum of 1/(d+i) for i=0..k-1 as Fraction"""
        s = Fraction(0, 1)
        for i in range(k):
            s += Fraction(1, d + i)
        return s

    def dfs(R, d_prev, k, terms):
        nonlocal best_terms, best_count, deadline
        
        if time.time() > deadline:
            return False
            
        if k == 0:
            if R == 0:
                return True
            return False
            
        if R == 0:
            # Cannot add more terms
            return False

        if R < 0:
            return False

        # Pruning: Max sum
        d_start_min = max(d_prev + 1, 5)
        max_possible = max_sum_frac(d_start_min, k)
        if R > max_possible:
            return False
            
        # Determine range for d
        # d >= d_start_min
        # d >= ceil(1/R)
        d_lower = max(d_start_min, (R.denominator + R.numerator - 1) // R.numerator)
        
        # Upper bound check
        # Iterate d from d_lower
        # Cap d at 10^7
        for d in range(d_lower, 10000000):
            if time.time() > deadline:
                return False
                
            if k > 1:
                rem = R - Fraction(1, d)
                if rem < 0:
                    # Should not happen if d >= ceil(1/R)
                    break
                max_rem = max_sum_frac(d + 1, k - 1)
                if rem > max_rem:
                    break
            else:
                # k == 1
                if R == Fraction(1, d):
                    new_terms = terms + [d]
                    if len(new_terms) < best_count:
                        best_terms = new_terms
                        best_count = len(new_terms)
                    return True
                else:
                    # If R != 1/d, no solution for this k=1 branch
                    # Since d is increasing, 1/d is decreasing.
                    # If 1/d < R, then for larger d, 1/d < R still.
                    # So no solution.
                    return False

            terms.append(d)
            if dfs(R - Fraction(1, d), d, k - 1, terms):
                return True
            terms.pop()
            
        return False

    # Try to improve by searching for smaller counts
    for target_count in range(best_count - 1, 4, -1):
        if time.time() > deadline:
            break
        # We only need to find ONE solution with this count to update best
        # If found, we continue to smaller counts
        if dfs(Fraction(1, 1), 4, target_count, []):
            continue # Found one, try smaller
        
    return best_terms
# EVOLVE-BLOCK-END
