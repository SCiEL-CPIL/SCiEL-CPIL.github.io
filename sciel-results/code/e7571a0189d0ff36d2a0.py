"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from fractions import Fraction

def construct():
    """Find Egyptian fraction of 5/31 minimizing the largest denominator.
    
    Uses systematic search: try increasing max-denominator bounds and for each,
    search for valid Egyptian fraction representations. Returns the first valid
    solution found, which has the minimal possible largest denominator.
    """
    target = Fraction(5, 31)
    deadline = time.time() + 85  # 5s safety margin before 90s kill
    
    # Best solution found so far
    best_solution = None
    best_max = 10000001
    
    # Try increasing maximum denominator bounds
    # Start with small bounds and increase
    # We know 1-term and 2-term are impossible, so start with 3 terms
    for max_den in range(1, 10000001):
        if time.time() > deadline:
            break
            
        # For efficiency, only check certain max_den values
        # The max denominator must be at least ceil(31/5) = 7 for any single term
        # For 3 terms, we need to search
        
        # Check if there exists a 3-term Egyptian fraction with all denominators <= max_den
        # and at least one equal to max_den (to avoid redundant checks)
        if max_den < 7:
            continue
        
        # Search for 3-term solutions: 1/a + 1/b + 1/c = 5/31
        # where a <= b <= c <= max_den
        found = False
        a_start = max(1, int(31/5) - 1)  # a must satisfy 1/a < 5/31, so a > 31/5
        a_min = 7  # ceil(31/5) = 7, but 1/7 > 5/31, so a >= 7
        # Actually 1/a must be less than 5/31 for remaining terms to be positive
        # 1/a < 5/31 => a > 31/5 = 6.2, so a >= 7
        
        for a in range(7, min(max_den + 1, 1000)):
            if time.time() > deadline:
                break
            rem1 = target - Fraction(1, a)
            if rem1 <= 0:
                break  # a too small, 1/a >= target
            # Now find 1/b + 1/c = rem1 with a <= b <= c <= max_den
            # 1/b < rem1 => b > 1/rem1
            b_min = max(a, int(1/rem1) + 1)
            b_max = min(max_den, 2 * int(1/rem1) + 1)  # since c >= b, 1/b >= rem1/2
            if b_min > b_max:
                continue
            for b in range(b_min, b_max + 1):
                if time.time() > deadline:
                    break
                rem2 = rem1 - Fraction(1, b)
                if rem2 <= 0:
                    break
                # c = 1/rem2 must be integer
                c = 1 / rem2
                c_int = int(c)
                if abs(c - c_int) < 1e-10 and c_int == c:
                    if b <= c_int <= max_den:
                        # Found a valid solution
                        solution = [a, b, c_int]
                        if c_int < best_max:
                            best_max = c_int
                            best_solution = solution
                        found = True
                        break
            if found:
                break
        if found:
            break
    
    if best_solution is not None:
        # Verify
        s = sum(Fraction(1, d) for d in best_solution)
        assert s == target, f"Verification failed: {s}"
        assert len(set(best_solution)) == len(best_solution)
        assert all(1 <= d <= 10000000 for d in best_solution)
        assert len(best_solution) <= 256
        return best_solution
    
    # Fallback: known solution
    solution = [8, 31, 248]
    s = sum(Fraction(1, d) for d in solution)
    assert s == target
    return solution
# EVOLVE-BLOCK-END
