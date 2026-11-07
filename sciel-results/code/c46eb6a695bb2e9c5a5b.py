import time
from math import gcd
from functools import reduce

def construct():
    divisors = [6, 10, 14, 15, 21, 22, 26, 33, 35, 39]
    n = len(divisors)
    deadline = time.time() + 85  # 85s safety margin (90s limit)
    
    # Compute LCM of all divisors
    def lcm(a, b):
        return a * b // gcd(a, b)
    
    L = reduce(lcm, divisors)  # LCM = 30030
    
    # For a given window [s, s+R], we need to assign distinct multiples of each divisor
    # within that window. The key insight is that the problem is periodic with period L.
    
    # Strategy: For each R from 1 upward, check if there exists a window [s, s+R] (mod L)
    # where we can assign distinct multiples. We use backtracking with MRV heuristic.
    
    best_range = float('inf')
    best_assignment = None
    
    def check_window(s, R):
        """Check if we can assign distinct multiples in [s, s+R] to each divisor.
        Returns the assignment or None."""
        if time.time() > deadline - 2:
            return None
        
        # Compute valid multiples for each divisor in [s, s+R]
        valid = []
        for d in divisors:
            # First multiple of d >= s (and >= 1)
            if s <= 0:
                first = d  # Smallest positive multiple
            else:
                first = ((s + d - 1) // d) * d
                if first < 1:
                    first = d
            
            mults = []
            m = first
            while m <= s + R:
                if m >= 1:
                    mults.append(m)
                m += d
            valid.append(mults)
        
        # Check if any divisor has no valid multiples
        for v in valid:
            if not v:
                return None
        
        # Backtracking with MRV
        used = set()
        assignment = [None] * n
        
        def backtrack():
            if time.time() > deadline - 1:
                return False
            
            # MRV: pick the unassigned divisor with fewest available options
            min_count = float('inf')
            min_idx = -1
            for i in range(n):
                if assignment[i] is None:
                    avail = [m for m in valid[i] if m not in used]
                    if len(avail) < min_count:
                        min_count = len(avail)
                        min_idx = i
                    if min_count == 0:
                        break
            
            if min_idx == -1:
                return True  # All assigned
            if min_count == 0:
                return False
            
            # Try values in order of least constraining value heuristic (just use natural order)
            for m in valid[min_idx]:
                if m not in used:
                    used.add(m)
                    assignment[min_idx] = m
                    if backtrack():
                        return True
                    used.remove(m)
                    assignment[min_idx] = None
            return False
        
        if backtrack():
            return assignment[:]
        return None
    
    # Search: try R from 1 to 500, for each R try s from 0 to L-1
    # But we can be smarter: only try s values that are "interesting"
    # For a given R, the window [s, s+R] contains valid multiples for d_i iff
    # there exists an integer k such that s <= k*d_i <= s+R.
    
    # Quick check: does each divisor have at least one multiple in [s, s+R]?
    def has_multiple(s, R, d):
        if s <= 0:
            # Window [s, s+R] includes 0 or starts before 1
            # We need positive multiples, so check [max(1,s), s+R]
            low = max(1, s)
            if low > s + R:
                return False
            first = ((low + d - 1) // d) * d
            return first <= s + R
        else:
            first = ((s + d - 1) // d) * d
            return first <= s + R
    
    for R in range(1, 500):
        if time.time() > deadline - 5:
            break
        
        found = False
        for s in range(0, L):
            if time.time() > deadline - 3:
                break
            
            # Quick check: does each divisor have at least one multiple in [s, s+R]?
            valid = True
            for d in divisors:
                if not has_multiple(s, R, d):
                    valid = False
                    break
            
            if not valid:
                continue
            
            result = check_window(s, R)
            if result is not None:
                # Verify
                if len(set(result)) == n and all(isinstance(x, int) and x > 0 and x <= 1000000 for x in result):
                    r = max(result) - min(result)
                    if r < best_range:
                        best_range = r
                        best_assignment = result[:]
                        if R == 1:
                            return [int(x) for x in best_assignment]
                        found = True
                        break  # Found a valid window for this R
        
        if found:
            break
    
    if best_assignment is None:
        # Fallback: use the divisors themselves
        return [int(d) for d in divisors]
    
    return [int(x) for x in best_assignment]
