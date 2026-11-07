import math
from itertools import product

def construct():
    divisors = [6, 10, 14, 15, 21, 22, 26, 33, 35, 39]
    n = len(divisors)
    
    # Compute LCM
    def lcm(a, b):
        return a * b // math.gcd(a, b)
    
    L = divisors[0]
    for d in divisors[1:]:
        L = lcm(L, d)
    
    best_range = float('inf')
    best_vals = None
    
    # Search over base points in one LCM period
    for b in range(1, L + 1):
        # For each divisor, get the two closest multiples to b
        candidates = []
        for d in divisors:
            k = b // d
            c1 = k * d
            c2 = (k + 1) * d
            # Ensure positive
            if c1 < 1:
                c1 = d
            candidates.append((c1, c2))
        
        # Try all 2^n combinations
        for combo in product([0, 1], repeat=n):
            vals = [candidates[i][combo[i]] for i in range(n)]
            # Check distinct
            if len(set(vals)) != n:
                continue
            # Check all positive and <= 1000000
            if any(v < 1 or v > 1000000 for v in vals):
                continue
            r = max(vals) - min(vals)
            if r < best_range:
                best_range = r
                best_vals = vals[:]
    
    # Return as list of ints
    return [int(v) for v in best_vals]
