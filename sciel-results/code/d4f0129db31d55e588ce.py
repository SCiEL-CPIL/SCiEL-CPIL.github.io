import time
import math

# EVOLVE-BLOCK-START

def _lcm(a, b):
    return a * b // math.gcd(a, b)

def construct():
    divisors = [6, 10, 14, 15, 21, 22, 26, 33, 35, 39]
    n = len(divisors)
    
    # Calculate LCM of all divisors
    L = divisors[0]
    for i in range(1, n):
        L = _lcm(L, divisors[i])
        
    start_time = time.time()
    deadline = start_time + 58.0 # 2s margin for return overhead
    
    # Fallback solution: the divisors themselves
    best_span = max(divisors) - min(divisors)
    best_solution = divisors[:]
    
    # We search for minimum W
    # Using bitset masks to filter Y values efficiently
    
    for W in range(1, best_span):
        if time.time() > deadline:
            break
            
        # 1. Compute mask for each divisor
        # mask_i has bit j set if there is a multiple of divisors[i] in [j, j+W]
        masks = [0] * n
        
        for i in range(n):
            d = divisors[i]
            mask = 0
            # Optimization: The condition depends on j % d
            # We can compute the pattern for one period d and repeat it
            # Pattern: for r in 0..d-1, offset = (d - r) % d.
            # If offset <= W, then residue r is "good".
            
            # Build a bitmask for one period of length d
            period_mask = 0
            for r in range(d):
                offset = 0 if r == 0 else d - r
                if offset <= W:
                    period_mask |= (1 << r)
            
            # Repeat this pattern to length L
            # L is a multiple of d
            repeats = L // d
            # We can construct the full mask by shifting and OR-ing
            # Or simpler: loop j from 0 to L-1?
            # L=30030. Looping 30030 * 10 = 300k is fine.
            # But we can be smarter.
            
            # Actually, just looping j is simpler and fast enough in Python if not inside a tight nested loop with backtracking.
            # But here it's just bit setting.
            
            # Let's stick to the period mask repetition for elegance and speed.
            # mask = period_mask | (period_mask << d) | ...
            
            current = period_mask
            shift = d
            while shift < L:
                current |= (period_mask << shift)
                shift += d
            
            masks[i] = current
            
        # 2. Intersect all masks
        common_mask = masks[0]
        for i in range(1, n):
            common_mask &= masks[i]
            
        if common_mask == 0:
            continue
            
        # 3. Iterate through set bits in common_mask
        # Extract Y values
        cm = common_mask
        while cm:
            if time.time() > deadline:
                break
                
            # Get lowest set bit
            lsb = cm & -cm
            y = lsb.bit_length() - 1
            cm ^= lsb
            
            # Check SDR for this Y
            # Build candidates
            candidates = [None] * n
            valid = True
            for i in range(n):
                d = divisors[i]
                rem = y % d
                offset = 0 if rem == 0 else d - rem
                # We know offset <= W because y is in common_mask
                cands = []
                val = y + offset
                while val <= y + W:
                    cands.append(val)
                    val += d
                candidates[i] = cands
                
            # Order by size
            order = sorted(range(n), key=lambda i: len(candidates[i]))
            
            used_values = set()
            path = [None] * n
            
            # Backtracking
            def backtrack(idx):
                if idx == n:
                    return True
                i = order[idx]
                for v in candidates[i]:
                    if v not in used_values:
                        used_values.add(v)
                        path[i] = v
                        if backtrack(idx + 1):
                            return True
                        used_values.remove(v)
                        path[i] = None
                return False
            
            if backtrack(0):
                return [int(x) for x in path]
                
    return [int(x) for x in best_solution]

# EVOLVE-BLOCK-END
