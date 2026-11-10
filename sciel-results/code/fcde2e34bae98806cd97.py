import time
from math import gcd
from itertools import product

def construct():
    divisors = [6, 10, 14, 15, 21, 22, 26, 33, 35, 39]
    n = len(divisors)
    
    start_time = time.time()
    deadline = start_time + 85  # Safety margin for 90s limit
    
    # LCM of divisors: 6=2*3, 10=2*5, 14=2*7, 15=3*5, 21=3*7, 22=2*11, 26=2*13, 33=3*11, 35=5*7, 39=3*13
    # LCM = 2 * 3 * 5 * 7 * 11 * 13 = 30030
    LCM = 30030
    
    # Precompute pairwise LCMs for conflict checking
    pair_lcm = [[1] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            pair_lcm[i][j] = divisors[i] * divisors[j] // gcd(divisors[i], divisors[j])
    
    # For each W, try to find a valid window [L, L+W] containing distinct multiples
    # Key insight: for a fixed L, the valid offsets for divisor d are:
    #   {r in [0, W] : r ≡ (-L) mod d}
    # i.e., r = r0 + k*d where r0 = (-L) % d
    
    # Strategy: iterate W from 1 upward. For each W, iterate L over a reduced set.
    # Since the pattern repeats every LCM, we only need L in [0, LCM-1].
    # But we can be smarter: for small W, each divisor has very few choices.
    
    # For W < 6: only divisor 6 can have 2 choices, all others have exactly 1.
    # For W < 10: divisors 6 and 10 can have 2 choices.
    # etc.
    
    # Optimized backtracking with forward checking
    
    def solve_for_W(W):
        """Try to find a valid window of size W. Returns list of values or None."""
        
        # For each L in [0, LCM), compute offset sets and check feasibility
        # Optimization: we can skip many L values using modular constraints.
        
        # Actually, let's just iterate L and use fast backtracking.
        # The key speedup: precompute r0 for each divisor for each L.
        # r0_i(L) = (-L) % divisors[i]
        
        # For a given L, the offsets for divisor i are:
        #   r0_i, r0_i + d_i, r0_i + 2*d_i, ... <= W
        # The number of offsets is floor((W - r0_i) / d_i) + 1 if r0_i <= W, else 0.
        
        # If any divisor has 0 offsets, L is infeasible.
        # If all divisors have exactly 1 offset, we just need all offsets to be distinct.
        
        # Let's iterate L from 0 to LCM-1, but with early termination.
        
        for L in range(LCM):
            if time.time() > deadline:
                return None
            
            # Compute offsets for each divisor
            offset_sets = []
            feasible = True
            
            for i, d in enumerate(divisors):
                r0 = (-L) % d
                if r0 > W:
                    feasible = False
                    break
                # Number of offsets: floor((W - r0) / d) + 1
                count = (W - r0) // d + 1
                offset_sets.append((r0, d, count))
            
            if not feasible:
                continue
            
            # Check if all counts are 1: just verify all r0 values are distinct
            if all(count == 1 for _, _, count in offset_sets):
                r0s = [r0 for r0, _, _ in offset_sets]
                if len(set(r0s)) == n:
                    result = [L + r0 for r0, _, _ in offset_sets]
                    return result
            
            # Otherwise, use backtracking
            # Sort by count (MRV heuristic)
            indices = list(range(n))
            indices.sort(key=lambda i: offset_sets[i][2])
            
            used = [False] * (W + 1)
            choices = [0] * n
            
            def backtrack(idx):
                if idx == n:
                    return True
                
                i = indices[idx]
                r0, d, count = offset_sets[i]
                
                for k in range(count):
                    r = r0 + k * d
                    if not used[r]:
                        used[r] = True
                        choices[i] = r
                        if backtrack(idx + 1):
                            return True
                        choices[i] = 0
                        used[r] = False
                
                return False
            
            if backtrack(0):
                result = [L + choices[i] for i in range(n)]
                return result
        
        return None
    
    # Try increasing W
    for W in range(1, 200):
        if time.time() > deadline:
            break
        
        result = solve_for_W(W)
        if result is not None:
            # Verify
            assert len(set(result)) == n, f"Result not distinct: {result}"
            assert all(result[i] % divisors[i] == 0 for i in range(n)), f"Divisibility check failed: {result}"
            assert all(0 < x <= 1000000 for x in result), f"Range check failed: {result}"
            assert max(result) - min(result) == W, f"Span check failed: {max(result)-min(result)} != {W}"
            
            return [int(x) for x in result]
    
    # Fallback: return the divisors themselves
    return [int(d) for d in divisors]
