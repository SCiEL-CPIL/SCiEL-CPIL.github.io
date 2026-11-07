import time
import math

def construct():
    divisors = [6, 10, 14, 15, 21, 22, 26, 33, 35, 39]
    n = len(divisors)
    
    # Compute LCM
    def gcd(a, b):
        while b:
            a, b = b, a % b
        return a
    
    def lcm(a, b):
        return abs(a * b) // gcd(a, b)
    
    L = 1
    for d in divisors:
        L = lcm(L, d)
    # L should be 30030
    
    start_time = time.time()
    deadline = 85.0  # Safety margin for 90s limit
    
    best_range = float('inf')
    best_solution = None
    
    # For each residue r in [0, L), find best assignment
    # Optimization: only consider r where the first multiples are "close"
    
    for r in range(L):
        if time.time() - start_time > deadline:
            break
        
        # For each divisor, find the first few multiples >= r (and > 0)
        # We'll look at multiples in [r, r + L) since the pattern repeats
        candidates_by_div = []
        feasible = True
        
        for d in divisors:
            # First multiple of d that is >= r and > 0
            if r == 0:
                first = d  # smallest positive multiple
            else:
                # ceil(r/d) * d
                k = (r + d - 1) // d
                if k == 0:
                    k = 1
                first = k * d
                if first <= 0:
                    first = d
            
            # Generate multiples: first, first+d, first+2d, ... up to first + (L//d)*d
            # But we only need a few
            mults = []
            for j in range(15):  # 15 candidates should be enough
                val = first + j * d
                if val > r + L:
                    break
                mults.append(val)
            
            if not mults:
                feasible = False
                break
            
            candidates_by_div.append(mults)
        
        if not feasible:
            continue
        
        # Now find assignment minimizing max-min
        # Use backtracking with pruning
        # Order divisors by number of candidates (fewest first) for efficiency
        order = sorted(range(n), key=lambda i: len(candidates_by_div[i]))
        
        chosen = [None] * n
        used = set()
        
        def backtrack(idx, cur_min, cur_max):
            nonlocal best_range, best_solution
            if idx == n:
                rng = cur_max - cur_min
                if rng < best_range:
                    best_range = rng
                    best_solution = chosen[:]
                return
            
            if cur_max - cur_min >= best_range:
                return
            
            di = order[idx]
            for val in candidates_by_div[di]:
                if val in used:
                    continue
                
                new_min = min(cur_min, val) if cur_min != float('inf') else val
                new_max = max(cur_max, val) if cur_max != -float('inf') else val
                
                if new_max - new_min >= best_range:
                    continue
                
                chosen[di] = val
                used.add(val)
                backtrack(idx + 1, new_min, new_max)
                used.remove(val)
                chosen[di] = None
        
        backtrack(0, float('inf'), -float('inf'))
        
        if best_range <= 1:
            break
    
    if best_solution is None:
        # Fallback
        return divisors[:]
    
    return [int(x) for x in best_solution]
