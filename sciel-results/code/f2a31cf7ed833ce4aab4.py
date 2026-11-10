import time
from fractions import Fraction

def construct():
    start_time = time.time()
    deadline = start_time + 50.0  # Leave 10s margin
    
    target = Fraction(5, 31)
    
    # We will search for increasing M
    # A solution with small M is highly likely to exist.
    # Let's try M from 10 to 10000
    # To speed up, we can jump M if we are confident no solution exists for small M?
    # No, just iterate. DFS with pruning is fast for small M.
    
    # Precompute Harmonic sums H[d] = sum(1/i for i in 1..d)
    # We need H[d] for d up to some max_M.
    # Let's estimate max_M we might need.
    # If M=100 works, we are done.
    # If not, maybe 1000.
    # Let's precompute up to 10000 initially.
    max_precomp = 10000
    H = [Fraction(0, 1)] * (max_precomp + 1)
    for i in range(1, max_precomp + 1):
        H[i] = H[i-1] + Fraction(1, i)
        
    def dfs(rem, min_d, max_d, path):
        if time.time() > deadline:
            return None
            
        if rem == 0:
            if path:
                return list(path)
            return None
            
        if rem < 0:
            return None
            
        # Pruning: Can we form 'rem' with denominators > min_d and <= max_d?
        # Max possible sum is H[max_d] - H[min_d-1]
        if min_d - 1 > max_precomp:
             # Fallback if min_d is huge
             pass
        else:
             max_sum = H[max_d] - H[min_d-1]
             if rem > max_sum:
                 return None

        # Min denominator to pick
        # 1/d <= rem  => d >= 1/rem
        # Also d >= min_d
        d_min = max(min_d, (rem.denominator + rem.numerator - 1) // rem.numerator)
        
        # Max denominator to pick
        # We must leave enough sum for subsequent terms?
        # Actually, if we pick d, rem' = rem - 1/d.
        # rem' must be formable by denominators > d.
        # So rem' <= H[max_d] - H[d]
        # rem - 1/d <= H[max_d] - H[d]
        # 1/d >= rem - (H[max_d] - H[d])
        
        # Iterate d from d_min to max_d
        # To make it faster, we can check the condition above to find d_max_effective
        # But simple iteration with pruning inside recursion is often fine for small depths.
        
        for d in range(d_min, max_d + 1):
            # Check if rem - 1/d is formable by denominators > d
            next_rem = rem - Fraction(1, d)
            if next_rem < 0:
                continue
                
            # Check upper bound on next_rem
            if d < max_precomp:
                max_next_sum = H[max_d] - H[d]
                if next_rem > max_next_sum:
                    continue
            else:
                # If d is very large, the sum of remaining reciprocals is small
                # Approximate or just let recursion handle it (it will fail quickly)
                pass

            path.append(d)
            res = dfs(next_rem, d + 1, max_d, path)
            if res is not None:
                return res
            path.pop()
            
        return None

    # Search for M
    # Start with small M, increase
    # If we find a solution, return it.
    # We want the SMALLEST M to maximize score.
    
    # Heuristic: Try a few specific values or range
    # Let's try M from 10 to 500 first.
    for M in range(10, 501):
        if time.time() > deadline:
            break
        # Ensure max_d in dfs is M
        # Note: dfs uses global H and max_precomp
        res = dfs(target, 1, M, [])
        if res is not None:
            # Found solution with max denominator M
            # Verify distinctness and sum
            # The DFS ensures distinctness (min_d increases) and sum (rem==0)
            # Check if all d <= 10^7 (M is small, so yes)
            return res
            
    # If no solution found in 10..500, try larger ranges or fallback
    # Fallback: Greedy solution from seed
    r, values = Fraction(5, 31) - Fraction(1, 8), [8]
    while r:
        d = (r.denominator + r.numerator - 1) // r.numerator
        values.append(d)
        r -= Fraction(1, d)
    return values
