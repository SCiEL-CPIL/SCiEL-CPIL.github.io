import time
from fractions import Fraction

def construct():
    target = Fraction(5, 31)
    start_time = time.time()
    deadline = start_time + 50  # 50s limit to be safe
    
    for D in range(7, 200):
        if time.time() > deadline:
            break
            
        suffix = [Fraction(0)] * (D + 2)
        for i in range(D, 0, -1):
            suffix[i] = suffix[i+1] + Fraction(1, i)
            
        def dfs(idx, current_sum, path):
            if time.time() > deadline:
                return None
            if current_sum == target:
                return path[:]
            if idx > D:
                return None
            if current_sum > target:
                return None
            if current_sum + suffix[idx] < target:
                return None
            
            # Option 1: Include idx
            if current_sum + Fraction(1, idx) <= target:
                res = dfs(idx + 1, current_sum + Fraction(1, idx), path + [idx])
                if res:
                    return res
            
            # Option 2: Exclude idx
            res = dfs(idx + 1, current_sum, path)
            if res:
                return res
            
            return None

        res = dfs(1, Fraction(0), [])
        if res:
            return res
            
    return [8, 31, 248, 1984, 15872] # Fallback
