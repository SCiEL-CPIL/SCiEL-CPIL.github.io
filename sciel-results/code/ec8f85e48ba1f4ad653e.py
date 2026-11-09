import time
from fractions import Fraction

def construct():
    target = Fraction(5, 31)
    # Fallback valid solution: 1/8 + 1/31 + 1/248 = 5/31
    fallback = [8, 31, 248]
    
    start_time = time.time()
    
    # We expect the optimal M to be small.
    # 248 is a known solution.
    # Let's search up to 248 first.
    for M in range(7, 249):
        if time.time() - start_time > 45:
            return fallback
            
        # Precompute suffix sums of reciprocals for 1..M
        # S[k] = sum(1/j for j in range(k, M+1))
        S = [0] * (M + 2)
        for k in range(M, 0, -1):
            S[k] = S[k+1] + Fraction(1, k)
        
        # DFS
        def dfs(last_d, rem, path):
            if rem == 0:
                return True
            if rem < 0:
                return False
            
            # Min next denominator
            # 1/d <= rem => d >= 1/rem
            # d must be > last_d
            d_min = max(last_d + 1, (rem.denominator + rem.numerator - 1) // rem.numerator)
            
            if d_min > M:
                return False
            
            # Pruning:
            # Max sum we can add with denominators >= d_min is S[d_min]
            # If rem > S[d_min], we can't reach rem.
            if rem > S[d_min]:
                return False
            
            for d in range(d_min, M + 1):
                # 1/d must be <= rem
                # Since d >= d_min >= 1/rem, this is always true.
                
                # Check if remaining sum can be achieved
                new_rem = rem - Fraction(1, d)
                
                # The next denominators must be > d.
                # Max sum from d+1..M is S[d+1]
                if new_rem > S[d+1]:
                    # If new_rem is too large, we need a larger 1/d (smaller d)
                    # But we are increasing d, so 1/d is getting smaller.
                    # So new_rem is getting larger.
                    # And S[d+1] is getting smaller.
                    # So if it fails here, it will fail for all larger d.
                    break
                
                path.append(d)
                if dfs(d, new_rem, path):
                    return True
                path.pop()
                
            return False

        path = []
        if dfs(0, target, path):
            return path

    # If we didn't find anything up to 248, or timed out, return fallback
    # Note: If we timed out in the middle, we return fallback.
    # If we finished the loop without finding, we return fallback.
    return fallback
