import math
from functools import lru_cache

P = {'n': 12, 'maximum_start': 1000000}

def construct():
    n = P['n']
    LCM = 27720
    best_m = 0
    best_L = -1
    indices = tuple(range(1, n + 1))
    
    def check_match(m, L):
        # Returns True if matching exists
        items = []
        for i in indices:
            start = ((m // i) + 1) * i
            if start > m + L:
                return False
            count = (m + L - start) // i + 1
            items.append((count, i, start))
            
        items.sort()
        
        def dfs(k, used_mask):
            if k == n:
                return True
            count, i, start = items[k]
            for j in range(count):
                v = start + j * i
                bit_idx = v - m - 1
                bit = 1 << bit_idx
                if not (used_mask & bit):
                    if dfs(k + 1, used_mask | bit):
                        return True
            return False
            
        return dfs(0, 0)

    for m in range(LCM):
        # Calculate lower bound L_min
        min_L = 0
        for i in indices:
            fm = ((m // i) + 1) * i
            gap = fm - m
            if gap > min_L:
                min_L = gap
                
        lo = min_L
        hi = 144
        
        if check_match(m, lo):
            ans_L = lo
        else:
            while lo < hi:
                mid = (lo + hi) // 2
                if check_match(m, mid):
                    hi = mid
                else:
                    lo = mid + 1
            ans_L = lo
            
        if ans_L > best_L:
            best_L = ans_L
            best_m = m
            if best_L == 144:
                return int(best_m)
                
    return int(best_m)
