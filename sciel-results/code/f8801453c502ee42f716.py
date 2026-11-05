from math import gcd
import time

def construct():
    start_time = time.time()
    deadline = 55.0

    def dfs(rem_num, rem_den, last, depth, k, path):
        if time.time() - start_time > deadline:
            return None
        
        if depth == k:
            if rem_num == 0:
                return path[:]
            return None

        if rem_num <= 0:
            return None
            
        # d >= ceil(rem_den / rem_num)
        min_d = (rem_den + rem_num - 1) // rem_num
        if min_d <= last:
            min_d = last + 1
            
        # d <= (k - depth) * rem_den / rem_num
        max_d = (k - depth) * rem_den // rem_num
        
        if max_d < min_d:
            return None

        for d in range(min_d, max_d + 1):
            if time.time() - start_time > deadline:
                return None
            
            new_num = rem_num * d - rem_den
            if new_num < 0:
                break
                
            g = gcd(new_num, rem_den * d)
            new_num //= g
            new_den = (rem_den * d) // g
            
            path.append(d)
            res = dfs(new_num, new_den, d, depth + 1, k, path)
            if res:
                return res
            path.pop()
            
        return None

    # k=8 is the minimum
    res = dfs(1, 1, 4, 0, 8, [])
    if res:
        return res
        
    # Fallback if k=8 search fails (should not happen)
    # Try k=9 just in case
    res = dfs(1, 1, 4, 0, 9, [])
    if res:
        return res
        
    return []
