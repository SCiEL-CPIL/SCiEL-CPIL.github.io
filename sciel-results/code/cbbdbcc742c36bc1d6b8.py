import time
import sys
from math import gcd

def construct():
    sys.setrecursionlimit(10000)
    start_time = time.time()
    deadline = start_time + 85.0
    
    target_num, target_den = 5, 31
    
    for D_max in range(200, 2000):
        if time.time() > deadline:
            break
            
        H = [0.0] * (D_max + 2)
        for i in range(D_max, 0, -1):
            H[i] = H[i+1] + 1.0/i
            
        def dfs(rem_num, rem_den, last_a):
            if rem_num == 0:
                return []
            if rem_num < 0:
                return None
            
            if rem_num * D_max < rem_den:
                return None
                
            max_possible_sum = H[D_max] - H[last_a]
            if rem_num * max_possible_sum < rem_den * 0.999999999:
                return None
                
            min_a = max(last_a + 1, rem_den // rem_num + (1 if rem_den % rem_num != 0 else 0))
            max_a = D_max
            
            if min_a > max_a:
                return None
                
            for a in range(min_a, max_a + 1):
                new_num = rem_num * a - rem_den
                if new_num < 0:
                    continue
                    
                new_den = rem_den * a
                
                res = dfs(new_num, new_den, a)
                if res is not None:
                    return [a] + res
            return None

        res = dfs(target_num, target_den, 0)
        if res is not None:
            return res

    return [7, 8, 9, 10000000]
