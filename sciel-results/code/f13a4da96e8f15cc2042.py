import time
from math import gcd

TARGET_NUM = 5
TARGET_DEN = 31
MAX_DENOM = 10_000_000
MAX_TERMS = 256
DEADLINE = 85.0

def construct():
    start_time = time.time()
    deadline = start_time + DEADLINE
    
    def check_time():
        return time.time() > deadline

    best_solution = [8, 31, 248]
    best_max = 248

    def search(rem_num, rem_den, used, min_d, depth, limit):
        if check_time():
            return None
        if rem_num == 0:
            return []
        if depth >= MAX_TERMS:
            return None
        if rem_num < 0:
            return None
        
        g = gcd(rem_num, rem_den)
        rem_num //= g
        rem_den //= g
        
        min_d_eff = max(min_d, rem_den // rem_num + (1 if rem_den % rem_num != 0 else 0))
        
        if min_d_eff >= limit:
            return None
            
        terms_left = MAX_TERMS - depth
        if rem_num * min_d_eff > rem_den * terms_left:
            return None

        for d in range(min_d_eff, limit):
            if d in used:
                continue
            if check_time():
                return None
            
            new_num = rem_num * d - rem_den
            new_den = rem_den * d
            
            if new_num < 0:
                continue # Should not happen due to min_d_eff, but safe
                
            if new_num == 0:
                return [d]
                
            res = search(new_num, new_den, used | {d}, d + 1, depth + 1, limit)
            if res is not None:
                return [d] + res
                
        return None

    current_limit = 248
    while current_limit > 7:
        if check_time():
            break
            
        res = search(TARGET_NUM, TARGET_DEN, set(), 7, 0, current_limit)
        if res is not None:
            cand_max = max(res)
            if cand_max < best_max:
                best_max = cand_max
                best_solution = res
            current_limit = best_max
        else:
            current_limit -= 1
            
    return best_solution
