import numpy as np
import time

def _is_prime(x):
    if x < 2:
        return False
    if x < 4:
        return True
    if x % 2 == 0:
        return False
    for i in range(3, int(x**0.5) + 1, 2):
        if x % i == 0:
            return False
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 80
    
    primes = [x for x in range(1, 513) if _is_prime(x)]
    others = [x for x in range(1, 513) if not _is_prime(x)]
    candidates = primes + others
    
    selected = []
    n_sel = 0
    max_sum = 0
    dp = [1]
    
    for cand in candidates:
        if len(selected) >= 48:
            break
        if time.time() > deadline:
            break
        
        valid = True
        for k in range(2, n_sel + 1):
            target = k * cand
            if target > max_sum:
                break
            if (dp[k] >> target) & 1:
                valid = False
                break
        
        if valid:
            for a in selected:
                for k in range(1, n_sel + 1):
                    target = (k + 1) * a - cand
                    if target < 0 or target > max_sum:
                        continue
                    if (dp[k] >> target) & 1:
                        valid = False
                        break
                if not valid:
                    break
        
        if valid:
            selected.append(cand)
            n_sel += 1
            max_sum += cand
            dp.append(0)
            for k in range(n_sel, 0, -1):
                dp[k] |= (dp[k-1] << cand)
    
    return selected

