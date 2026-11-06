import time

def construct():
    deadline = time.time() + 80  # safety margin before 90s kill
    
    max_val = 1000001  # n+1 can be up to 1000001
    lpf = [0] * (max_val + 1)
    
    # Sieve for largest prime factor
    for p in range(2, max_val + 1):
        if lpf[p] == 0:  # p is prime
            for m in range(p, max_val + 1, p):
                lpf[m] = p
    
    # Find n in [10000, 1000000] minimizing max(lpf[n], lpf[n+1])
    best_n = 10000
    best_val = max(lpf[10000], lpf[10001])
    
    for n in range(10001, 1000001):
        val = max(lpf[n], lpf[n + 1])
        if val < best_val:
            best_val = val
            best_n = n
        if time.time() > deadline:
            break
    
    return best_n
