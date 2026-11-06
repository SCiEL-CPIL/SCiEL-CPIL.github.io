import time

def construct():
    t0 = time.time()
    deadline = t0 + 85.0
    
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23]
    
    MAX_IDX = 1000024
    MAX_N = 1000000
    MIN_N = 1000
    
    smooth = [1] * (MAX_IDX + 1)
    smooth_local = smooth
    
    for p in primes:
        for mult in range(p, MAX_IDX + 1, p):
            v = mult
            power = 1
            while v % p == 0:
                power *= p
                v //= p
            smooth_local[mult] *= power
    
    wc = {}
    for i in range(MIN_N + 1, MIN_N + 25):
        val = smooth_local[i]
        wc[val] = wc.get(val, 0) + 1
        
    min_distinct = len(wc)
    best_n = MIN_N
    
    if min_distinct == 1:
        return best_n
    
    n = MIN_N + 1
    while n <= MAX_N:
        if (n - MIN_N) % 1000 == 0 and time.time() > deadline:
            break
            
        val_out = smooth_local[n]
        cnt = wc[val_out] - 1
        if cnt == 0:
            del wc[val_out]
        else:
            wc[val_out] = cnt
            
        val_in = smooth_local[n + 24]
        wc[val_in] = wc.get(val_in, 0) + 1
        
        current_distinct = len(wc)
        if current_distinct < min_distinct:
            min_distinct = current_distinct
            best_n = n
            if min_distinct == 1:
                break
        
        n += 1
        
    return best_n
