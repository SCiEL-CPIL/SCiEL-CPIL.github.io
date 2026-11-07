import time
import math
import bisect

# EVOLVE-BLOCK-START

def construct():
    start_time = time.time()
    DEADLINE = 85  # safety margin before 90s kill
    
    MAX_N = 10_000_000
    
    # Fast sigma computation using smallest prime factor sieve
    spf = list(range(MAX_N + 1))
    for i in range(2, int(math.isqrt(MAX_N)) + 1):
        if spf[i] == i:
            for j in range(i * i, MAX_N + 1, i):
                if spf[j] == j:
                    spf[j] = i
    
    sigma = [0] * (MAX_N + 1)
    sigma[1] = 1
    for n in range(2, MAX_N + 1):
        p = spf[n]
        m = n // p
        power = p
        while m > 1 and m % p == 0:
            power *= p
            m //= p
        factor = (power * p - 1) // (p - 1)
        if m == 1:
            sigma[n] = factor
        else:
            sigma[n] = sigma[m] * factor
            
    if time.time() - start_time > DEADLINE:
        return [11, 6]

    # Group indices by sigma value
    groups = {}
    for i in range(1, MAX_N + 1):
        s = sigma[i]
        if s in groups:
            groups[s].append(i)
        else:
            groups[s] = [i]
            
    if time.time() - start_time > DEADLINE:
        return [11, 6]

    best_error = float('inf')
    best_a = None
    best_b = None
    
    # Process each group
    # We want to minimize |a/b - 1.5|
    # a and b must be in the same group.
    
    for s, lst in groups.items():
        if time.time() - start_time > DEADLINE:
            break
            
        if len(lst) < 2:
            continue
            
        # lst is already sorted because we appended in increasing order of i
        
        # We can iterate through b in lst and find best a
        # To optimize, we can stop early if the minimum possible error for remaining b is > best_error
        # Min error for b is roughly 1/(2b) if there is an a close to 1.5b.
        # But we don't know if such a exists.
        
        # Let's just iterate.
        # To speed up, we can sort lst by value? It is already sorted.
        
        # We can use two pointers?
        # We want a/b ~ 1.5 => a ~ 1.5 b.
        # Since lst is sorted, we can maintain a pointer for a.
        # For each b, we want a close to 1.5 b.
        # As b increases, 1.5 b increases. So a should increase.
        
        ptr_a = 0
        n_lst = len(lst)
        
        for b in lst:
            # Move ptr_a to be close to 1.5 * b
            target_a = 1.5 * b
            while ptr_a < n_lst and lst[ptr_a] < target_a:
                ptr_a += 1
                
            # Check ptr_a and ptr_a - 1
            for idx in [ptr_a, ptr_a - 1]:
                if 0 <= idx < n_lst:
                    a = lst[idx]
                    if a == b:
                        continue
                    err_num = abs(2*a - 3*b)
                    if err_num == 0:
                        continue
                    err = err_num / (2.0 * b)
                    if err < best_error:
                        best_error = err
                        best_a = a
                        best_b = b
                        
    if best_a is None:
        return [11, 6]
        
    return [int(best_a), int(best_b)]

# EVOLVE-BLOCK-END
