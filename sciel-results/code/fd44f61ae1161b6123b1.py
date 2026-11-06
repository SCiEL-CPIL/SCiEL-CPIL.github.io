import time

def construct():
    start_time = time.time()
    deadline = start_time + 50  # Safety margin before 60s limit
    
    MAX_N = 1000001
    
    # 1. Compute Smallest Prime Factor (SPF)
    spf = list(range(MAX_N + 1))
    for i in range(2, int(MAX_N**0.5) + 1):
        if spf[i] == i:
            for j in range(i*i, MAX_N + 1, i):
                if spf[j] == j:
                    spf[j] = i
    
    # 2. Compute Powerful Part (pp) using DP
    pp = [1] * (MAX_N + 1)
    # pp[1] is 1 by default.
    # We need to handle primes correctly.
    # If i is prime, spf[i] == i.
    # The recurrence:
    # p = spf[i]
    # j = i // p
    # if j % p != 0: pp[i] = pp[j]
    # else:
    #   k = j // p
    #   if k % p != 0: pp[i] = pp[j] * p * p
    #   else: pp[i] = pp[j] * p
    
    # Let's trace for a prime i.
    # p = i. j = 1.
    # 1 % i != 0 (since i >= 2).
    # So pp[i] = pp[1] = 1. Correct.
    
    for i in range(2, MAX_N + 1):
        p = spf[i]
        j = i // p
        if j % p != 0:
            pp[i] = pp[j]
        else:
            k = j // p
            if k % p != 0:
                pp[i] = pp[j] * p * p
            else:
                pp[i] = pp[j] * p
        
        # Time check
        if i % 100000 == 0 and time.time() > deadline:
            break

    # 3. Find max product
    best_n = 10000
    best_val = pp[10000] * pp[10001]
    
    for n in range(10000, MAX_N):
        val = pp[n] * pp[n+1]
        if val > best_val:
            best_val = val
            best_n = n
            
        if time.time() > deadline:
            break

    return best_n
