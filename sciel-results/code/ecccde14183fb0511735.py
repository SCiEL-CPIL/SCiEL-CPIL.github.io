import time

def construct():
    N = 123457
    start_time = time.time()
    deadline = 60.0 # Safety margin
    
    # 1. Sieve for Smallest Prime Factor
    spf = list(range(N + 1))
    for i in range(2, int(N**0.5) + 1):
        if spf[i] == i: # i is prime
            for j in range(i*i, N + 1, i):
                if spf[j] == j:
                    spf[j] = i
    
    # 2. Compute Largest Prime Factor for each n
    lpf = [1] * (N + 1)
    for i in range(2, N + 1):
        n = i
        max_p = 1
        while n > 1:
            p = spf[n]
            if p > max_p:
                max_p = p
            while n % p == 0:
                n //= p
        lpf[i] = max_p
        
    # 3. Find optimal pair
    best_val = float('inf')
    best_pair = [1, N-1]
    
    for a in range(1, N // 2 + 1):
        b = N - a
        val = max(lpf[a], lpf[b])
        if val < best_val:
            best_val = val
            best_pair = [a, b]
            
        # Time check (unlikely to be needed, but good practice)
        if time.time() - start_time > deadline - 5:
            break
            
    return best_pair
