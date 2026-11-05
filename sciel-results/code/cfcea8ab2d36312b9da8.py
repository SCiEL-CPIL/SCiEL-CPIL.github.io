import time
import math

def construct():
    start_time = time.time()
    deadline = start_time + 50.0
    
    # Max n is 1000000, so we need factors up to 1000001
    N = 1000002
    
    # Sieve for Smallest Prime Factor
    spf = list(range(N))
    for i in range(2, int(math.isqrt(N)) + 1):
        if spf[i] == i:
            for j in range(i*i, N, i):
                if spf[j] == j:
                    spf[j] = i
                    
    # Compute Largest Prime Factor
    lpf = [0] * N
    lpf[1] = 1
    for i in range(2, N):
        x = i
        last_prime = 1
        while x > 1:
            p = spf[x]
            last_prime = p
            while x % p == 0:
                x //= p
        lpf[i] = last_prime
        
    # Find best n
    best_n = 10000
    best_val = float('inf')
    
    for n in range(10000, 1000001):
        if time.time() > deadline:
            break
        val = max(lpf[n], lpf[n+1])
        if val < best_val:
            best_val = val
            best_n = n
            
    return best_n
