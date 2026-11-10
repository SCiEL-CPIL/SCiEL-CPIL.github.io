import time
import math

def construct():
    start_time = time.time()
    deadline = start_time + 50  # Safety margin before 60s limit
    
    N = 1000000
    LOW = 10000
    
    # Build smallest prime factor array
    spf = list(range(N + 1))
    for i in range(2, int(math.isqrt(N)) + 1):
        if spf[i] == i:  # i is prime
            for j in range(i * i, N + 1, i):
                if spf[j] == j:
                    spf[j] = i
    
    # Compute largest prime factor for each number
    lpf = [0] * (N + 1)
    lpf[1] = 1
    for i in range(2, N + 1):
        x = i
        max_pf = 1
        while x > 1:
            p = spf[x]
            max_pf = max(max_pf, p)
            while x % p == 0:
                x //= p
        lpf[i] = max_pf
    
    # Find best consecutive pair
    best_val = float('inf')
    best_n = LOW
    
    for n in range(LOW, N):
        if time.time() > deadline:
            break
        val = max(lpf[n], lpf[n + 1])
        if val < best_val:
            best_val = val
            best_n = n
    
    return best_n
