import math

def construct():
    N = 123457
    
    # Sieve for smallest prime factor
    spf = list(range(N + 1))
    for i in range(2, int(math.isqrt(N)) + 1):
        if spf[i] == i:  # i is prime
            for j in range(i * i, N + 1, i):
                if spf[j] == j:
                    spf[j] = i
    
    # Compute largest prime factor
    lpf = [0] * (N + 1)
    lpf[1] = 1
    for i in range(2, N + 1):
        p = spf[i]
        lpf[i] = max(p, lpf[i // p])
        
    best_a = 1
    best_b = N - 1
    best_val = max(lpf[1], lpf[N - 1])
    
    for a in range(1, N // 2 + 1):
        b = N - a
        val = max(lpf[a], lpf[b])
        if val < best_val:
            best_val = val
            best_a = a
            best_b = b
            
    return [best_a, best_b]
