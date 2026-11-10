import math

def construct():
    max_n = 1000000
    limit = max_n + 1
    
    # Sieve for Smallest Prime Factor
    spf = list(range(limit + 1))
    for i in range(2, int(math.isqrt(limit)) + 1):
        if spf[i] == i:  # i is prime
            for j in range(i * i, limit + 1, i):
                if spf[j] == j:
                    spf[j] = i
    
    # Precompute powerful parts for all numbers up to limit
    pp = [1] * (limit + 1)
    for i in range(2, limit + 1):
        x = i
        res = 1
        while x > 1:
            p = spf[x]
            power = 0
            while x % p == 0:
                x //= p
                power += 1
            if power >= 2:
                res *= (p ** power)
        pp[i] = res
    
    best_n = 10000
    best_val = 0
    
    for n in range(10000, max_n):
        val = pp[n] * pp[n + 1]
        if val > best_val:
            best_val = val
            best_n = n
            
    return best_n
