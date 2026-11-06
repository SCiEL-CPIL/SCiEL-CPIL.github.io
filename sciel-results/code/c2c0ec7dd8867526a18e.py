import time

def construct():
    deadline = time.time() + 85
    TARGET = 123457
    
    # Sieve of Eratosthenes
    is_prime = [True] * (TARGET + 1)
    is_prime[0] = is_prime[1] = False
    limit = int(TARGET**0.5)
    for i in range(2, limit + 1):
        if is_prime[i]:
            for j in range(i*i, TARGET + 1, i):
                is_prime[j] = False
                
    primes = [i for i in range(2, TARGET + 1) if is_prime[i]]
    
    # Compute largest prime factor (LPF) for each number
    # lpf[n] is the largest prime factor of n
    lpf = [0] * (TARGET + 1)
    lpf[1] = 1  # P(1) = 1
    
    for p in primes:
        # Optimization: if p > TARGET, break (not possible here as primes <= TARGET)
        for j in range(p, TARGET + 1, p):
            lpf[j] = p
            
    # Find the pair (a, b) minimizing max(P(a), P(b))
    best_val = float('inf')
    best_a = 1
    best_b = TARGET - 1
    
    for a in range(1, TARGET):
        if time.time() > deadline - 5:
            break
        b = TARGET - a
        pa = lpf[a]
        pb = lpf[b]
        m = pa if pa > pb else pb
        if m < best_val:
            best_val = m
            best_a = a
            best_b = b
            
    return [best_a, best_b]
