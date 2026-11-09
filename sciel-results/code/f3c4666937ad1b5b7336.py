import time
import math

def construct():
    start_time = time.time()
    deadline = start_time + 80.0
    
    N_MIN = 10000
    N_MAX = 1000000
    MAX_N = N_MAX + 1
    
    # Linear Sieve for Powerful Parts
    # spf[i]: smallest prime factor
    # spf_pow[i]: p^k where p is spf and k is exponent
    # pp[i]: powerful part of i
    
    spf = [0] * (MAX_N + 1)
    spf_pow = [1] * (MAX_N + 1)
    pp = [1] * (MAX_N + 1)
    
    primes = []
    
    for i in range(2, MAX_N + 1):
        if spf[i] == 0:
            spf[i] = i
            spf_pow[i] = i
            pp[i] = 1
            primes.append(i)
        
        for p in primes:
            if p > spf[i] or i * p > MAX_N:
                break
            
            spf[i * p] = p
            
            if p == spf[i]:
                # p divides i
                spf_pow[i * p] = spf_pow[i] * p
                if spf_pow[i] == p:
                    # exponent was 1, now 2
                    pp[i * p] = pp[i] * p * p
                else:
                    # exponent was >= 2, now >= 3
                    pp[i * p] = pp[i] * p
                break
            else:
                # p < spf[i], so p is a new prime factor with exponent 1
                spf_pow[i * p] = p
                pp[i * p] = pp[i]
                
    # Find best n
    best_n = N_MIN
    best_val = pp[N_MIN] * pp[N_MIN + 1]
    
    for n in range(N_MIN, N_MAX + 1):
        val = pp[n] * pp[n + 1]
        if val > best_val:
            best_val = val
            best_n = n
        if time.time() > deadline:
            break
            
    return best_n
