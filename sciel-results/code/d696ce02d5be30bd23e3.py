import sys
import time
import math

# EVOLVE-BLOCK-START

def construct():
    start_time = time.time()
    timeout = 110  # seconds, leave margin for overhead
    
    # We want to maximize L (max k to check)
    # k * sigma(k) <= 10^18
    # sigma(k) >= k, so k^2 <= 10^18 => k <= 10^9
    # But Python is slow. Let's start with a limit and increase if time permits.
    # Or just set a high limit and break on time.
    
    L = 20_000_000 # 20 million
    
    # Sieve of Smallest Prime Factors (SPF)
    spf = list(range(L + 1))
    for i in range(2, int(L**0.5) + 1):
        if spf[i] == i:  # i is prime
            for j in range(i*i, L + 1, i):
                if spf[j] == j:
                    spf[j] = i
    
    def sigma(n):
        if n == 1:
            return 1
        orig_n = n
        res = 1
        while n > 1:
            p = spf[n]
            power = 0
            p_sum = 1
            while n % p == 0:
                n //= p
                power += 1
                p_sum *= p
            res *= (p_sum * (p + 1) // (p - 1)) if p != 1 else p_sum
        return res

    # Actually, the above sigma is a bit slow due to function call and loop.
    # Let's try to optimize or just accept the speed.
    # 20 million calls to sigma might be too slow in pure Python.
    # sigma(n) is O(log n) operations.
    # 20M * ~10 ops = 200M ops. Python does ~10-50M ops/sec. So 4-20 seconds.
    # This should fit in the budget.
    
    best_M = 333312
    best_ks = [336, 372, 434]
    best_count = 3
    
    freq = {}
    
    # Pre-compute sigma for small n? No, just compute on the fly.
    # To speed up, we can iterate and compute.
    
    for k in range(1, L + 1):
        if time.time() - start_time > timeout:
            break
            
        # Quick bound check: k*k is minimum M.
        if k > 10**9:
            break
            
        # Calculate sigma(k)
        # Optimization: use the spf array
        n = k
        sig = 1
        while n > 1:
            p = spf[n]
            # Sum of powers of p in n
            p_power_sum = 0
            p_pow = 1
            while n % p == 0:
                n //= p
                p_pow *= p
                p_power_sum += p_pow
            sig *= p_power_sum
            
        M = k * sig
        
        if M > 10**18:
            # Since k increases, M generally increases. 
            # However, sig(k)/k can decrease. 
            # But k^2 term dominates. 
            # We can't strictly break here unless we are sure.
            # But if k is large, M is likely > 10^18.
            # Let's just skip adding to freq if M > 10^18.
            # Actually, if M > 10^18, it's invalid.
            continue
        
        if M not in freq:
            freq[M] = []
        freq[M].append(k)
        
        cnt = len(freq[M])
        if cnt > best_count:
            best_count = cnt
            best_M = M
            best_ks = freq[M][:] # Copy list
            
        # Optional: Early termination if best_count is very high?
        # Unlikely to get much higher than 5-10 in this range.
        
    return {'M': best_M, 'k': best_ks}

# EVOLVE-BLOCK-END
