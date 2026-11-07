import time
import math
import bisect
from collections import defaultdict

def construct():
    start_time = time.time()
    DEADLINE = 85.0  # Safety margin before 90s kill
    
    MAX_N = 10_000_000
    
    # 1. Sieve for smallest prime factor (SPF)
    spf = list(range(MAX_N + 1))
    for i in range(2, int(MAX_N**0.5) + 1):
        if spf[i] == i:  # i is prime
            for j in range(i * i, MAX_N + 1, i):
                if spf[j] == j:
                    spf[j] = i

    # 2. Compute sigma[n] for all n
    # sigma[n] = product of (p^(e+1)-1)/(p-1) for p^e || n
    sigma = [0] * (MAX_N + 1)
    sigma[1] = 1
    
    # We can compute sigma iteratively using spf
    # sigma[n] = sigma[n / p^e] * (p^(e+1)-1)/(p-1)
    # But we need to be careful with the power.
    # Let's use a helper array to store the largest power of spf[n] dividing n?
    # Or just factorize on the fly?
    # Factorization on the fly for 10^7 numbers might be slow in Python.
    
    # Better: 
    # Let p = spf[n].
    # Let m = n // p.
    # If m % p == 0:
    #    sigma[n] = sigma[m] + p^k ... this is getting complicated.
    
    # Let's stick to the additive sieve? 
    # O(N log N) = 10^7 * 16 = 1.6 * 10^8 operations.
    # In Python, 10^8 simple operations takes ~10-20 seconds.
    # It might be borderline.
    
    # Let's try the multiplicative sieve with a helper array for "sigma of p^k".
    # Actually, we can do:
    # sigma[n] = sigma[n // p] * (1 + p) if p does not divide n//p
    #           = sigma[n // p] + p^k * sigma[n // p^k] ... no.
    
    # Standard recurrence:
    # Let p = spf[n].
    # Let n = p^k * m, where gcd(m, p) = 1.
    # sigma[n] = sigma[m] * (1 + p + ... + p^k)
    
    # We can precompute `p_power[n]` and `sigma_p_power[n]`?
    # Or just store the "remaining part" after removing all factors of spf.
    
    # Let's use arrays:
    # spf[n]
    # exp[n]: exponent of spf[n] in n
    # core[n]: n / (spf[n] ** exp[n])
    
    exp = [0] * (MAX_N + 1)
    core = [0] * (MAX_N + 1)
    
    sigma[1] = 1
    exp[1] = 0
    core[1] = 1
    
    for n in range(2, MAX_N + 1):
        p = spf[n]
        m = n // p
        if m % p == 0:
            exp[n] = exp[m] + 1
            core[n] = core[m]
        else:
            exp[n] = 1
            core[n] = m
        
        # sigma[n] = sigma[core[n]] * (1 + p + ... + p^(exp[n]))
        # Let S_k = 1 + p + ... + p^k
        # We can compute S_k recursively or just sum it.
        # Since exp[n] is small (max ~23 for 2^23), summing is fine.
        # But doing it for every n is O(N * max_exp).
        
        # Optimization:
        # If exp[n] == 1: sigma[n] = sigma[m] * (1 + p)
        # If exp[n] > 1: sigma[n] = sigma[n // p] + p * sigma[n // p^2] ... no.
        # sigma[n] = sigma[core[n]] * (p^(exp[n]+1) - 1) // (p - 1)
        
        # To avoid repeated power calculation:
        # We can store `sigma_p_pow[n]` = 1 + p + ... + p^(exp[n])?
        # Let's just calculate it.
        
        # Actually, we can store `sigma_val` for the prime power part.
        # Let pp_sigma[n] = 1 + p + ... + p^(exp[n])
        # If exp[n] == 1: pp_sigma[n] = 1 + p
        # Else: pp_sigma[n] = pp_sigma[m] + p^(exp[n])
        
        # This requires storing p^exp.
        
        # Let's just use the formula with pow.
        # pow(p, exp[n]+1) is fast.
        
        k = exp[n]
        # sigma[n] = sigma[core[n]] * (p^(k+1) - 1) // (p - 1)
        # Note: core[n] is coprime to p, so sigma is multiplicative.
        
        # Calculate sum of geometric series
        # sum = 1
        # term = 1
        # for i in range(k):
        #   term *= p
        #   sum += term
        # This is O(k).
        
        # Faster:
        # if k == 1: s = 1 + p
        # else: s = (p**(k+1) - 1) // (p - 1)
        
        # p**(k+1) might be large but fits in int.
        
        s = 0
        term = 1
        for _ in range(k + 1):
            s += term
            term *= p
            
        sigma[n] = sigma[core[n]] * s

    # 3. Group by sigma
    groups = defaultdict(list)
    for n in range(1, MAX_N + 1):
        groups[sigma[n]].append(n)

    best_a = 11
    best_b = 6
    best_err_num = 4
    best_err_den = 12
    # Error = |2a - 3b| / (2b)
    
    # 4. Search
    # Iterate through groups
    # For each group, for each b, find best a
    
    count = 0
    for sig_val, lst in groups.items():
        if time.time() > DEADLINE:
            break
            
        if len(lst) < 2:
            continue
            
        # lst is already sorted
        for b in lst:
            target = 1.5 * b
            pos = bisect.bisect_left(lst, target)
            
            # Check candidates around pos
            # We want to minimize |2a - 3b| / (2b)
            # This is equivalent to minimizing |2a - 3b| for fixed b
            # So we want a closest to 1.5b
            
            candidates = []
            # Check pos and pos-1
            if pos < len(lst):
                candidates.append(lst[pos])
            if pos > 0:
                candidates.append(lst[pos-1])
            
            # In case the closest one is exactly 1.5b (error 0), we need the next one
            # So let's check a slightly wider window to be safe, or handle the 0 case.
            # If we find error 0, we skip it and look for the next best.
            # So we should check pos-1, pos, pos+1, pos-2, pos+2?
            # The error function is convex. The minimum is at 1.5b.
            # If the integer a at 1.5b gives 0, the next best are the neighbors.
            
            # Let's just check a small range around pos
            start_idx = max(0, pos - 2)
            end_idx = min(len(lst), pos + 3)
            
            for i in range(start_idx, end_idx):
                a = lst[i]
                num = abs(2 * a - 3 * b)
                if num == 0:
                    continue
                den = 2 * b
                # Compare num/den with best_err_num/best_err_den
                # num * best_err_den < best_err_num * den
                if num * best_err_den < best_err_num * den:
                    best_a = a
                    best_b = b
                    best_err_num = num
                    best_err_den = den
            count += 1
            if count % 1000000 == 0:
                if time.time() > DEADLINE:
                    break

    if best_a == 11 and best_b == 6:
        # Fallback if no better found (should not happen)
        return [11, 6]
        
    return [int(best_a), int(best_b)]
