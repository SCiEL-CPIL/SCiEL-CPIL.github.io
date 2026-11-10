import time
import math
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 85  # 85s safety margin for 90s kill switch
    
    N = 1000000
    # We need powerful parts for n and n+1, so up to N+1
    max_val = 1
    best_n = 10000
    
    # Array to store powerful parts
    pp = np.ones(N + 2, dtype=np.int64)
    
    # Sieve of Eratosthenes to get primes up to sqrt(N)
    limit = int(math.isqrt(N))
    is_prime = np.ones(limit + 1, dtype=bool)
    is_prime[0] = is_prime[1] = False
    for i in range(2, int(math.isqrt(limit)) + 1):
        if is_prime[i]:
            is_prime[i*i:limit+1:i] = False
                
    primes = np.where(is_prime)[0]
    
    # For each prime, update the powerful part for its multiples
    for p in primes:
        p = int(p)
        p2 = p * p
        if p2 > N + 1:
            continue
        
        # Iterate through multiples of p^2
        # For each multiple m, we need to find the full power of p in m
        # and multiply pp[m] by that power.
        
        # Precompute powers of p: p^2, p^3, ...
        powers = [p2]
        while powers[-1] * p <= N + 1:
            powers.append(powers[-1] * p)
        
        # For each power p^e (e>=2), mark all multiples of p^e
        # that are NOT multiples of p^(e+1)
        # This way, each number gets multiplied by exactly p^e where e is its exact exponent
        
        for i, pe in enumerate(powers):
            # Multiples of pe: pe, 2*pe, 3*pe, ...
            # But we only want those where the exponent of p is exactly i+2
            # i.e., multiples of pe but not multiples of pe*p
            
            if i + 1 < len(powers):
                next_power = powers[i+1]
                # Mark multiples of pe
                for m in range(pe, N + 2, pe):
                    pp[m] *= pe
                # Unmark multiples of next_power (they will be handled by higher powers)
                # But wait, we want to multiply by the FULL power.
                # So if m is divisible by p^3, we want to multiply by p^3, not p^2.
                # The approach: for each m, find the highest power of p dividing m.
                # Alternative: iterate m, and for each m divisible by p^2, compute the power.
                
                # Let's use a different approach:
                # For each m that is a multiple of p^2, compute the exact power.
                pass
            else:
                # This is the highest power
                for m in range(pe, N + 2, pe):
                    pp[m] *= pe
    
    # The above approach is flawed. Let's restart with the correct approach.
    
    # Reset pp
    pp = np.ones(N + 2, dtype=np.int64)
    
    for p in primes:
        p = int(p)
        p2 = p * p
        if p2 > N + 1:
            continue
        
        # For each multiple m of p^2, find the exact power of p in m
        # and multiply pp[m] by that power.
        
        # Efficient method using numpy:
        # For m = p^2, p^2+p, p^2+2p, ... (multiples of p^2)
        # Actually, just iterate over multiples of p^2.
        
        # Get all multiples of p^2
        multiples = np.arange(p2, N + 2, p2)
        
        # For each multiple, we need to find the exact power of p.
        # This is hard to vectorize easily. Let's use a loop but with numpy for the update.
        
        # Alternative: for each power p^e, find multiples of p^e that are not multiples of p^(e+1)
        # and multiply them by p^e.
        
        powers = []
        pe = p2
        while pe <= N + 1:
            powers.append(pe)
            pe *= p
        
        for i, pe in enumerate(powers):
            # Multiples of pe
            mult = np.arange(pe, N + 2, pe)
            # If there's a next power, exclude multiples of next power
            if i + 1 < len(powers):
                next_pe = powers[i+1]
                # Exclude multiples of next_pe
                # A number is a multiple of pe but not next_pe if it's not divisible by p^(i+3)
                # We can filter by checking m % next_pe != 0
                # But this is slow. Alternative:
                # Just multiply all multiples of pe by p, and then multiples of p^3 by p, etc.
                # No, that's not right either.
                
                # Let's just do it the simple way: for each multiple of p^2, compute the power.
                pass
            else:
                pp[mult] *= pe
    
    # This is getting complicated. Let's go back to the original approach but optimize it.
    
    pp = np.ones(N + 2, dtype=np.int64)
    
    for p in primes:
        p = int(p)
        p2 = p * p
        if p2 > N + 1:
            continue
        
        # For each multiple m of p^2, compute the exact power of p in m
        # and multiply pp[m] by that power.
        
        # We can do this by iterating over multiples of p^2 and computing the power.
        # To speed it up, we can precompute the powers and use a trick.
        
        # For m = k * p^2, the exponent of p in m is 2 + exponent of p in k.
        # So we can iterate over k and compute the exponent of p in k.
        
        max_k = (N + 1) // p2
        for k in range(1, max_k + 1):
            m = k * p2
            if m > N + 1:
                break
            # Compute exponent of p in k
            temp = k
            e = 2  # already have p^2
            while temp % p == 0:
                temp //= p
                e += 1
            pp[m] *= p**e
    
    # Now find the max product
    # pp[n] is the powerful part of n.
    # We want to maximize pp[n] * pp[n+1] for n in [10000, 1000000]
    
    max_prod = 0
    for n in range(10000, N + 1):
        prod = pp[n] * pp[n+1]
        if prod > max_prod:
            max_prod = prod
            best_n = n
            
        # Check deadline
        if time.time() > deadline:
            break
            
    return int(best_n)
