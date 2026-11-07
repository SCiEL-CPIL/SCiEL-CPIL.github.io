import numpy as np
import time
from bisect import bisect_left
from fractions import Fraction

def construct():
    start_time = time.time()
    deadline = start_time + 78.0  # 12s safety margin
    
    N = 10000000
    
    # Compute sigma using a fast sieve approach
    # sigma(n) = product over prime factors p^k || n of (p^(k+1)-1)/(p-1)
    # Use a modified sieve: for each prime p, multiply sigma by (1 + p + ... + p^k) for all n divisible by p
    
    sigma = np.ones(N + 1, dtype=np.int64)
    
    # Sieve of Eratosthenes to find primes, then update sigma
    is_prime = np.ones(N + 1, dtype=bool)
    is_prime[0] = False
    is_prime[1] = False
    
    for i in range(2, int(np.sqrt(N)) + 1):
        if time.time() > deadline - 50.0:
            return [11, 6]
        if is_prime[i]:
            is_prime[i*i::i] = False
    
    # For each prime, update sigma values
    for p in range(2, N + 1):
        if time.time() > deadline - 45.0:
            return [11, 6]
        if is_prime[p]:
            # For all multiples of p, multiply sigma by (1 + p + p^2 + ... + p^k)
            # where p^k is the highest power of p dividing n
            # We do this by iterating through powers
            pk = p
            while pk <= N:
                # For n = pk * m where gcd(m, p) = 1, sigma gets factor (pk*p - 1)/(p-1)
                # But we need to be careful: sigma is multiplicative
                # Better approach: for each n divisible by p, compute the p-power part
                # This is complex. Let's use the SPF approach but optimized.
                break
            # Alternative: use the formula directly
            # For each multiple of p, we need to know the exact power
            # Let's use a different approach: compute sigma via the multiplicative formula
            # sigma(n) = prod (p_i^(e_i+1) - 1) / (p_i - 1)
            
            # Update sigma for multiples of p
            # For n = p^k * m, gcd(m,p)=1:
            # sigma(n) = sigma(m) * (p^(k+1)-1)/(p-1)
            # We can iterate: for k=1,2,3,... while p^k <= N
            # For each k, for n = p^k * m where m not divisible by p:
            #   sigma[n] should be multiplied by p^k (if not already accounted)
            # Actually, let's use a cleaner approach:
            
            # Initialize: sigma[n] = 1 for all n
            # For each prime p, for each power p^k <= N:
            #   For n = p^k * m where gcd(m, p) = 1:
            #     sigma[n] *= p^k
            #   Then at the end, for each n, sigma[n] = product of (1 + p + ... + p^k) for each prime power
            # This doesn't work directly because we need the full geometric series.
            
            # Let's use the standard approach: 
            # sigma = ones, then for each prime p:
            #   for n = p, 2p, 3p, ...:
            #     count power of p in n, say k
            #     multiply sigma[n] by (p^(k+1) - 1) / (p - 1)
            # But this requires knowing k for each n, which is expensive.
            
            # Best approach: use the recurrence with SPF, but compute SPF efficiently
            pass
    
    # Let's go back to SPF but make it faster
    # Reset
    sigma = np.ones(N + 1, dtype=np.int64)
    
    # Fast SPF computation using a modified sieve
    spf = np.zeros(N + 1, dtype=np.int64)
    primes = []
    
    # Use a bytearray for is_prime to speed up
    is_prime = bytearray(N + 1)
    is_prime[2:] = b'\x01' * (N - 1)
    
    for i in range(2, N + 1):
        if time.time() > deadline - 45.0:
            return [11, 6]
        if is_prime[i]:
            spf[i] = i
            primes.append(i)
            if i * i <= N:
                # Mark multiples
                for j in range(i * i, N + 1, i):
                    if is_prime[j]:
                        is_prime[j] = 0
                        spf[j] = i
        else:
            if spf[i] == 0:
                spf[i] = i
    
    if time.time() > deadline - 35.0:
        return [11, 6]
    
    # Compute sigma using the recurrence: sigma(n) = sigma(n/p) * (p^(k+1)-1)/(p-1) / ((p^k - 1)/(p-1))
    # where p^k is the highest power of p dividing n
    # More precisely: let p = spf[n], m = n/p
    # if m % p == 0: sigma[n] = sigma[m] + p^k * sigma[m/p^k] where p^k || n... 
    # Simpler: use pk and rest arrays
    pk = np.ones(N + 1, dtype=np.int64)
    rest = np.ones(N + 1, dtype=np.int64)
    sigma[1] = 1
    
    for i in range(2, N + 1):
        if time.time() > deadline - 25.0:
            return [11, 6]
        p = spf[i]
        m = i // p
        if m % p == 0:
            pk[i] = pk[m] * p
            rest[i] = rest[m]
        else:
            pk[i] = p
            rest[i] = m
        # sigma[i] = sigma[rest[i]] * (pk[i] * p - 1) // (p - 1)
        sigma[i] = sigma[rest[i]] * (pk[i] * p - 1) // (p - 1)
    
    if time.time() > deadline - 20.0:
        return [11, 6]
    
    # Group by sigma value - use numpy for speed
    # Create a dictionary mapping sigma value to list of n
    groups = {}
    for n in range(1, N + 1):
        if time.time() > deadline - 10.0:
            break
        s = int(sigma[n])
        if s in groups:
            groups[s].append(n)
        else:
            groups[s] = [n]
    
    if time.time() > deadline - 5.0:
        return [11, 6]
    
    # Find best pair: minimize |a/b - 3/2|
    # This is equivalent to minimizing |2a - 3b| / (2b)
    # So we want 2a ≈ 3b, i.e., a ≈ 1.5*b
    best_val = Fraction(1, 3)  # |11/6 - 3/2| = 1/3
    best_pair = (11, 6)
    
    for s, nums in groups.items():
        if time.time() > deadline - 2.0:
            break
        k = len(nums)
        if k < 2:
            continue
        # For each b in nums, find a in nums closest to 1.5*b
        # Use two-pointer or binary search
        for j in range(k):
            b = nums[j]
            target = (3 * b) // 2
            # Binary search for target in nums
            pos = bisect_left(nums, target)
            for i in (pos - 1, pos):
                if 0 <= i < k:
                    a = nums[i]
                    if a == b:
                        continue
                    # Exact rational: |a/b - 3/2| = |2a - 3b| / (2b)
                    num = abs(2 * a - 3 * b)
                    den = 2 * b
                    val = Fraction(num, den)
                    if val < best_val:
                        best_val = val
                        best_pair = (a, b)
    
    a, b = best_pair
    return [int(a), int(b)]
