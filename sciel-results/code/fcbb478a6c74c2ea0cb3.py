import math
import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = 82.0  # Keep a margin for safety (90s kill)
    
    N = 1000000
    MAX_VAL = N + 1  # We need powerful parts up to N+1 = 1000001
    
    # 1. Sieve of Eratosthenes to find Smallest Prime Factor (SPF)
    # Using a standard list-based sieve for clarity and speed for this range
    spf = list(range(MAX_VAL + 1))
    if MAX_VAL >= 1:
        spf[0] = 0
    if MAX_VAL >= 2:
        spf[1] = 1
        
    for i in range(2, int(math.isqrt(MAX_VAL)) + 1):
        if spf[i] == i:  # i is prime
            for j in range(i*i, MAX_VAL + 1, i):
                if spf[j] == j:
                    spf[j] = i

    # 2. Compute Powerful Part for each number using DP
    # This is significantly faster than repeated factorization.
    # P[i] = powerful part of i
    # If i is divisible by p^2 (where p=spf[i]), extract full p^e and multiply by P[i/p^e]
    # Else, P[i] = P[i/p]
    P = [1] * (MAX_VAL + 1)
    
    for i in range(2, MAX_VAL + 1):
        p = spf[i]
        # Check if i is divisible by p^2
        if i % (p * p) == 0:
            # Extract all factors of p from i
            n = i
            count = 0
            while n % p == 0:
                n //= p
                count += 1
            # The powerful part includes p^count and the powerful part of the remaining n
            P[i] = (p ** count) * P[n]
        else:
            # i has only one factor of p, so p doesn't contribute to powerful part
            P[i] = P[i // p]
        
        # Check time
        if time.time() - start_time > deadline - 10:
            break

    # If we broke early, P might be incomplete, but for N=10^6 it should finish fast (<2s)
    
    # 3. Find max P[n] * P[n+1]
    # We want to maximize the product.
    # We can use numpy for the final scan if P is complete, but a simple loop is also fast enough
    # and safer if P was partially filled. However, P should be complete.
    # Let's use a numpy vectorized approach for the scan to be safe and fast.
    
    # Convert P to numpy array for vectorized multiplication
    P_np = np.array(P, dtype=np.int64)
    
    # We need n in [10000, 1000000]
    # So we look at indices 10000 to 1000000 in P_np
    # Product is P_np[n] * P_np[n+1]
    
    # Slicing: P_np[10000:1000001] * P_np[100001:1000002]
    # Note: P has size MAX_VAL+1 = 1000002. Index 1000001 is valid.
    
    if time.time() - start_time < deadline - 2:
        products = P_np[10000:1000001] * P_np[100001:1000002]
        max_idx = np.argmax(products)
        best_n = 10000 + max_idx
        best_val = products[max_idx]
    else:
        # Fallback to loop if numpy conversion or time is tight
        best_n = 10000
        best_val = -1
        for n in range(10000, N + 1):
            val = P[n] * P[n+1]
            if val > best_val:
                best_val = val
                best_n = n
            if time.time() - start_time > deadline - 1:
                break

    return best_n
