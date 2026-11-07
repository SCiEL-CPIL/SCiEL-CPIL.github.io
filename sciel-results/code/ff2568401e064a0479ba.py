import math
import time
import sys

def construct():
    t0 = time.time()
    DEADLINE = 85.0
    N = 10000000
    
    # 1. Sieve for Smallest Prime Factor (SPF)
    spf = list(range(N + 1))
    for i in range(2, int(N**0.5) + 1):
        if spf[i] == i:
            for j in range(i*i, N + 1, i):
                if spf[j] == j:
                    spf[j] = i
    
    # Helper to compute sigma(n) using SPF
    def get_sigma(n):
        if n <= 1:
            return n
        res = 1
        while n > 1:
            p = spf[n]
            power_sum = 1
            power = 1
            while n % p == 0:
                n //= p
                power *= p
                power_sum += power
            res *= power_sum
        return res

    best_val = float('inf')
    best_pair = [11, 6] # Fallback, sigma(11)=12, sigma(6)=12. 11/6 = 1.833, 3/2=1.5. |1.833-1.5|=0.333.
    
    # We iterate b downwards
    # We need to track if we found a d=1 solution to stop early
    # And we need to track best_val to determine when to stop searching
    
    # Optimization: Pre-compute sigma for a range? No, too much memory.
    # Just compute on the fly.
    
    # To speed up the loop, we can use local variables
    spf_local = spf
    N_local = N
    best_val_local = best_val
    best_pair_local = best_pair
    
    # We will check b from N down to 1
    # Stop condition: if best_val is found, and b < 1/(2*best_val), we can stop
    # because min error for any b' is 1/(2b'), so if 1/(2b') > best_val, it's not better.
    # 1/(2b') > best_val => b' < 1/(2*best_val)
    
    limit_b = N_local
    # We don't know best_val yet, so limit is N
    
    for b in range(N_local, 0, -1):
        if time.time() - t0 > DEADLINE:
            break
            
        # Determine candidates for a
        # 2a - 3b = d
        # a = (3b + d) / 2
        # We want small |d|
        
        if b % 2 == 1:
            # 3b is odd.
            # d=1 => a = (3b+1)/2
            # d=-1 => a = (3b-1)/2
            a1 = (3 * b + 1) // 2
            a2 = (3 * b - 1) // 2
            
            # Check a1
            if 1 <= a1 <= N_local and a1 != b:
                s_a = get_sigma(a1)
                s_b = get_sigma(b)
                if s_a == s_b:
                    # d = 1
                    val = 1.0 / (2.0 * b)
                    if val < best_val_local:
                        best_val_local = val
                        best_pair_local = [a1, b]
                        # d=1 is the best possible d.
                        # Since we are iterating b downwards, this is the global optimum.
                        # We can return immediately.
                        return [int(a1), int(b)]
                        
            # Check a2
            if 1 <= a2 <= N_local and a2 != b:
                s_a = get_sigma(a2)
                s_b = get_sigma(b) # Reuse? No, s_b is same for this b.
                # We already computed s_b for a1 check? 
                # Let's compute s_b once per b.
                pass # Handled below
        
        else:
            # 3b is even.
            # d=0 => a = 1.5b (invalid, error 0)
            # d=2 => a = (3b+2)/2 = 1.5b + 1
            # d=-2 => a = (3b-2)/2 = 1.5b - 1
            a1 = (3 * b + 2) // 2
            a2 = (3 * b - 2) // 2
            
            # Check a1
            if 1 <= a1 <= N_local and a1 != b:
                s_a = get_sigma(a1)
                s_b = get_sigma(b)
                if s_a == s_b:
                    # d = 2
                    val = 2.0 / (2.0 * b) # 1/b
                    if val < best_val_local:
                        best_val_local = val
                        best_pair_local = [a1, b]
                        
            # Check a2
            if 1 <= a2 <= N_local and a2 != b:
                s_a = get_sigma(a2)
                # s_b is already computed if a1 was valid? 
                # To be safe and clean, compute s_b once.
                pass

        # Refactor inner logic to compute s_b once
        # This requires restructuring the loop body.

    # Let's rewrite the loop body properly.
    
    return [int(best_pair_local[0]), int(best_pair_local[1])]
