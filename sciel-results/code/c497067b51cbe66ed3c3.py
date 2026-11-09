import math
import time

def construct():
    start_time = time.time()
    deadline = start_time + 50.0  # Safety margin
    
    max_val = 10210
    target_kernel = 210
    
    # Precompute smallest prime factor (SPF) for all numbers up to max_val
    # This allows O(log N) factorization
    spf = list(range(max_val + 1))
    for i in range(2, int(max_val**0.5) + 1):
        if spf[i] == i:  # i is prime
            for j in range(i*i, max_val + 1, i):
                if spf[j] == j:
                    spf[j] = i
    
    def get_squarefree_kernel(n):
        res = 1
        while n > 1:
            p = spf[n]
            cnt = 0
            while n % p == 0:
                n //= p
                cnt += 1
            if cnt % 2 == 1:
                res *= p
        return res

    # Precompute kernels and masks
    # We need masks for numbers 211..max_val
    # Mask is an integer where bit p is set if prime p is in the kernel
    masks = [0] * (max_val + 1)
    for i in range(211, max_val + 1):
        k = get_squarefree_kernel(i)
        mask = 0
        temp = k
        while temp > 1:
            p = spf[temp]
            mask |= (1 << p)
            while temp % p == 0:
                temp //= p
        masks[i] = mask
        
    # Target mask for 210
    # 210 = 2 * 3 * 5 * 7
    target_mask = (1 << 2) | (1 << 3) | (1 << 5) | (1 << 7)
    
    # Linear Basis
    # basis[p] = (vector, history_mask)
    basis = {}
    
    for M in range(211, max_val + 1):
        if time.time() > deadline:
            break
            
        v = masks[M]
        h = 1 << (M - 211) # History bit for this number
        
        # Insert v into basis
        temp_v = v
        temp_h = h
        while temp_v:
            p = temp_v.bit_length() - 1
            if p in basis:
                b_vec, b_hist = basis[p]
                temp_v ^= b_vec
                temp_h ^= b_hist
            else:
                basis[p] = (temp_v, temp_h)
                break
        
        # Check if target is in span
        t_v = target_mask
        t_h = 0
        while t_v:
            p = t_v.bit_length() - 1
            if p in basis:
                b_vec, b_hist = basis[p]
                t_v ^= b_vec
                t_h ^= b_hist
            else:
                break
                
        if t_v == 0:
            # Solution found
            sol_indices = []
            hh = t_h
            while hh:
                lsb = hh & -hh
                idx = lsb.bit_length() - 1
                sol_indices.append(211 + idx)
                hh ^= lsb
                
            sol = sol_indices
            if len(sol) <= 100:
                # Verify
                prod = 210
                for x in sol:
                    prod *= x
                r = math.isqrt(prod)
                if r * r == prod:
                    return [int(x) for x in sol]
            # If size > 100, continue? 
            # As discussed, unlikely for minimal M.
            pass
            
    # Fallback
    return [224, 240]
