import numpy as np
import time
from math import isqrt

def construct():
    start_time = time.time()
    deadline = start_time + 82.0  # safety margin before 90s kill
    
    N = 10000000
    
    # Step 1: Sieve of Eratosthenes to get smallest prime factor (SPF)
    # We need SPF up to N for factorization
    spf = np.arange(N + 1, dtype=np.int64)
    # Mark composites
    for i in range(2, isqrt(N) + 1):
        if spf[i] == i:  # i is prime
            for j in range(i * i, N + 1, i):
                if spf[j] == j:
                    spf[j] = i
    
    if time.time() > deadline - 20.0:
        return [23, 15]
    
    # Step 2: Compute sigma(n) using the SPF-based approach
    # sigma(n) = product over prime powers p^k || n of (p^(k+1)-1)/(p-1)
    # We compute this incrementally using a modified sieve approach
    
    sigma = np.ones(N + 1, dtype=np.int64)
    
    # Use a vectorized sieve for sigma
    # For each prime p, for each multiple m of p, we need to account for p^k dividing m
    # Alternative: use the multiplicative property with SPF
    
    # Actually, let's use a more direct sieve approach:
    # sigma(n) = sum of divisors. We can compute it via:
    # For i from 1 to N: for j from i to N step i: sigma[j] += i
    # But this is O(N log N) which might be too slow for N=10^7 in 60s
    
    # Better: use the formula sigma(n) = prod (p^(k+1)-1)/(p-1)
    # Compute using SPF factorization for each n
    
    # Precompute sigma using a linear sieve approach
    # sigma is multiplicative, so we can use a linear sieve
    
    sigma = np.ones(N + 1, dtype=np.int64)
    spf_arr = spf.copy()
    # We need to track the largest prime power dividing each number
    # Let's use a different approach: for each n, factor using spf and compute sigma
    
    # Actually, let's use an efficient multiplicative sieve:
    # For each prime p, for each power p^k, update sigma for multiples
    
    # Let me try a different approach: compute sigma via the divisor sum sieve
    # but optimized with numpy
    
    sigma = np.zeros(N + 1, dtype=np.int64)
    
    # Vectorized divisor sum sieve
    # For i in 1..N: sigma[i::i] += i
    # This is O(N log N) but with numpy it might be fast enough
    
    # Let's do it in chunks to avoid memory issues
    # Actually for N=10^7, sigma[i::i] for small i creates large arrays
    
    # Alternative: use the formula with SPF
    # For each n from 2 to N:
    #   p = spf[n]
    #   m = n // p
    #   if m % p == 0:
    #       # p^2 divides n, need to track powers
    #   else:
    #       sigma[n] = sigma[m] * (p+1)
    
    # This requires tracking the highest power of spf dividing n
    # Let's use arrays: sigma[n], and p_power[n] = p^k where p^k || n
    
    sigma = np.ones(N + 1, dtype=np.int64)
    p_power = np.ones(N + 1, dtype=np.int64)  # p_power[n] = p^k where p=spf[n], p^k || n
    
    for n in range(2, N + 1):
        if time.time() > deadline - 15.0:
            break
        p = int(spf[n])
        m = n // p
        if m % p == 0:
            # p^2 divides n
            p_power[n] = p_power[m] * p
            # sigma(n) = sigma(m) - p^(k-1) + p^k where p^k || n
            # Actually: if p^k || n, sigma(n) = sigma(n/p^k) * (1+p+...+p^k)
            # If p^(k-1) || m, then p^k || n
            # sigma(n) = sigma(m) - p^(k-1) + p^k ... no
            # Let me think again.
            # sigma is multiplicative. If n = p^k * r with gcd(p,r)=1, sigma(n) = sigma(p^k)*sigma(r)
            # sigma(p^k) = (p^(k+1)-1)/(p-1)
            # If m = p^(k-1)*r, sigma(m) = sigma(p^(k-1))*sigma(r)
            # sigma(n)/sigma(m) = sigma(p^k)/sigma(p^(k-1)) = (p^(k+1)-1)/(p^k-1)
            # So sigma(n) = sigma(m) * (p^(k+1)-1)/(p^k-1)
            # But this involves division which may not be integer in floating point
            # Better: sigma(n) = sigma(m) - p^(k-1) + p^k ... let's verify
            # sigma(p^k) = 1+p+...+p^k, sigma(p^(k-1)) = 1+p+...+p^(k-1)
            # sigma(p^k) = sigma(p^(k-1)) + p^k
            # So sigma(n) = sigma(r) * sigma(p^k) = sigma(r) * (sigma(p^(k-1)) + p^k) = sigma(m) + sigma(r)*p^k
            # But sigma(r) = sigma(m) / sigma(p^(k-1)) ... this is getting complicated
            # 
            # Simpler: just use sigma(n) = sigma(m) + p^k * sigma(r) where r = n/p^k
            # But we don't have r easily.
            #
            # Let's use: sigma(n) = sigma(m) - p^(k-1) + p^k is WRONG.
            # Correct: sigma(p^k) = sigma(p^(k-1)) + p^k
            # So sigma(n) = sigma(r) * sigma(p^k) = sigma(r) * (sigma(p^(k-1)) + p^k) = sigma(m) + sigma(r) * p^k
            # We need sigma(r). r = n / p^k.
            # 
            # Alternative simpler approach: just factor each n and compute sigma directly.
            # For N=10^7 this might be too slow in Python.
            
            # Let me use a cleaner multiplicative sieve:
            # sigma[n] for n where spf[n]=p:
            #   if n/p has spf different from p: sigma[n] = sigma[n/p] * (1+p)
            #   else: let p^k || n, then sigma[n] = sigma[n/p^k] * (1+p+...+p^k)
            
            # I'll track: for each n, let p=spf[n], and let n = p^k * r with gcd(p,r)=1
            # Store: p_pow[n] = p^k, and sigma_r[n] = sigma(r)
            # Then sigma[n] = sigma_r[n] * (p^(k+1)-1)//(p-1)
            
            # This is getting complex. Let me just do the direct divisor sum sieve with numpy.
            pass
        
        else:
            # p does not divide m, so n = p * m with gcd(p,m)=1
            # sigma[n] = sigma[m] * (1+p)
            sigma[n] = sigma[m] * (1 + p)
            p_power[n] = p
    
    # The above loop is too slow for N=10^7 in Python. Let me use a different approach.
    
    # Let's use the divisor sum sieve with numpy, but optimized:
    sigma = np.zeros(N + 1, dtype=np.int64)
    
    # Process in a smart way: for i from 1 to N, add i to all multiples
    # Use numpy slicing
    for i in range(1, N + 1):
        if time.time() > deadline - 10.0:
            break
        sigma[i::i] += i
    
    if time.time() > deadline - 5.0:
        return [23, 15]
    
    # Step 3: Group numbers by sigma value and find best pair
    # Create a dictionary mapping sigma value to list of (number, sigma)
    # But we want to minimize |a/b - 3/2|, so for each sigma value s,
    # we want to find a,b with sigma(a)=sigma(b)=s that minimize |a/b - 3/2|
    
    # For a given sigma value s, we have a set of numbers S_s.
    # We want to find a,b in S_s minimizing |a/b - 3/2| = |2a - 3b|/(2b)
    # For a fixed b, the best a is the one closest to 3b/2.
    
    # Approach: for each sigma value, sort the numbers. Then for each b, 
    # binary search for the closest a to 3b/2.
    
    # But there could be many sigma values. Let's be smart:
    # We only need to consider sigma values that appear at least twice.
    
    # First, find all sigma values that appear at least twice
    # Use numpy to do this efficiently
    
    # Create array of sigma values for n=1..N
    sig = sigma[1:N+1]
    nums = np.arange(1, N+1, dtype=np.int64)
    
    # Sort by sigma value
    order = np.argsort(sig, kind='stable')
    sorted_sig = sig[order]
    sorted_nums = nums[order]
    
    # Find groups with same sigma value and size >= 2
    # For each group, find best pair
    
    best_err = float('inf')
    best_pair = (23, 15)
    
    # Find boundaries of groups
    # A new group starts where sorted_sig changes
    changes = np.concatenate(([True], sorted_sig[1:] != sorted_sig[:-1]))
    group_starts = np.where(changes)[0]
    group_ends = np.concatenate((group_starts[1:], [len(sorted_sig)]))
    
    for gi in range(len(group_starts)):
        if time.time() > deadline - 3.0:
            break
        
        gs = group_starts[gi]
        ge = group_ends[gi]
        group_size = ge - gs
        
        if group_size < 2:
            continue
        
        # Get the numbers in this group
        group_nums = sorted_nums[gs:ge]  # already sorted since we used stable sort
        
        # For each b in group_nums, find a in group_nums closest to 3b/2
        # Since group_nums is sorted, we can use binary search
        
        # Vectorize: for each b, target = 1.5*b
        targets = 1.5 * group_nums.astype(np.float64)
        
        # Use searchsorted to find insertion points
        # For each target, find the closest element in group_nums
        # searchsorted gives the index where target would be inserted
        # The closest element is either at that index or index-1
        
        insert_pos = np.searchsorted(group_nums, targets, side='left')
        
        # Candidate indices: insert_pos and insert_pos-1
        # Clamp to valid range
        ip = np.clip(insert_pos, 0, len(group_nums) - 1)
        ip_minus = np.clip(insert_pos - 1, 0, len(group_nums) - 1)
        
        # Get candidate a values
        a_cand1 = group_nums[ip]
        a_cand2 = group_nums[ip_minus]
        
        # Compute error for each candidate
        # error = |a/b - 3/2| = |2a - 3b| / (2b)
        
        # For a_cand1
        b_vals = group_nums
        num1 = np.abs(2 * a_cand1 - 3 * b_vals)
        den1 = 2.0 * b_vals.astype(np.float64)
        err1 = num1 / den1
        
        # For a_cand2
        num2 = np.abs(2 * a_cand2 - 3 * b_vals)
        den2 = 2.0 * b_vals.astype(np.float64)
        err2 = num2 / den2
        
        # Take min
        err = np.minimum(err1, err2)
        
        # Also need to check a != b (they should be different numbers)
        # If a_cand1 == b_vals, that's the same number, invalid
        # But since we're looking for pairs, a and b should be distinct
        # Actually, the problem says find a,b with sigma(a)=sigma(b), a and b can be same?
        # "Find two numbers a, b" - likely distinct. Let's enforce a != b.
        
        # Mask out where a == b
        mask1 = (a_cand1 != b_vals)
        mask2 = (a_cand2 != b_vals)
        
        # For positions where a_cand1 == b_vals, use a_cand2 if valid
        final_a = np.where(mask1, a_cand1, a_cand2)
        final_err = np.where(mask1, err1, err2)
        
        # But if both a_cand1 and a_cand2 equal b_vals, skip
        both_same = (~mask1) & (~mask2)
        final_err[both_same] = float('inf')
        
        min_idx = np.argmin(final_err)
        min_err = final_err[min_idx]
        
        if min_err < best_err:
            min_a = int(final_a[min_idx])
            min_b = int(b_vals[min_idx])
            best_err = min_err
            best_pair = (min_a, min_b)
    
    return [int(best_pair[0]), int(best_pair[1])]
