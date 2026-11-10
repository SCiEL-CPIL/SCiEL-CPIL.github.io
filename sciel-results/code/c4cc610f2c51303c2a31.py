import time
import bisect
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = start_time + 55  # Safety margin
    
    N = 10**7
    
    # Linear sieve for sigma(n)
    sigma = [0] * (N + 1)
    sigma[1] = 1
    power = [0] * (N + 1)
    sigma_power = [0] * (N + 1)
    is_prime = [True] * (N + 1)
    primes = []
    
    sig = sigma
    pow_ = power
    spow = sigma_power
    isp = is_prime
    prm = primes
    N_local = N
    prm_append = prm.append
    
    for i in range(2, N_local + 1):
        if isp[i]:
            prm_append(i)
            sig[i] = 1 + i
            pow_[i] = i
            spow[i] = 1 + i
        
        for p in prm:
            ip = i * p
            if ip > N_local:
                break
            isp[ip] = False
            if i % p == 0:
                pow_[ip] = pow_[i] * p
                spow[ip] = spow[i] + pow_[ip]
                sig[ip] = sig[i] // spow[i] * spow[ip]
                break
            else:
                pow_[ip] = p
                spow[ip] = 1 + p
                sig[ip] = sig[i] * (1 + p)
                
    if time.time() > deadline - 20:
        return [11, 6]

    # Group numbers by sigma value
    sigma_groups = defaultdict(list)
    for n in range(1, N_local + 1):
        sigma_groups[sig[n]].append(n)
        
    if time.time() > deadline - 10:
        return [11, 6]

    best_a, best_b = 11, 6
    best_err = abs(11/6 - 1.5)
    
    # Process groups
    for s_val, nums in sigma_groups.items():
        if time.time() > deadline - 2:
            break
            
        if len(nums) < 2:
            continue
            
        nums.sort()
        min_val = nums[0]
        max_val = nums[-1]
        
        # Quick check: does the ratio range [min/max, max/min] cover 1.5?
        ratio_min = min_val / max_val
        ratio_max = max_val / min_val
        
        if ratio_max < 1.5:
            potential_err = 1.5 - ratio_max
        elif ratio_min > 1.5:
            potential_err = ratio_min - 1.5
        else:
            potential_err = 0.0
            
        if potential_err >= best_err:
            continue
            
        # Binary search for each b
        for b in nums:
            target_a = 1.5 * b
            pos = bisect.bisect_left(nums, target_a)
            
            # Check pos
            if pos < len(nums):
                a = nums[pos]
                if a != b:
                    err = abs(a/b - 1.5)
                    if err < best_err and err != 0:
                        best_err = err
                        best_a, best_b = a, b
            
            # Check pos - 1
            if pos - 1 >= 0:
                a = nums[pos - 1]
                if a != b:
                    err = abs(a/b - 1.5)
                    if err < best_err and err != 0:
                        best_err = err
                        best_a, best_b = a, b
                        
    return [int(best_a), int(best_b)]
