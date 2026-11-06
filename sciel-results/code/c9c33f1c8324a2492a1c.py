"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import math
import random
import time

def _init_spf(max_n):
    spf = list(range(max_n + 1))
    for i in range(2, int(math.isqrt(max_n)) + 1):
        if spf[i] == i:
            for j in range(i * i, max_n + 1, i):
                if spf[j] == j:
                    spf[j] = i
    return spf

SPF = _init_spf(20000)

def _factorize(n):
    if n <= 1:
        return set()
    primes = set()
    while n > 1:
        p = SPF[n]
        primes.add(p)
        while n % p == 0:
            n //= p
    return primes

def _get_cost(numbers):
    all_primes = set()
    n = len(numbers)
    for i in range(n):
        for j in range(i + 1, n):
            s = numbers[i] + numbers[j]
            all_primes |= _factorize(s)
    return len(all_primes)

def construct():
    n = 12
    max_val = 10000
    
    # Initialization: Try a few simple heuristics to pick a good start
    candidates = []
    for M in [2, 3, 4, 5, 6, 8, 10, 12]:
        nums = [M * k for k in range(1, n + 1)]
        if max(nums) <= max_val:
            candidates.append(nums)
            
    candidates.append([random.randint(1, max_val) for _ in range(n)])
    
    best_start = None
    best_start_cost = float('inf')
    for c in candidates:
        if len(set(c)) != n:
            continue
        cost = _get_cost(c)
        if cost < best_start_cost:
            best_start_cost = cost
            best_start = c.copy()
            
    if best_start is None:
        best_start = list(range(1, n + 1))
        best_start_cost = _get_cost(best_start)
        
    current_nums = best_start
    current_cost = best_start_cost
    best_nums = current_nums.copy()
    best_cost = current_cost
    
    start_time = time.time()
    timeout = 55.0
    
    T = 1.0
    T_min = 0.01
    alpha = 0.9995
    
    while T > T_min and time.time() - start_time < timeout:
        idx = random.randint(0, n - 1)
        old_val = current_nums[idx]
        
        if random.random() < 0.3:
            new_val = random.randint(1, max_val)
        else:
            delta = random.randint(-50, 50)
            new_val = old_val + delta
            if new_val < 1: new_val = 1
            if new_val > max_val: new_val = max_val
            
        if new_val == old_val or new_val in current_nums:
            continue
            
        current_nums[idx] = new_val
        new_cost = _get_cost(current_nums)
        
        if new_cost < current_cost:
            current_cost = new_cost
            if new_cost < best_cost:
                best_cost = new_cost
                best_nums = current_nums.copy()
        else:
            delta_cost = new_cost - current_cost
            if delta_cost > 0:
                prob = math.exp(-delta_cost / T)
                if random.random() < prob:
                    current_cost = new_cost
                else:
                    current_nums[idx] = old_val
        
        T *= alpha
        
    return best_nums
# EVOLVE-BLOCK-END
