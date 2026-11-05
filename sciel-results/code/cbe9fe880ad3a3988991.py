import time
from math import gcd
from functools import reduce

def _lcm(a, b):
    return a * b // gcd(a, b)

def _can_form_target_int(target_denom, values_denoms):
    """
    Check if 1/target_denom can be formed by sum of 1/y for y in values_denoms.
    Equivalent to: L/target_denom = sum(L/y) where L is LCM of all denoms.
    """
    if not values_denoms:
        return False
    
    # Calculate LCM of target and all values
    # To avoid huge numbers, we can work with normalized fractions or just check sums.
    # But LCM of numbers up to 24 is manageable (LCM(1..24) = 5354228880).
    
    denoms = [target_denom] + list(values_denoms)
    L = reduce(_lcm, denoms)
    
    target_num = L // target_denom
    values_nums = [L // y for y in values_denoms]
    
    # Subset sum check: can we sum a subset of values_nums to get target_num?
    # values_nums are integers.
    if sum(values_nums) < target_num:
        return False
    
    n = len(values_nums)
    # Sort descending for better pruning
    values_nums.sort(reverse=True)
    
    # Suffix sums for pruning
    suffix = [0] * (n + 1)
    for i in range(n - 1, -1, -1):
        suffix[i] = suffix[i + 1] + values_nums[i]
        
    def dfs(idx, current):
        if current == target_num:
            return True
        if idx == n:
            return False
        if current > target_num:
            return False
        if current + suffix[idx] < target_num:
            return False
            
        # Include
        if dfs(idx + 1, current + values_nums[idx]):
            return True
        # Exclude
        if dfs(idx + 1, current):
            return True
        return False
        
    return dfs(0, 0)

def _is_valid_set(S):
    S = list(S)
    for x in S:
        others = [y for y in S if y != x]
        if _can_form_target_int(x, others):
            return False
    return True

def construct():
    start = time.time()
    deadline = start + 80
    
    # Try multiple initial seeds to avoid local optima
    # Seed 1: 13-24 (current best)
    # Seed 2: 12-24
    # Seed 3: 11-24
    # ...
    # Actually, let's just run a greedy from empty? 
    # Greedy from empty (24 down to 1) is fast.
    # Let's try that first.
    
    best_size = 0
    best_set = set()
    
    # Strategy 1: Greedy from largest to smallest
    current = set()
    for c in range(24, 0, -1):
        if time.time() > deadline - 5:
            break
        trial = current | {c}
        if _is_valid_set(trial):
            current = trial
    if len(current) > best_size:
        best_size = len(current)
        best_set = current
        
    # Strategy 2: Try specific large seeds and extend downwards
    seeds = [
        list(range(13, 25)),
        list(range(12, 25)),
        list(range(11, 25)),
        list(range(10, 25)),
        list(range(14, 25)),
    ]
    
    for seed in seeds:
        if time.time() > deadline - 10:
            break
            
        current = set(seed)
        # Check if seed itself is valid
        if not _is_valid_set(current):
            continue
            
        # Try to add elements not in seed, from largest to smallest? 
        # Or from smallest to largest?
        # Usually adding smaller elements is harder because 1/small is large.
        # Let's try adding from 1 to 24 (excluding those in seed)
        # Actually, the previous code added from 12 down to 1.
        # Let's try adding from 1 to 24.
        
        for c in range(1, 25):
            if c in current:
                continue
            if time.time() > deadline - 5:
                break
            trial = current | {c}
            if _is_valid_set(trial):
                current = trial
                
        if len(current) > best_size:
            best_size = len(current)
            best_set = current
            
    return sorted(best_set)
