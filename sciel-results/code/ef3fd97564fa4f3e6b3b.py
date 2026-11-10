import time
from math import gcd
from functools import reduce

def _lcm(a, b):
    return a * b // gcd(a, b)

def _lcm_list(lst):
    if not lst:
        return 1
    return reduce(_lcm, lst)

def _can_form_target(target_int, candidate_vals):
    """Check if target_int can be formed as a non-empty subset sum of candidate_vals."""
    n = len(candidate_vals)
    if n == 0:
        return False
    if target_int <= 0:
        return False
    if sum(candidate_vals) < target_int:
        return False
    
    # Meet in the middle for n <= 30
    if n <= 30:
        half = n // 2
        left = candidate_vals[:half]
        right = candidate_vals[half:]
        
        left_sums = set([0])
        for v in left:
            left_sums = left_sums | {s + v for s in left_sums}
        
        right_sums = set([0])
        for v in right:
            right_sums = right_sums | {s + v for s in right_sums}
        
        for ls in left_sums:
            if target_int - ls in right_sums:
                # Make sure it's a non-empty subset
                if ls == 0 and (target_int - ls) == 0:
                    continue
                return True
        return False
    else:
        # Fallback: check if any single element equals target, or try small subsets
        if target_int in candidate_vals:
            return True
        # Check pairs
        for i in range(n):
            for j in range(i+1, n):
                if candidate_vals[i] + candidate_vals[j] == target_int:
                    return True
        # Check triples
        for i in range(n):
            for j in range(i+1, n):
                for k in range(j+1, n):
                    if candidate_vals[i] + candidate_vals[j] + candidate_vals[k] == target_int:
                        return True
        return False

def _is_valid_addition(x, current_set):
    """Check if adding x to current_set maintains the reciprocal-subsum-free property."""
    if not current_set:
        return True
    
    others = list(current_set)
    
    # Check 1: Is 1/x equal to sum of reciprocals of any non-empty subset of others?
    all_vals = [x] + others
    lcm_val = _lcm_list(all_vals)
    target_int = lcm_val // x
    candidate_vals = [lcm_val // y for y in others]
    
    if _can_form_target(target_int, candidate_vals):
        return False
    
    # Check 2: For each y in current_set, is 1/y equal to sum of reciprocals of any non-empty subset of (current_set ∪ {x}) \ {y}?
    for y in current_set:
        new_others = [z for z in others if z != y] + [x]
        all_vals2 = [y] + new_others
        lcm_val2 = _lcm_list(all_vals2)
        target_int2 = lcm_val2 // y
        candidate_vals2 = [lcm_val2 // z for z in new_others]
        
        if _can_form_target(target_int2, candidate_vals2):
            return False
    
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 80.0
    
    result = set()
    
    # Greedy from largest to smallest
    for x in range(24, 0, -1):
        if time.time() > deadline:
            break
        if _is_valid_addition(x, result):
            result.add(x)
    
    if not result:
        result = {24}
    
    return sorted(result)
