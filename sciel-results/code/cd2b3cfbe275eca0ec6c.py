"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random
from math import gcd
from functools import reduce

def _lcm(a, b):
    return a * b // gcd(a, b)

def _is_valid(subset):
    """Check if all reciprocal subset sums are distinct using integer numerators."""
    if not subset:
        return True
    k = len(subset)
    # Calculate LCM of the subset elements
    try:
        L = reduce(_lcm, subset)
    except ZeroDivisionError:
        return False
        
    # Calculate all subset sums as integers (numerators with common denominator L)
    # sum_{i in S} 1/a_i = (sum_{i in S} L/a_i) / L
    # We just need to check if all numerators are distinct.
    nums = [L // x for x in subset]
    
    # Use a set to store seen numerators
    seen = set()
    # Iterate through all 2^k subsets
    # Optimization: If k is large, this might be slow, but k<=22 is max 2^22 which is too big.
    # However, we expect k to be small (<=12). 2^12 = 4096, which is fast.
    for mask in range(1 << k):
        s = 0
        for i in range(k):
            if mask & (1 << i):
                s += nums[i]
        if s in seen:
            return False
        seen.add(s)
    return True

def construct():
    # We want to maximize the size of the subset.
    # Max possible size is 22, but realistically limited by the dissociated condition.
    # Let's try to find the maximum k by testing k from 12 down to 1.
    # For each k, we try to find a valid subset of size k using randomized greedy.
    
    all_nums = list(range(1, 23))
    
    # Try sizes from high to low
    for k in range(12, 0, -1):
        # Try multiple random restarts for each k
        for attempt in range(1000):
            random.shuffle(all_nums)
            current_set = []
            for num in all_nums:
                if len(current_set) >= k:
                    break
                # Check if adding num keeps the set valid
                candidate = current_set + [num]
                if _is_valid(candidate):
                    current_set.append(num)
            
            if len(current_set) == k:
                return current_set
        
        # If we couldn't find a set of size k in 1000 tries, try k-1
        # Note: It's possible a set of size k exists but greedy failed.
        # Given the small search space, if k is the true max, we should find it.
        # If k is not the max, we will find it in the next iteration.
        
    # Fallback: return a known valid small set
    # {1, 2, 4, 8, 16} is valid? 
    # Recips: 1, 1/2, 1/4, 1/8, 1/16.
    # Sums are distinct because binary representation is unique.
    return [1, 2, 4, 8, 16]
# EVOLVE-BLOCK-END
