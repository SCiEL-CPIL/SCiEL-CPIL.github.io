# EVOLVE-BLOCK-START
import random
from math import gcd
from functools import reduce

def _lcm(a, b):
    return abs(a * b) // gcd(a, b)

def _lcm_list(lst):
    return reduce(_lcm, lst, 1)

def construct():
    """
    Find a nonempty subset of [1, 22] with maximum cardinality such that
    all reciprocal subset sums (including empty sum = 0) are distinct.
    
    Uses integer arithmetic by scaling by the LCM of the selected elements.
    Uses greedy approach with multiple random restarts.
    """
    candidates = list(range(1, 23))  # [1, 2, ..., 22]
    best_set = []
    
    # Try deterministic orderings first
    orderings = [
        sorted(candidates, reverse=True),  # Descending
        sorted(candidates),                # Ascending
    ]
    
    # Add many random orderings
    rng = random.Random(42)
    for _ in range(200):
        order = candidates[:]
        rng.shuffle(order)
        orderings.append(order)
        
    for order in orderings:
        current = []
        current_lcm = 1
        current_sums = {0}  # Integer sums scaled by current_lcm
        
        for candidate in order:
            # If we add 'candidate', new LCM is lcm(current_lcm, candidate)
            new_lcm = _lcm(current_lcm, candidate)
            
            # Scale factor for existing sums
            scale = new_lcm // current_lcm
            
            # New value for the candidate element
            val = new_lcm // candidate
            
            # New sums are {s * scale, s * scale + val for s in current_sums}
            # We need to check if these 2 * len(current_sums) values are distinct.
            
            new_sums = set()
            for s in current_sums:
                scaled_s = s * scale
                new_sums.add(scaled_s)
                new_sums.add(scaled_s + val)
                
            if len(new_sums) == 2 * len(current_sums):
                # No collision
                current.append(candidate)
                current_lcm = new_lcm
                current_sums = new_sums
            # else: collision, skip
            
        if len(current) > len(best_set):
            best_set = current[:]
            
    if not best_set:
        best_set = [1]
        
    return sorted(best_set)
# EVOLVE-BLOCK-END
