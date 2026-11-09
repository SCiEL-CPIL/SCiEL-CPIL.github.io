import time
import random
from math import lcm
from functools import reduce

# Universe is [1, 24]
UNIVERSE = list(range(1, 25))
L = reduce(lcm, UNIVERSE)

def is_valid_subset(S):
    """
    Check if the set S is reciprocal-subsum-free.
    For every x in S, L/x must not be representable as a sum of a subset of {L/y for y in S, y != x}.
    """
    if not S:
        return True
    
    # Precompute weights
    weights = {x: L // x for x in S}
    
    # For each x in S, check if weights[x] is in the subset sums of others
    for x in S:
        target = weights[x]
        others = [weights[y] for y in S if y != x]
        
        # Quick check: if sum(others) < target, impossible
        if sum(others) < target:
            continue
            
        # If any weight is exactly target, and there's only one such weight? 
        # Wait, subset sum allows empty set? 
        # "sum of reciprocals of a subset of the OTHER selected elements"
        # Usually "subset" can be empty. Sum of empty set is 0.
        # 1/x is never 0, so empty set is never a solution for x >= 1.
        # So we look for non-empty subsets.
        
        # Use meet-in-the-middle for subset sum check
        # Since max size is 24, max others is 23.
        # 2^23 is too big for brute force. Meet in the middle is 2^11 + 2^12 ~ 6000, very fast.
        
        n = len(others)
        if n == 0:
            continue
            
        mid = n // 2
        left = others[:mid]
        right = others[mid:]
        
        left_sums = set()
        # Generate all subset sums of left
        for i in range(1 << len(left)):
            s = 0
            for j in range(len(left)):
                if i & (1 << j):
                    s += left[j]
            left_sums.add(s)
            
        # Check if target - right_sum is in left_sums
        found = False
        for i in range(1 << len(right)):
            s = 0
            for j in range(len(right)):
                if i & (1 << j):
                    s += right[j]
            if target - s in left_sums:
                # Ensure the subset is not empty if target==0? 
                # Target L/x is always >= 1. So subset sum 0 is irrelevant unless target=0.
                # So any match is a valid conflict.
                found = True
                break
        
        if found:
            return False
            
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 80.0  # Keep 10s safety margin
    
    best_set = set()
    
    # Strategy 1: Greedy with different orders
    # We want to maximize cardinality.
    # Let's try adding elements in various orders.
    
    orders = [
        list(reversed(UNIVERSE)),  # Descending: 24, 23, ..., 1
        list(UNIVERSE),            # Ascending: 1, 2, ..., 24
        list(reversed(UNIVERSE)),  # Duplicate for now, will add randoms
    ]
    
    # Add some random permutations
    for _ in range(10):
        random.shuffle(UNIVERSE)
        orders.append(list(UNIVERSE))
    
    for order in orders:
        if time.time() > deadline:
            break
            
        current_set = []
        for x in order:
            if time.time() > deadline:
                break
            candidate_set = current_set + [x]
            if is_valid_subset(candidate_set):
                current_set = candidate_set
                
        if len(current_set) > len(best_set):
            best_set = set(current_set)
            
    # If we have time, try local search / hill climbing
    # Try to add any remaining element
    if time.time() < deadline - 5:
        improved = True
        while improved and time.time() < deadline - 5:
            improved = False
            for x in UNIVERSE:
                if x in best_set:
                    continue
                candidate_set = list(best_set) + [x]
                if is_valid_subset(candidate_set):
                    best_set.add(x)
                    improved = True
                    break # Restart loop as set changed
            
            # Try removing one element to add two? (Swap)
            # This is complex. Simple addition is a good start.
            
    return list(best_set)
