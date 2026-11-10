import random
import time
from fractions import Fraction

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin
    
    # Precompute reciprocals
    recip = {i: Fraction(1, i) for i in range(1, 25)}
    
    def can_make_sum(target, others, idx, remaining_sum):
        if target == 0:
            return True
        if idx == len(others) or remaining_sum < target:
            return False
        
        # Pruning: if the largest remaining element is smaller than target, 
        # we might still be able to sum smaller ones, so no simple prune here 
        # except remaining_sum check.
        
        # If current element is too big, we MUST skip it
        if others[idx] > target:
            return can_make_sum(target, others, idx + 1, remaining_sum - others[idx])
        
        # Try taking it
        if can_make_sum(target - others[idx], others, idx + 1, remaining_sum - others[idx]):
            return True
        
        # Try skipping it
        if can_make_sum(target, others, idx + 1, remaining_sum - others[idx]):
            return True
        
        return False

    def is_valid(S):
        S_list = list(S)
        n = len(S_list)
        if n < 2:
            return True
        
        for i, x in enumerate(S_list):
            target = recip[x]
            # Get reciprocals of others
            others = [recip[y] for y in S_list if y != x]
            # Sort descending for better pruning (larger values first)
            others.sort(reverse=True)
            
            # Calculate total sum of others
            total_sum = sum(others)
            if total_sum < target:
                continue # Cannot possibly reach target
            
            # Check if target can be formed
            if can_make_sum(target, others, 0, total_sum):
                return False
        
        return True

    best_set = list(range(13, 25)) # Seed
    best_size = len(best_set)
    
    # Local Search / Randomized Greedy
    # We try to build a set by adding elements in random order
    # If adding an element causes invalidity, we skip it.
    # To find larger sets, we try many random orders.
    
    elements = list(range(1, 25))
    
    while time.time() < deadline:
        random.shuffle(elements)
        S = set()
        
        for x in elements:
            # Check if adding x keeps S valid
            # Optimization: only check x against existing S, and existing S against x?
            # is_valid(S | {x}) checks all.
            # We can optimize: 
            # 1. Check if 1/x is sum of subset of S.
            # 2. Check if for any y in S, 1/y is sum of subset of (S \ {y}) U {x}.
            #    This is expensive.
            # Given N is small, just use is_valid.
            
            if is_valid(S | {x}):
                S.add(x)
        
        if len(S) > best_size:
            best_size = len(S)
            best_set = list(S)
            
            # If we found a very large set, we can try to improve it
            # by trying to add remaining elements?
            # The loop above already tries all elements in random order.
            
            # If we reached 24, we can break
            if best_size == 24:
                break

    return best_set
