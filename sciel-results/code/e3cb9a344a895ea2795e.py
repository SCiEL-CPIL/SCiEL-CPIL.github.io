"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from math import gcd
from itertools import combinations

# Precompute LCM of 1..24 to use integer arithmetic
def _get_lcm_1_to_24():
    lcm_val = 1
    for i in range(1, 25):
        lcm_val = lcm_val * i // gcd(lcm_val, i)
    return lcm_val

W = _get_lcm_1_to_24()
# Precompute integer reciprocals: int_recip[x] = W // x
INT_RECIP = {x: W // x for x in range(1, 25)}

def _can_form_target(target_int, elements_ints):
    """
    Check if target_int can be formed by a non-empty subset of elements_ints.
    elements_ints are integer scaled reciprocals.
    Uses meet-in-the-middle for larger sets to improve speed.
    """
    if not elements_ints:
        return False
    
    n = len(elements_ints)
    
    # If target is smaller than the smallest element, it can't be formed
    # (since all elements are positive and distinct, and we need a non-empty subset)
    # Actually, if min(elements_ints) > target_int, return False.
    if min(elements_ints) > target_int:
        return False
        
    # If the sum of all elements is less than target, return False
    if sum(elements_ints) < target_int:
        return False
        
    # For small n, simple DP or set of sums is fine
    if n <= 12:
        sums = {0}
        for val in elements_ints:
            new_sums = set()
            for s in sums:
                ns = s + val
                if ns == target_int:
                    return True
                if ns < target_int:
                    new_sums.add(ns)
            sums.update(new_sums)
        return target_int in sums

    # For larger n, use meet-in-the-middle
    # Split elements into two halves
    mid = n // 2
    left = elements_ints[:mid]
    right = elements_ints[mid:]
    
    # Generate all subset sums for left half
    left_sums = {0}
    for val in left:
        new_sums = set()
        for s in left_sums:
            ns = s + val
            if ns <= target_int: # Prune sums that exceed target
                new_sums.add(ns)
        left_sums.update(new_sums)
    
    # Generate all subset sums for right half
    right_sums = {0}
    for val in right:
        new_sums = set()
        for s in right_sums:
            ns = s + val
            if ns <= target_int: # Prune sums that exceed target
                new_sums.add(ns)
        right_sums.update(new_sums)
    
    # Check if any pair of sums (one from left, one from right) adds up to target
    # We need to ensure the subset is non-empty.
    # If target == 0, it's invalid because we need a non-empty subset. But target_int is always positive.
    # So if we find l + r == target, and (l, r) != (0, 0), it's a valid subset.
    # Since target > 0, (0,0) sums to 0, not target. So any match is valid.
    
    # Convert one set to a set for O(1) lookup
    right_sums_set = right_sums
    for l in left_sums:
        needed = target_int - l
        if needed in right_sums_set:
            # Check if the subset is non-empty
            # Since target > 0, l + needed = target > 0 implies not (l=0 and needed=0).
            # So the subset is non-empty.
            return True
            
    return False

def _is_valid(S):
    """
    Check if S is a valid reciprocal-subsum-free set.
    """
    S_list = list(S)
    for x in S_list:
        target_int = INT_RECIP[x]
        others_ints = [INT_RECIP[y] for y in S_list if y != x]
        if _can_form_target(target_int, others_ints):
            return False
    return True

def _find_best_set():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin for 60s limit
    
    # Strategy: Greedy construction + Local Search
    
    # 1. Greedy Initialization
    # Try a few different orders to get a good starting point.
    orders = [
        list(range(1, 25)),
        list(range(24, 0, -1)),
        # Interleaved: 1, 24, 2, 23, ...
        [i for i in (1, 24, 2, 23, 3, 22, 4, 21, 5, 20, 6, 19, 7, 18, 8, 17, 9, 16, 10, 15, 11, 14, 12, 13)]
    ]
    
    best_set = []
    best_size = 0
    
    for order in orders:
        current_set = []
        for x in order:
            if time.time() > deadline - 10.0:
                break
            candidate_set = current_set + [x]
            if _is_valid(candidate_set):
                current_set = candidate_set
        
        if len(current_set) > best_size:
            best_size = len(current_set)
            best_set = list(current_set)
            
        if time.time() > deadline - 10.0:
            break

    # 2. Local Search
    current = best_set[:]
    current_set = set(current)
    all_elements = set(range(1, 25))
    
    improved = True
    while improved and time.time() < deadline - 2.0:
        improved = False
        
        # Try adding any element
        for x in all_elements - current_set:
            if time.time() > deadline - 2.0:
                break
            test_set = current + [x]
            if _is_valid(test_set):
                current = test_set
                current_set = set(current)
                improved = True
                break # Restart loop
        
        # If no addition was possible, try swaps (remove y, add z)
        # We only accept a swap if it leads to a state where we can add MORE elements later?
        # That's hard to check. 
        # Instead, let's try: Remove y. Try to add any z. 
        # If we can add z, and z != y, we have a swap.
        # To avoid cycling, we can keep a history of visited states or just rely on the "improved" flag 
        # only being set if we actually increased size.
        # But swaps keep size same. 
        # Let's do a "2-step" lookahead:
        # Remove y. 
        # Try to add z.
        # If valid, try to add w (w != z, w != y).
        # If w can be added, then we have net +1.
        # This is expensive but might be necessary to escape local optima.
        
        if not improved:
            # Try 1-removal, 2-addition sequences
            found_improvement = False
            for y in current:
                if time.time() > deadline - 2.0:
                    break
                reduced = [z for z in current if z != y]
                reduced_set = set(reduced)
                
                # Try to add any z to reduced
                candidates_to_add = list(all_elements - reduced_set)
                # Sort candidates? Maybe smaller numbers are more "valuable" or "dangerous"?
                # Just iterate.
                for z in candidates_to_add:
                    if time.time() > deadline - 2.0:
                        break
                    test_set_1 = reduced + [z]
                    if _is_valid(test_set_1):
                        # Now try to add any w to test_set_1
                        test_set_1_set = set(test_set_1)
                        for w in all_elements - test_set_1_set:
                            if time.time() > deadline - 2.0:
                                break
                            test_set_2 = test_set_1 + [w]
                            if _is_valid(test_set_2):
                                # We found a sequence: remove y, add z, add w.
                                # Net gain: +1.
                                current = test_set_2
                                current_set = set(current)
                                improved = True
                                found_improvement = True
                                break
                        if found_improvement:
                            break
                if found_improvement:
                    break

    return current

def construct():
    start_time = time.time()
    
    result = _find_best_set()
    
    if not result:
        result = [13]
    
    # Final validation check
    if not _is_valid(result):
        result = list(range(13, 25))
        
    return result
# EVOLVE-BLOCK-END
