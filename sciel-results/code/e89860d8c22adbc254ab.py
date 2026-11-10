"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
from math import gcd
from functools import reduce

# EVOLVE-BLOCK-START
def _lcm(a, b):
    return a * b // gcd(a, b)

def _check_reciprocal_subsum(target_x, others):
    """Check if 1/target_x equals sum of 1/y for some subset of others."""
    if not others:
        return False
    # Convert to integer arithmetic: 1/target_x = sum(1/y for y in subset)
    # Multiply both sides by LCM of [target_x] + others
    # 1/target_x * LCM = sum(1/y * LCM for y in subset)
    # LCM / target_x = sum(LCM / y for y in subset)
    
    all_vals = [target_x] + list(others)
    lcm_val = reduce(_lcm, all_vals)
    
    target_int = lcm_val // target_x
    candidate_vals = [lcm_val // y for y in others]
    
    # Check if target_int can be formed as subset sum of candidate_vals
    # Use meet-in-the-middle for efficiency
    n = len(candidate_vals)
    if n == 0:
        return False
    
    half = n // 2
    left = candidate_vals[:half]
    right = candidate_vals[half:]
    
    left_sums = {0}
    for v in left:
        left_sums = left_sums | {s + v for s in left_sums}
    
    right_sums = {0}
    for v in right:
        right_sums = right_sums | {s + v for s in right_sums}
    
    for s in left_sums:
        if target_int - s in right_sums:
            return True
    
    return False

def _is_valid_set(selected):
    """Check if selected set is reciprocal-subsum-free."""
    selected = list(selected)
    for i, x in enumerate(selected):
        others = selected[:i] + selected[i+1:]
        if _check_reciprocal_subsum(x, others):
            return False
    return True

def _branch_and_bound(elements, best_set, deadline):
    """Branch and bound to find maximum valid subset."""
    if time.time() > deadline:
        return best_set
    
    if not elements:
        return best_set
    
    # Try including the first element
    x = elements[0]
    remaining = elements[1:]
    
    # Check if x can be added to current best without conflict
    # We need to check against all elements in remaining that might be selected
    # For now, try adding x and see if we can extend
    new_set = best_set + [x]
    
    # Quick check: is new_set valid?
    if _is_valid_set(new_set):
        # Recurse with remaining elements
        result = _branch_and_bound(remaining, new_set, deadline)
        if len(result) > len(best_set):
            best_set = result
    
    # Try not including x
    if time.time() > deadline:
        return best_set
    result = _branch_and_bound(remaining, best_set, deadline)
    if len(result) > len(best_set):
        best_set = result
    
    return best_set

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin before 90s limit
    
    # Start with a good initial set: larger numbers are less likely to have conflicts
    # 1/x for large x is small, so sums of reciprocals of other large numbers
    # are unlikely to equal 1/x
    initial_set = list(range(13, 25))
    
    if not _is_valid_set(initial_set):
        # Find a valid subset of initial_set
        valid_set = []
        for x in initial_set:
            test_set = valid_set + [x]
            if _is_valid_set(test_set):
                valid_set = test_set
        initial_set = valid_set
    
    best_set = initial_set
    
    # Try to improve by adding elements from 1-12
    candidates = list(range(1, 13))
    
    # Greedy improvement: try adding each candidate
    for x in candidates:
        if time.time() > deadline:
            break
        test_set = best_set + [x]
        if _is_valid_set(test_set):
            best_set = test_set
    
    # Try removing elements to free up space for others
    # Simple local search: try swapping
    for i in range(len(best_set)):
        if time.time() > deadline:
            break
        for j in range(1, 13):
            if j in best_set:
                continue
            # Try replacing best_set[i] with j
            new_set = best_set[:i] + best_set[i+1:] + [j]
            if _is_valid_set(new_set):
                # Check if this helps (e.g., allows adding more later)
                # For now, just accept if valid
                pass  # Swapping doesn't increase cardinality, skip
    
    # Use branch and bound on remaining candidates
    # Sort candidates by value (larger first to reduce conflicts)
    remaining_candidates = [x for x in range(1, 25) if x not in best_set]
    remaining_candidates.sort(reverse=True)
    
    if remaining_candidates and time.time() < deadline - 5:
        # Try to add elements one by one in greedy fashion
        for x in remaining_candidates:
            if time.time() > deadline:
                break
            test_set = best_set + [x]
            if _is_valid_set(test_set):
                best_set = test_set
    
    return best_set
# EVOLVE-BLOCK-END
