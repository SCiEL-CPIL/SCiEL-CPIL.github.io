"""Program for large reciprocal-subsum-free sets using Branch and Bound with integer Diophantine checks."""
import time
from math import gcd
from functools import reduce

_START_TIME = time.time()
_DEADLINE = 85.0  # Safety margin under the 90s wall-clock limit

def _time_remaining():
    return _DEADLINE - (time.time() - _START_TIME)

def _lcm(a, b):
    return a * b // gcd(a, b)

def _lcm_list(lst):
    if not lst:
        return 1
    return reduce(_lcm, lst)

def _is_subset_sum_integer(target_int, weights):
    """
    Check if target_int can be formed by a sum of a non-empty subset of weights.
    weights are positive integers.
    Uses meet-in-the-middle for efficiency.
    """
    n = len(weights)
    if n == 0:
        return False
    
    # Quick checks
    total = sum(weights)
    if target_int > total:
        return False
    if target_int == 0:
        return False
        
    # If target is exactly one of the weights
    if target_int in weights:
        return True

    if n <= 20:
        # Meet in the middle
        half = n // 2
        w1 = weights[:half]
        w2 = weights[half:]
        
        # Generate all subset sums for w1
        sums1 = set([0])
        for w in w1:
            new_sums = set()
            for s in sums1:
                ns = s + w
                if ns <= target_int: # Pruning
                    new_sums.add(ns)
            sums1.update(new_sums)
            
        # Generate all subset sums for w2
        sums2 = set([0])
        for w in w2:
            new_sums = set()
            for s in sums2:
                ns = s + w
                if ns <= target_int: # Pruning
                    new_sums.add(ns)
            sums2.update(new_sums)
            
        # Check if any pair sums to target
        for s1 in sums1:
            if (target_int - s1) in sums2:
                return True
        return False
    else:
        # For n > 20, use a slightly more optimized MITM or DFS.
        # Given n <= 24, 2^12 * 2^12 = 16M ops max.
        # This is acceptable for a few calls.
        half = n // 2
        w1 = weights[:half]
        w2 = weights[half:]
        
        sums1 = set([0])
        for w in w1:
            new_sums = set()
            for s in sums1:
                ns = s + w
                if ns <= target_int:
                    new_sums.add(ns)
            sums1.update(new_sums)
            
        sums2 = set([0])
        for w in w2:
            new_sums = set()
            for s in sums2:
                ns = s + w
                if ns <= target_int:
                    new_sums.add(ns)
            sums2.update(new_sums)
            
        for s1 in sums1:
            if (target_int - s1) in sums2:
                return True
        return False

def _can_add(k, selected):
    """
    Check if k can be added to selected set S.
    1. 1/k is not a subsum of S.
    2. For all x in S, 1/x is not a subsum of (S \ {x}) U {k}.
       => If 1/x > 1/k, check if (1/x - 1/k) is a subsum of S \ {x}.
    """
    if not selected:
        return True
        
    # Scale factor for S
    L_S = _lcm_list(selected)
    
    # Check 1: Is 1/k a subsum of S?
    target_1 = L_S // k
    weights_1 = [L_S // y for y in selected]
    if _is_subset_sum_integer(target_1, weights_1):
        return False
        
    # Check 2: For each x in S
    for x in selected:
        if x >= k:
            # 1/x <= 1/k, so 1/x - 1/k <= 0. Sum of positive reciprocals > 0.
            continue
            
        # 1/x > 1/k
        S_prime = [y for y in selected if y != x]
        if not S_prime:
            continue
            
        L_S_prime = _lcm_list(S_prime)
        
        numerator = L_S_prime * (k - x)
        denominator = x * k
        
        if numerator % denominator != 0:
            continue # Not an integer, so impossible
            
        target_2 = numerator // denominator
        weights_2 = [L_S_prime // y for y in S_prime]
        
        if _is_subset_sum_integer(target_2, weights_2):
            return False
            
    return True

def _greedy_completion(current_set, candidates):
    """
    Estimate the maximum number of additional elements that can be added
    by greedily picking from candidates in descending order.
    """
    count = 0
    temp_set = list(current_set)
    # Sort candidates descending for greedy safety
    for c in sorted(candidates, reverse=True):
        if _can_add(c, temp_set):
            temp_set.append(c)
            count += 1
    return count

def construct():
    """
    Construct the largest subset of [1, 24] that is reciprocal-subsum-free
    using Branch and Bound.
    """
    global _best_set, _best_size
    
    _best_set = []
    _best_size = 0
    
    # Initial candidates: 24 down to 1
    candidates = list(range(24, 0, -1))
    
    # Helper for B&B
    def search(current_set, cand_indices):
        nonlocal _best_set, _best_size
        
        if _time_remaining() < 1.0:
            return
            
        current_size = len(current_set)
        
        # Update best
        if current_size > _best_size:
            _best_size = current_size
            _best_set = list(current_set)
            
        # Prune: Even if we add all remaining candidates, can we beat best?
        if current_size + len(cand_indices) <= _best_size:
            return
            
        # Heuristic Pruning: Greedy completion
        # This is expensive, so only do it if we are close to the limit or at shallow depth?
        # Or just do it always if time permits?
        # Let's do it if remaining candidates > 0
        if cand_indices:
            # To save time, only do greedy if we have a decent chance
            # If _best_size is already high, this prunes a lot.
            if _time_remaining() > 5.0:
                greedy_add = _greedy_completion(current_set, [candidates[i] for i in cand_indices])
                if current_size + greedy_add <= _best_size:
                    return
        
        if not cand_indices:
            return
            
        # Branch on the first candidate
        first_idx = cand_indices[0]
        c = candidates[first_idx]
        rest_indices = cand_indices[1:]
        
        # Branch 1: Include c
        if _can_add(c, current_set):
            new_set = current_set + [c]
            # Note: We don't filter rest_indices for conflicts with c here, 
            # because _can_add in the next step will handle it.
            # However, we can prune candidates in rest_indices that conflict with c?
            # If x in rest_indices conflicts with c, we can't add x if c is in set.
            # But we might skip c (Branch 2). So we can't remove x from rest_indices permanently.
            # We just pass rest_indices as is.
            search(new_set, rest_indices)
            
        # Branch 2: Exclude c
        search(current_set, rest_indices)

    search([], list(range(24)))
    
    return _best_set

# Quick self-test
if __name__ == "__main__":
    res = construct()
    print(f"Result: {res}, Size: {len(res)}")
    
    # Verify
    def verify(S):
        S = set(S)
        for x in S:
            others = [y for y in S if y != x]
            if not others:
                continue
            L = _lcm_list(others)
            target = L // x
            weights = [L // y for y in others]
            if _is_subset_sum_integer(target, weights):
                return False
        return True
        
    print(f"Valid: {verify(res)}")
