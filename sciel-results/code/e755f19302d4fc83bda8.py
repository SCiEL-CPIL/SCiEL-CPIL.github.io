import time
from math import gcd
from functools import reduce

_START = time.time()
_DEADLINE = 85.0  # Safety margin for 90s limit

def _time_ok():
    return (time.time() - _START) < _DEADLINE

def construct():
    """
    Returns a nonempty subset of [1, 22] such that all reciprocal subset sums are distinct.
    Maximizes cardinality.
    """
    # 1. Calculate LCM of 1..22
    def lcm(a, b):
        return a * b // gcd(a, b)
    
    L = reduce(lcm, range(1, 23), 1)
    weights = [L // k for k in range(1, 23)]
    n_elem = 22
    
    # We want to maximize the size of the dissociated set.
    # Dissociated means all subset sums are distinct.
    # Let's use backtracking with pruning.
    
    best_size = 0
    best_set = []
    
    # Precompute conflicts to prune early?
    # If w_i + w_j == w_k, then {i, j, k} is not dissociated.
    # Specifically, if we have i and j, we cannot have k.
    # If we have i and k, we cannot have j (since w_j = w_i + w_k - 2w_k? No, w_i+w_j=w_k -> w_j = w_k - w_i).
    # Basically, any relation c1*w_i + c2*w_j + c3*w_k = 0 with c in {-1,0,1} and not all 0 is bad.
    
    # We will perform a DFS.
    # Order elements by weight descending to find large sets faster and prune via sum collisions.
    indices = list(range(n_elem))
    indices.sort(key=lambda i: weights[i], reverse=True)
    
    def can_add(current_sums, w):
        """
        Check if adding weight w to current_sums maintains dissociation.
        Returns the new set of sums if valid, None otherwise.
        """
        # Collision if (s + w) is in current_sums for any s in current_sums
        # Or if w is in current_sums (covered by s=0)
        # We iterate through current_sums and check if s+w is already in the set.
        # Note: We must check against the ORIGINAL current_sums, not the updated one.
        
        # Optimization: if w is very large, collisions are less likely with small sums.
        # But we must check all.
        
        # A faster check:
        # For each s in current_sums, if (s + w) in current_sums: return None
        # Then new_sums = current_sums | {s + w for s in current_sums}
        
        # To avoid creating the set twice, we can do:
        new_sums = set(current_sums)
        for s in current_sums:
            ns = s + w
            if ns in current_sums:
                return None
            new_sums.add(ns)
        return new_sums

    def backtrack(current_sums, chosen, remaining):
        nonlocal best_size, best_set
        
        if not _time_ok():
            return
        
        if len(chosen) > best_size:
            best_size = len(chosen)
            best_set = chosen[:]
            
        if not remaining:
            return
            
        # Pruning: even if we add all remaining, can we beat best_size?
        if len(chosen) + len(remaining) <= best_size:
            return
            
        # Try adding each element from remaining
        # We iterate in the order they are in 'remaining' (which is sorted by weight desc initially,
        # but we should keep it sorted for the recursion to maintain heuristic)
        
        for i in range(len(remaining)):
            if not _time_ok():
                return
            
            idx = remaining[i]
            w = weights[idx]
            
            # Check if we can add this element
            new_sums = can_add(current_sums, w)
            if new_sums is None:
                continue
                
            # Recurse
            new_chosen = chosen + [idx]
            new_remaining = remaining[i+1:]
            
            backtrack(new_sums, new_chosen, new_remaining)

    # Start backtracking
    # Initial sums: {0}
    backtrack({0}, [], indices)
    
    # If we found a set, convert indices back to values (1..22)
    # Indices are 0..21 corresponding to 1..22
    if best_set:
        result = sorted([idx + 1 for idx in best_set])
        return result
    else:
        # Fallback: single element is always valid
        return [1]

# Verification logic for debugging (optional, but good for safety)
def verify_dissociated(selected_values):
    # Convert values to weights
    def lcm(a, b):
        return a * b // gcd(a, b)
    L = reduce(lcm, range(1, 23), 1)
    weights = [L // k for k in selected_values]
    
    sums = set()
    n = len(weights)
    for mask in range(1 << n):
        s = 0
        for b in range(n):
            if mask & (1 << b):
                s += weights[b]
        if s in sums:
            return False
        sums.add(s)
    return True

# To ensure we don't return an invalid set due to a bug in can_add
# We will verify the result before returning.
_original_construct = construct

def construct():
    res = _original_construct()
    # If verification fails (shouldn't happen), fallback to a known valid set
    # A known valid set is any single element, e.g., [1]
    # Or we can hardcode a large valid set if we find one.
    # For now, trust the algorithm but add a check.
    if not verify_dissociated(res):
        # Fallback to a greedy construction of a valid set
        L = reduce(lcm, range(1, 23), 1)
        weights = [L // k for k in range(1, 23)]
        indices = list(range(22))
        indices.sort(key=lambda i: weights[i], reverse=True)
        current_sums = {0}
        valid_indices = []
        for idx in indices:
            w = weights[idx]
            new_sums = set(current_sums)
            valid = True
            for s in current_sums:
                if s + w in current_sums:
                    valid = False
                    break
            if valid:
                for s in current_sums:
                    new_sums.add(s + w)
                current_sums = new_sums
                valid_indices.append(idx)
        return sorted([i + 1 for i in valid_indices])
    return res
