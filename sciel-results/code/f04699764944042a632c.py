import math
from math import gcd

def construct():
    N = 512
    # Vertices are 1..511
    # We want to maximize sum 1/(512-a)
    # Let's map a to an index.
    # To make the bound tight, sort by weight descending.
    # Weight w(a) = 1/(512-a)
    # a=511 -> w=1
    # a=510 -> w=0.5
    # ...
    # a=1 -> w=1/511
    
    # Create list of (a, weight)
    items = [(a, 1.0 / (N - a)) for a in range(1, N)]
    # Sort by weight descending (which is a descending)
    items.sort(key=lambda x: -x[1])
    
    num_nodes = len(items)
    weights = [item[1] for item in items]
    values = [item[0] for item in items] # The actual integer a
    
    # Build conflict graph G
    # G[i] is a bitmask of nodes j such that gcd(values[i], values[j]) > 1
    G = [0] * num_nodes
    for i in range(num_nodes):
        for j in range(i + 1, num_nodes):
            if gcd(values[i], values[j]) > 1:
                G[i] |= (1 << j)
                G[j] |= (1 << i)
                
    # Precompute suffix sums of weights for upper bound
    # suffix_sum[i] = sum(weights[i] + ... + weights[num_nodes-1])
    suffix_sum = [0.0] * (num_nodes + 1)
    for i in range(num_nodes - 1, -1, -1):
        suffix_sum[i] = suffix_sum[i + 1] + weights[i]
        
    best_score = 0.0
    best_set = []
    
    # To find min index in mask:
    # Python int.bit_length() - 1 gives index of highest set bit.
    # We want lowest set bit.
    # (mask & -mask).bit_length() - 1 gives index of lowest set bit.
    
    def solve(mask, current_score, current_set_indices):
        nonlocal best_score, best_set
        
        if mask == 0:
            if current_score > best_score:
                best_score = current_score
                best_set = list(current_set_indices)
            return
            
        # Find lowest index in mask (highest weight)
        # low_bit = mask & -mask
        # v = low_bit.bit_length() - 1
        
        # Upper bound: current_score + sum of weights of all nodes in mask
        # Using suffix_sum[v] is a valid upper bound (overestimate)
        # Because mask is a subset of {v, v+1, ..., num_nodes-1}
        # Actually, mask might not contain v+1, etc.
        # But sum(mask weights) <= suffix_sum[v]
        
        # To be tighter, we can compute exact sum if mask is small?
        # Let's try exact sum first, it's only 511 bits.
        # But iterating 511 times in recursion might be slow.
        # Let's use the suffix_sum bound first. If it's not pruning enough, we'll see.
        
        # Get min index
        low_bit = mask & -mask
        v = low_bit.bit_length() - 1
        
        ub = current_score + suffix_sum[v]
        if ub <= best_score:
            return
            
        # Branch 1: Include v
        # New mask: remove v and its neighbors
        new_mask_include = mask & ~(1 << v) & ~G[v]
        current_set_indices.append(v)
        solve(new_mask_include, current_score + weights[v], current_set_indices)
        current_set_indices.pop()
        
        # Branch 2: Exclude v
        new_mask_exclude = mask & ~(1 << v)
        solve(new_mask_exclude, current_score, current_set_indices)
        
    # Initial mask: all nodes
    initial_mask = (1 << num_nodes) - 1
    solve(initial_mask, 0.0, [])
    
    # Reconstruct the set of integers
    result = [values[i] for i in best_set]
    return result
