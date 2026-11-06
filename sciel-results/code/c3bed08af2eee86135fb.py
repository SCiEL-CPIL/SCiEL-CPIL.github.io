import numpy as np
from itertools import combinations

def construct():
    p = 67
    squares = {x * x % p for x in range(1, p)}
    
    # out[i] is a bitmask of vertices j such that i -> j
    out = [0] * p
    # in_mask[j] is a bitmask of vertices i such that i -> j
    in_mask = [0] * p
    
    for i in range(p):
        for j in range(p):
            if i == j:
                continue
            if (j - i) % p in squares:
                out[i] |= (1 << j)
                in_mask[j] |= (1 << i)
                
    vertices = list(range(p))
    all_4sets = list(combinations(vertices, 4))
    num_sets = len(all_4sets)
    
    # Precompute S_mask and DomMask for each set
    S_masks = [0] * num_sets
    DomMasks = [0] * num_sets
    
    # doms[v] will store indices of sets dominated by v
    doms = [[] for _ in range(p)]
    
    for idx, s in enumerate(all_4sets):
        m = 0
        dom = (1 << p) - 1
        for v in s:
            m |= (1 << v)
            dom &= in_mask[v]
        S_masks[idx] = m
        DomMasks[idx] = dom
        
        # Add idx to doms[v] for all v in dom
        # Iterate through set bits of dom
        d = dom
        while d:
            lsb = d & -d
            v = lsb.bit_length() - 1
            doms[v].append(idx)
            d -= lsb

    active_mask = (1 << p) - 1
    removed = [False] * p
    
    # Greedy removal
    for v in range(p):
        if removed[v]:
            continue
            
        # Check if v is critical
        # v is critical if there exists a set S such that:
        # 1. S is fully active
        # 2. v is the unique active dominator of S
        is_critical = False
        for idx in doms[v]:
            # Check if S is fully active
            if S_masks[idx] & ~active_mask:
                continue
                
            # Check if v is the unique active dominator
            active_doms = DomMasks[idx] & active_mask
            if active_doms == (1 << v):
                is_critical = True
                break
                
        if not is_critical:
            removed[v] = True
            active_mask &= ~(1 << v)
            
    remaining_vertices = [v for v in range(p) if not removed[v]]
    n = len(remaining_vertices)
    
    # Map old indices to new indices
    old_to_new = {old: new for new, old in enumerate(remaining_vertices)}
    
    orientation = []
    for i in range(n):
        for j in range(i + 1, n):
            old_i = remaining_vertices[i]
            old_j = remaining_vertices[j]
            if (out[old_i] >> old_j) & 1:
                orientation.append(0)
            else:
                orientation.append(1)
                
    return {'n': n, 'orientation': orientation}
