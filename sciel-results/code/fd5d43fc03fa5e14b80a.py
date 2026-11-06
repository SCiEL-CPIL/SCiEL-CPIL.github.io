import numpy as np
from itertools import combinations

def construct():
    p = 67
    squares = {x * x % p for x in range(1, p)}
    
    # out[i] is a bitmask of vertices j such that i -> j
    out = [0] * p
    for i in range(p):
        mask = 0
        for j in range(p):
            if i == j:
                continue
            if (j - i) % p in squares:
                mask |= (1 << j)
        out[i] = mask
        
    # Generate all 4-sets
    vertices = list(range(p))
    all_4sets = list(combinations(vertices, 4))
    num_sets = len(all_4sets)
    
    # Map tuple to index
    tuple_to_idx = {s: i for i, s in enumerate(all_4sets)}
    
    # Precompute masks for each 4-set
    set_masks = [0] * num_sets
    for idx, s in enumerate(all_4sets):
        m = 0
        for v in s:
            m |= (1 << v)
        set_masks[idx] = m
        
    # Initialize counts and doms/sets_with
    counts = [0] * num_sets
    doms = [[] for _ in range(p)]
    sets_with = [[] for _ in range(p)]
    
    # Populate doms, sets_with, and initial counts
    # For each v, find 4-subsets of N+(v)
    for v in range(p):
        neighbors = [j for j in range(p) if (out[v] >> j) & 1]
        for s in combinations(neighbors, 4):
            idx = tuple_to_idx[s]
            counts[idx] += 1
            doms[v].append(idx)
            
    for idx, s in enumerate(all_4sets):
        for v in s:
            sets_with[v].append(idx)
            
    # Initialize unique_counts
    unique_counts = [0] * p
    for idx in range(num_sets):
        if counts[idx] == 1:
            # Find the unique dominator
            m = set_masks[idx]
            for v in range(p):
                if (out[v] & m) == m:
                    unique_counts[v] += 1
                    break
                    
    active = set(range(p))
    
    # Pruning loop
    while True:
        # Find a free vertex
        free_v = -1
        for v in range(p):
            if v in active and unique_counts[v] == 0:
                free_v = v
                break
                
        if free_v == -1:
            break
            
        v = free_v
        active.remove(v)
        
        # Update for sets containing v (they become inactive)
        for idx in sets_with[v]:
            s = all_4sets[idx]
            # Check if s was active (all other 3 are in active)
            # v is removed, so check others
            is_active = True
            for u in s:
                if u != v and u not in active:
                    is_active = False
                    break
            if is_active:
                # s was active, now inactive
                if counts[idx] == 1:
                    # Find unique dominator
                    m = set_masks[idx]
                    for u in active:
                        if (out[u] & m) == m:
                            unique_counts[u] -= 1
                            break
                            
        # Update for sets dominated by v (they remain active, but lose a dominator)
        for idx in doms[v]:
            s = all_4sets[idx]
            # Check if s is still active (all 4 in active)
            # v is not in s, so just check if all 4 are in active
            is_active = True
            for u in s:
                if u not in active:
                    is_active = False
                    break
            if is_active:
                old_count = counts[idx]
                counts[idx] = old_count - 1
                if old_count == 2:
                    # Now count is 1. Find new unique dominator.
                    m = set_masks[idx]
                    for u in active:
                        if (out[u] & m) == m:
                            unique_counts[u] += 1
                            break
                            
    # Construct the result
    remaining_vertices = sorted(list(active))
    n = len(remaining_vertices)
    
    # Map old indices to new indices
    old_to_new = {old: new for new, old in enumerate(remaining_vertices)}
    
    orientation = []
    for i in range(n):
        for j in range(i + 1, n):
            old_i = remaining_vertices[i]
            old_j = remaining_vertices[j]
            # Check orientation in original graph
            # 0 means i -> j, 1 means j -> i
            if (out[old_i] >> old_j) & 1:
                orientation.append(0)
            else:
                orientation.append(1)
                
    return {'n': n, 'orientation': orientation}
