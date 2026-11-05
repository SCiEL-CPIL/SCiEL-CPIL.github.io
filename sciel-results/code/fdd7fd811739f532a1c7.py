"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
from itertools import combinations
import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin before 90s kill
    n = 16
    k = 4
    
    # Generate all 4-element subsets of [1,16]
    all_subsets = list(combinations(range(1, n+1), k))
    total = len(all_subsets)
    
    # Precompute bitmasks for fast intersection
    masks = [0] * total
    for i, s in enumerate(all_subsets):
        m = 0
        for bit in s:
            m |= (1 << (bit - 1))
        masks[i] = m
    
    # Precompute pairwise intersections as a 2D array for fast access
    # inter[i][j] = masks[i] & masks[j]
    # This is 1820x1820 = ~3.3M integers, manageable
    inter = np.zeros((total, total), dtype=np.int32)
    for i in range(total):
        mi = masks[i]
        for j in range(i + 1, total):
            val = mi & masks[j]
            inter[i][j] = val
            inter[j][i] = val
    
    # Helper to check if adding index 'new_idx' to family (list of indices) is valid
    # We use the bucketing technique: group family members by their intersection with new_idx
    # For each group with common intersection I, check if any pair A, B in group has A&B == I
    def can_add(new_idx, family):
        m_new = masks[new_idx]
        # Bucket family members by intersection with new_idx
        buckets = {}
        for idx in family:
            inter_val = inter[new_idx][idx]
            if inter_val not in buckets:
                buckets[inter_val] = []
            buckets[inter_val].append(idx)
        
        # For each bucket, check if any pair has intersection equal to the bucket key
        for inter_val, members in buckets.items():
            if len(members) < 2:
                continue
            # Check pairs in this bucket
            for a_pos in range(len(members)):
                a_idx = members[a_pos]
                for b_pos in range(a_pos + 1, len(members)):
                    b_idx = members[b_pos]
                    if inter[a_idx][b_idx] == inter_val:
                        return False
        return True

    best_family = []
    best_size = 0
    random.seed(42)
    
    # Multiple random restarts with greedy construction
    # We also try a "smart" order: prioritize sets that are "far" from existing ones?
    # Or just random shuffles with different seeds/strategies.
    
    num_restarts = 0
    while time.time() < deadline - 5.0:
        num_restarts += 1
        # Random order
        order = list(range(total))
        random.shuffle(order)
        
        family = []
        for idx in order:
            if time.time() > deadline - 3.0:
                break
            if can_add(idx, family):
                family.append(idx)
        
        if len(family) > best_size:
            best_size = len(family)
            best_family = family[:]
            
            # If we've found a very large family, we can increase the deadline slightly
            # or just continue. Let's keep searching.
            # If best_size is very high (e.g., > 200), we might want to do local search.
            # But for now, let's stick to greedy restarts.
            
        # Early stopping if we've hit a theoretical upper bound?
        # The maximum size is unknown, but 250 is the cap in the problem statement.
        # If we reach 250, we can stop.
        if best_size >= 250:
            break
            
        # Occasionally, try a different strategy: 
        # Start with a random subset of the current best family and extend?
        # Or just keep shuffling.
        
    # Convert indices back to tuples of elements
    result = [sorted(all_subsets[idx]) for idx in best_family]
    return result
# EVOLVE-BLOCK-END
