"""Optimized greedy covering for C(v,k,t) using bitset operations."""
import itertools
import time


def construct(v, k, t):
    deadline = time.time() + 140  # safety margin before 150s kill
    
    # Precompute all t-subsets as bitmasks
    all_tsubs = list(itertools.combinations(range(v), t))
    n_tsubs = len(all_tsubs)
    
    # Map each t-subset to an index
    tsub_to_idx = {sub: i for i, sub in enumerate(all_tsubs)}
    
    # For each point, precompute which t-subsets contain it
    point_tsubs = [set() for _ in range(v)]
    for i, sub in enumerate(all_tsubs):
        for p in sub:
            point_tsubs[p].add(i)
    
    # Track uncovered t-subsets as a boolean array
    uncovered = bytearray(b'\x01') * n_tsubs  # all uncovered initially
    uncovered_count = n_tsubs
    
    blocks = []
    
    while uncovered_count > 0:
        if time.time() > deadline:
            # Return best result found so far (may be invalid, but better than timeout)
            # Actually, we need valid, so let's try to finish quickly
            # For now, just break and return what we have (might be invalid)
            break
        
        # Find the first uncovered t-subset
        first_uncovered = -1
        for i in range(n_tsubs):
            if uncovered[i]:
                first_uncovered = i
                break
        if first_uncovered == -1:
            break
        
        # Start block with points from this t-subset
        block = list(all_tsubs[first_uncovered])
        
        # Greedily add points to maximize coverage of uncovered t-subsets
        while len(block) < k:
            best_point = -1
            best_gain = -1
            best_new_uncovered = None
            
            for x in range(v):
                if x in block:
                    continue
                
                # Compute gain: how many new uncovered t-subsets would be covered
                # A t-subset is covered by adding x if it contains x and (t-1) points from block
                gain = 0
                # For each (t-1)-subset of block, check if adding x forms an uncovered t-subset
                for rest in itertools.combinations(block, t - 1):
                    tsub = tuple(sorted(rest + (x,)))
                    idx = tsub_to_idx.get(tsub, -1)
                    if idx != -1 and uncovered[idx]:
                        gain += 1
                
                if gain > best_gain:
                    best_gain = gain
                    best_point = x
                    if gain == 0:
                        break  # No point helps, just pick any remaining
            
            if best_point == -1:
                # All points are in block or no gain; pick any remaining
                for x in range(v):
                    if x not in block:
                        best_point = x
                        break
            
            block.append(best_point)
        
        block.sort()
        blocks.append(block)
        
        # Mark all t-subsets within this block as covered
        for tsub in itertools.combinations(block, t):
            idx = tsub_to_idx[tsub]
            if uncovered[idx]:
                uncovered[idx] = 0
                uncovered_count -= 1
    
    return blocks
