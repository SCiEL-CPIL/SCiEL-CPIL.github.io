"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import itertools
import time
import random
import numpy as np
from collections import defaultdict


def construct(v, k, t):
    """
    Construct a t-(v,k,1) covering design.
    Uses a vectorized greedy construction followed by local search refinement.
    """
    start_time = time.time()
    deadline = start_time + 115  # 115s safety margin within 120s limit

    # Precompute all t-subsets as integer indices for efficient tracking
    all_tsubs = list(itertools.combinations(range(v), t))
    n_tsubs = len(all_tsubs)
    
    # Map each t-subset to an index
    tsub_to_idx = {sub: i for i, sub in enumerate(all_tsubs)}
    
    # uncovered[i] = True if t-subset i is not yet covered
    uncovered = np.ones(n_tsubs, dtype=bool)
    
    # For each point, precompute which t-subsets it participates in
    # point_tsubs[p] = list of t-subset indices containing point p
    point_tsubs = [[] for _ in range(v)]
    for sub in all_tsubs:
        for p in sub:
            point_tsubs[p].append(tsub_to_idx[sub])
    
    # For each pair of points, precompute which t-subsets contain both
    # This is useful for gain calculation when t=3
    if t == 3:
        pair_tsubs = defaultdict(list)
        for sub in all_tsubs:
            for i in range(len(sub)):
                for j in range(i+1, len(sub)):
                    pair_tsubs[(sub[i], sub[j])].append(tsub_to_idx[sub])
    else:
        pair_tsubs = None
    
    blocks = []
    
    # Helper: compute gain of adding point x to current partial block
    def compute_gain(block, x):
        """Compute how many uncovered t-subsets would be covered by adding x to block."""
        if t == 1:
            return 1 if x not in block else 0
        
        if t == 2:
            # Adding x to block covers pairs (x, y) for y in block that are uncovered
            gain = 0
            for y in block:
                idx = tsub_to_idx.get(tuple(sorted((x, y))))
                if idx is not None and uncovered[idx]:
                    gain += 1
            return gain
        
        if t == 3:
            # Adding x to block covers triples (x, y, z) for y,z in block
            gain = 0
            for i in range(len(block)):
                for j in range(i+1, len(block)):
                    key = (min(block[i], block[j]), max(block[i], block[j]))
                    if key in pair_tsubs:
                        for idx in pair_tsubs[key]:
                            # Check if the triple (x, block[i], block[j]) is uncovered
                            # Actually, we need to check if (x, block[i], block[j]) is in uncovered
                            triple = tuple(sorted((x, block[i], block[j])))
                            idx2 = tsub_to_idx.get(triple)
                            if idx2 is not None and uncovered[idx2]:
                                gain += 1
            return gain
        
        # General case
        gain = 0
        for rest in itertools.combinations(block, t-1):
            sub = tuple(sorted(rest + (x,)))
            idx = tsub_to_idx.get(sub)
            if idx is not None and uncovered[idx]:
                gain += 1
        return gain
    
    # Helper: apply a block - mark its t-subsets as covered
    def apply_block(block):
        for sub in itertools.combinations(block, t):
            idx = tsub_to_idx[sub]
            if uncovered[idx]:
                uncovered[idx] = False
    
    # Greedy construction
    while np.any(uncovered):
        if time.time() > deadline - 30:
            break
        
        # Find the uncovered t-subset with minimum index (or first uncovered)
        uncov_indices = np.where(uncovered)[0]
        if len(uncov_indices) == 0:
            break
        first_uncov = all_tsubs[uncov_indices[0]]
        
        # Start block with points from the first uncovered t-subset
        block = list(first_uncov)
        
        # Greedily add points to maximize coverage
        while len(block) < k:
            if time.time() > deadline - 30:
                break
            
            # Find the point that maximizes gain
            best_gain = -1
            best_point = None
            for x in range(v):
                if x in block:
                    continue
                g = compute_gain(block, x)
                if g > best_gain:
                    best_gain = g
                    best_point = x
            
            if best_point is None or best_gain <= 0:
                break
            block.append(best_point)
        
        block.sort()
        blocks.append(block)
        apply_block(block)
    
    # Local search refinement: try to replace blocks with better ones
    # We'll try to find blocks that cover more uncovered t-subsets
    
    # Rebuild uncovered state
    uncovered = np.ones(n_tsubs, dtype=bool)
    for block in blocks:
        for sub in itertools.combinations(block, t):
            idx = tsub_to_idx[sub]
            uncovered[idx] = False
    
    # Try to improve by replacing blocks
    improved = True
    max_iterations = 50
    iteration = 0
    while improved and time.time() < deadline - 10 and iteration < max_iterations:
        improved = False
        iteration += 1
        
        # For each block, try to find a replacement that covers more
        for i in range(len(blocks)):
            if time.time() > deadline - 10:
                break
            
            # Temporarily remove block i
            temp_uncovered = uncovered.copy()
            for sub in itertools.combinations(blocks[i], t):
                idx = tsub_to_idx[sub]
                if not temp_uncovered[idx]:
                    # Check if this t-subset is covered by another block
                    covered_by_other = False
                    for j, block in enumerate(blocks):
                        if j == i:
                            continue
                        if all(p in block for p in sub):
                            covered_by_other = True
                            break
                    if not covered_by_other:
                        temp_uncovered[idx] = True
            
            # Now try to build a new block that covers more of temp_uncovered
            if not np.any(temp_uncovered):
                continue
            
            uncov_indices = np.where(temp_uncovered)[0]
            first_uncov = all_tsubs[uncov_indices[0]]
            new_block = list(first_uncov)
            
            while len(new_block) < k:
                if time.time() > deadline - 10:
                    break
                best_gain = -1
                best_point = None
                for x in range(v):
                    if x in new_block:
                        continue
                    g = 0
                    for rest in itertools.combinations(new_block, t-1):
                        sub = tuple(sorted(rest + (x,)))
                        idx = tsub_to_idx.get(sub)
                        if idx is not None and temp_uncovered[idx]:
                            g += 1
                    if g > best_gain:
                        best_gain = g
                        best_point = x
                if best_point is None or best_gain <= 0:
                    break
                new_block.append(best_point)
            
            new_block.sort()
            
            # Check if new block is better
            new_gain = 0
            for sub in itertools.combinations(new_block, t):
                idx = tsub_to_idx[sub]
                if temp_uncovered[idx]:
                    new_gain += 1
            
            old_gain = 0
            for sub in itertools.combinations(blocks[i], t):
                idx = tsub_to_idx[sub]
                if temp_uncovered[idx]:
                    old_gain += 1
            
            if new_gain > old_gain:
                blocks[i] = new_block
                improved = True
    
    # Rebuild final uncovered state
    uncovered = np.ones(n_tsubs, dtype=bool)
    for block in blocks:
        for sub in itertools.combinations(block, t):
            idx = tsub_to_idx[sub]
            uncovered[idx] = False
    
    # If not all covered, continue greedy
    while np.any(uncovered) and time.time() < deadline - 5:
        uncov_indices = np.where(uncovered)[0]
        if len(uncov_indices) == 0:
            break
        first_uncov = all_tsubs[uncov_indices[0]]
        block = list(first_uncov)
        
        while len(block) < k:
            if time.time() > deadline - 5:
                break
            best_gain = -1
            best_point = None
            for x in range(v):
                if x in block:
                    continue
                g = 0
                for rest in itertools.combinations(block, t-1):
                    sub = tuple(sorted(rest + (x,)))
                    idx = tsub_to_idx.get(sub)
                    if idx is not None and uncovered[idx]:
                        g += 1
                if g > best_gain:
                    best_gain = g
                    best_point = x
            if best_point is None or best_gain <= 0:
                break
            block.append(best_point)
        
        block.sort()
        blocks.append(block)
        for sub in itertools.combinations(block, t):
            idx = tsub_to_idx[sub]
            uncovered[idx] = False
    
    # Verify completeness - if not complete, we need to ensure validity
    # Final check: make sure all t-subsets are covered
    uncovered_final = np.ones(n_tsubs, dtype=bool)
    for block in blocks:
        for sub in itertools.combinations(block, t):
            idx = tsub_to_idx[sub]
            uncovered_final[idx] = False
    
    if np.any(uncovered_final):
        # Emergency: add blocks for remaining uncovered t-subsets
        uncov_indices = np.where(uncovered_final)[0]
        for idx in uncov_indices:
            sub = all_tsubs[idx]
            # Find a block containing this sub, or create one
            found = False
            for block in blocks:
                if all(p in block for p in sub):
                    found = True
                    break
            if not found:
                # Create a new block containing this sub
                new_block = list(sub)
                while len(new_block) < k:
                    x = random.randint(0, v-1)
                    if x not in new_block:
                        new_block.append(x)
                new_block.sort()
                blocks.append(new_block)
                for s2 in itertools.combinations(new_block, t):
                    idx2 = tsub_to_idx[s2]
                    uncovered_final[idx2] = False
    
    return blocks
# EVOLVE-BLOCK-END
