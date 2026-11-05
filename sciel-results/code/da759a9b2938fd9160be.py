"""Optimized covering design C(24,7,3) solver using greedy + local search."""
import itertools
import random
import time
from collections import defaultdict

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 115.0  # 115s safety margin within 120s limit
    
    # Precompute all t-subsets
    all_tsubsets = list(itertools.combinations(range(v), t))
    n_tsubsets = len(all_tsubsets)
    
    # Map each t-subset to an index
    tsubset_to_idx = {ts: i for i, ts in enumerate(all_tsubsets)}
    
    # Precompute for each point, the list of (t-1)-subsets it can complete
    # For each point x and each (t-1)-subset s, the t-subset is s + (x,)
    point_tsubsets = defaultdict(list)
    for x in range(v):
        for s in itertools.combinations(range(v), t - 1):
            if x not in s:
                ts = tuple(sorted(s + (x,)))
                point_tsubsets[x].append(tsubset_to_idx[ts])
    
    # Greedy set cover
    uncovered = [1] * n_tsubsets  # 1 means uncovered
    blocks = []
    
    while any(uncovered):
        if time.time() > deadline - 5:
            break  # Safety: stop if running out of time
        
        # Find the first uncovered t-subset to anchor the block
        first_uncovered = None
        for i in range(n_tsubsets):
            if uncovered[i]:
                first_uncovered = i
                break
        
        if first_uncovered is None:
            break
        
        anchor = all_tsubsets[first_uncovered]
        block = set(anchor)
        
        # Greedily add points that cover the most uncovered t-subsets
        while len(block) < k:
            best_point = None
            best_gain = -1
            
            for x in range(v):
                if x in block:
                    continue
                
                gain = 0
                for ts_idx in point_tsubsets[x]:
                    if uncovered[ts_idx]:
                        # Check if the t-subset contains all but x from block
                        # i.e., the (t-1)-subset of the t-subset that excludes x must be subset of block
                        ts = all_tsubsets[ts_idx]
                        # The (t-1)-subset without x
                        rest = tuple(p for p in ts if p != x)
                        if all(p in block for p in rest):
                            gain += 1
                
                if gain > best_gain:
                    best_gain = gain
                    best_point = x
            
            if best_point is None or best_gain <= 0:
                # Pick a random point not in block
                candidates = [x for x in range(v) if x not in block]
                if candidates:
                    best_point = random.choice(candidates)
                else:
                    break
            
            block.add(best_point)
        
        block = sorted(block)
        blocks.append(block)
        
        # Mark covered t-subsets
        for ts in itertools.combinations(block, t):
            idx = tsubset_to_idx[tuple(ts)]
            uncovered[idx] = 0
    
    # Local search: try to remove blocks
    if time.time() > deadline - 10:
        return blocks
    
    # Try to remove each block if its t-subsets are still covered by other blocks
    changed = True
    while changed and time.time() < deadline - 10:
        changed = False
        for i in range(len(blocks) - 1, -1, -1):
            if time.time() > deadline - 10:
                break
            
            # Check if removing block i still covers all its t-subsets
            removable = True
            block_i = blocks[i]
            for ts in itertools.combinations(block_i, t):
                idx = tsubset_to_idx[tuple(ts)]
                if uncovered[idx] == 0:
                    # This t-subset is covered; check if another block covers it
                    found_in_other = False
                    for j in range(len(blocks)):
                        if j == i:
                            continue
                        block_j = blocks[j]
                        if all(p in block_j for p in ts):
                            found_in_other = True
                            break
                    if not found_in_other:
                        removable = False
                        break
            
            if removable:
                blocks.pop(i)
                changed = True
    
    # Local search: try to swap blocks for better coverage
    if time.time() > deadline - 15:
        return blocks
    
    # Try to find better blocks by replacing existing ones
    for iteration in range(20):
        if time.time() > deadline - 15:
            break
        
        improved = False
        for i in range(len(blocks)):
            if time.time() > deadline - 15:
                break
            
            # Try to find a better replacement for block i
            # Count how many t-subsets only this block covers
            unique_count = 0
            for ts in itertools.combinations(blocks[i], t):
                idx = tsubset_to_idx[tuple(ts)]
                if uncovered[idx] == 0:
                    found_in_other = False
                    for j in range(len(blocks)):
                        if j == i:
                            continue
                        if all(p in blocks[j] for p in ts):
                            found_in_other = True
                            break
                    if not found_in_other:
                        unique_count += 1
            
            # If no unique coverage, this block is redundant
            if unique_count == 0:
                blocks.pop(i)
                improved = True
                break
        
        if not improved:
            break
    
    return blocks
