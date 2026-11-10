"""Optimized covering design C(19,8,5) using bitmask-based fast greedy with randomized restarts."""
# EVOLVE-BLOCK-START
import time
import random
import itertools
import numpy as np

def _precompute(v, k, t):
    """Precompute all t-subsets, point mappings, and bitmask lookups."""
    all_tsubs = list(itertools.combinations(range(v), t))
    n_tsubs = len(all_tsubs)
    
    # Map each t-subset tuple to its index
    tsub_to_idx = {ts: i for i, ts in enumerate(all_tsubs)}
    
    # For each point, list of t-subset indices containing that point
    point_tsubs = [[] for _ in range(v)]
    for i, ts in enumerate(all_tsubs):
        for p in ts:
            point_tsubs[p].append(i)
    
    # For each k-subset, precompute the bitmask of t-subsets it covers
    # This is expensive but we only need it for candidate evaluation
    # Instead, we compute coverage on-the-fly using the point_tsubs structure
    
    return all_tsubs, tsub_to_idx, point_tsubs, n_tsubs

def _greedy_cover(v, k, t, seed=42, max_time=30.0, randomize=True):
    """
    Fast greedy covering using bitmask operations.
    """
    rng = random.Random(seed)
    start_time = time.time()
    deadline = start_time + max_time
    
    all_tsubs, tsub_to_idx, point_tsubs, n_tsubs = _precompute(v, k, t)
    
    # covered[i] = True if t-subset i is covered
    covered = np.zeros(n_tsubs, dtype=bool)
    num_uncovered = n_tsubs
    
    blocks = []
    seen_blocks = set()
    
    while num_uncovered > 0:
        if time.time() > deadline:
            break
        
        # Find an uncovered t-subset to focus on
        uncovered_indices = np.where(~covered)[0]
        if len(uncovered_indices) == 0:
            break
        
        # Sample some uncovered t-subsets as seeds for candidate blocks
        if randomize and len(uncovered_indices) > 10:
            sample_size = min(10, len(uncovered_indices))
            sample = rng.sample(list(uncovered_indices), sample_size)
        else:
            sample = list(uncovered_indices[:10])
        
        best_block = None
        best_gain = -1
        best_newly_covered = None
        
        for idx in sample:
            focus = all_tsubs[idx]
            block_set = set(focus)
            
            # Greedily add points to maximize uncovered t-subsets covered
            while len(block_set) < k:
                best_point = None
                best_local_gain = -1
                
                for x in range(v):
                    if x in block_set:
                        continue
                    
                    # Count uncovered t-subsets that would become covered by adding x
                    # A t-subset is covered if all its elements are in block_set ∪ {x}
                    # It must contain x and (t-1) elements from block_set
                    gain = 0
                    block_list = list(block_set)
                    for rest in itertools.combinations(block_list, t - 1):
                        ts = tuple(sorted(rest + (x,)))
                        i = tsub_to_idx.get(ts, -1)
                        if i != -1 and not covered[i]:
                            gain += 1
                    
                    if gain > best_local_gain:
                        best_local_gain = gain
                        best_point = x
                    elif gain == best_local_gain and best_point is not None and randomize and rng.random() < 0.3:
                        best_point = x
                
                if best_point is None:
                    break
                block_set.add(best_point)
            
            block = tuple(sorted(block_set))
            
            if block in seen_blocks:
                continue
            
            # Count how many currently uncovered t-subsets this block covers
            newly_covered = []
            for ts in itertools.combinations(block, t):
                i = tsub_to_idx.get(ts, -1)
                if i != -1 and not covered[i]:
                    newly_covered.append(i)
            
            gain = len(newly_covered)
            
            if gain > best_gain:
                best_gain = gain
                best_block = block
                best_newly_covered = newly_covered
            elif gain == best_gain and best_block is not None and randomize and rng.random() < 0.2:
                best_block = block
                best_newly_covered = newly_covered
        
        if best_block is None:
            break
        
        seen_blocks.add(best_block)
        blocks.append(list(best_block))
        
        for i in best_newly_covered:
            if not covered[i]:
                covered[i] = True
                num_uncovered -= 1
    
    return blocks

def _remove_redundant(blocks, v, k, t, max_time=10.0):
    """Remove blocks whose t-subsets are all covered by other blocks."""
    start_time = time.time()
    deadline = start_time + max_time
    
    all_tsubs, tsub_to_idx, point_tsubs, n_tsubs = _precompute(v, k, t)
    
    # Count coverage of each t-subset
    tsub_count = np.zeros(n_tsubs, dtype=np.int32)
    for block in blocks:
        for ts in itertools.combinations(block, t):
            i = tsub_to_idx.get(ts, -1)
            if i != -1:
                tsub_count[i] += 1
    
    improved = True
    while improved:
        if time.time() > deadline:
            break
        improved = False
        for i in range(len(blocks) - 1, -1, -1):
            if time.time() > deadline:
                break
            block = blocks[i]
            redundant = True
            for ts in itertools.combinations(block, t):
                idx = tsub_to_idx.get(ts, -1)
                if idx != -1 and tsub_count[idx] <= 1:
                    redundant = False
                    break
            if redundant:
                for ts in itertools.combinations(block, t):
                    idx = tsub_to_idx.get(ts, -1)
                    if idx != -1:
                        tsub_count[idx] -= 1
                blocks.pop(i)
                improved = True
    
    return blocks

def construct(v, k, t):
    """
    Construct a covering design C(v,k,t).
    Returns a list of blocks, each a list of k distinct integers in 0..v-1.
    """
    total_start = time.time()
    total_deadline = total_start + 145.0  # Leave 5s safety margin before 150s kill
    
    best_blocks = None
    best_len = float('inf')
    
    # Run multiple greedy strategies with different seeds
    seeds = [1, 42, 123, 777, 2024, 999, 555, 314, 271, 1618, 31415, 27182, 16180, 14142, 17320]
    
    for i, seed in enumerate(seeds):
        if time.time() > total_deadline:
            break
        
        remaining = total_deadline - time.time()
        if remaining < 5.0:
            break
        
        # Allocate time: more for early runs, less for later
        if i < 3:
            alloc = min(remaining * 0.4, 20.0)
        elif i < 8:
            alloc = min(remaining * 0.3, 15.0)
        else:
            alloc = min(remaining * 0.25, 12.0)
        
        blocks = _greedy_cover(v, k, t, seed=seed, max_time=alloc, randomize=True)
        
        if blocks and len(blocks) < best_len:
            best_len = len(blocks)
            best_blocks = blocks
    
    if best_blocks is None:
        return []
    
    # Local refinement: remove redundant blocks
    if time.time() < total_deadline - 3.0:
        refine_time = min(total_deadline - time.time() - 3.0, 15.0)
        best_blocks = _remove_redundant(best_blocks, v, k, t, max_time=refine_time)
    
    return best_blocks
# EVOLVE-BLOCK-END
