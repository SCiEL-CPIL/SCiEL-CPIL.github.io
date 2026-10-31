"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import itertools
import numpy as np
from collections import Counter


def _all_tsubsets(v, t):
    """Generate all t-subsets as tuples."""
    return list(itertools.combinations(range(v), t))


def _block_tsubsets(block, t):
    """Generate all t-subsets of a block."""
    return set(itertools.combinations(block, t))


def _greedy_construct(v, k, t, deadline, random_seed=0):
    """Vectorized greedy covering with randomization."""
    rng = np.random.default_rng(random_seed)
    
    # Precompute all t-subsets
    all_tsubs = list(itertools.combinations(range(v), t))
    tsub_to_idx = {ts: i for i, ts in enumerate(all_tsubs)}
    n_tsubs = len(all_tsubs)
    
    # Coverage tracking: which t-subsets are covered
    covered = np.zeros(n_tsubs, dtype=bool)
    blocks = []
    
    # For efficiency, precompute for each point the list of t-subsets containing it
    # point -> list of t-subset indices
    point_to_tsubs = [[] for _ in range(v)]
    for i, ts in enumerate(all_tsubs):
        for p in ts:
            point_to_tsubs[p].append(i)
    
    while not covered.all():
        # Find the first uncovered t-subset to start a new block
        uncovered_indices = np.where(~covered)[0]
        if len(uncovered_indices) == 0:
            break
        
        # Pick a random uncovered t-subset to seed the block
        start_idx = uncovered_indices[rng.integers(len(uncovered_indices))]
        start_tsub = all_tsubs[start_idx]
        block = list(start_tsub)
        
        # Greedily add points to maximize coverage
        for _ in range(k - len(block)):
            best_point = -1
            best_gain = -1
            
            for p in range(v):
                if p in block:
                    continue
                
                # Compute how many new t-subsets this point would cover
                gain = 0
                for rest in itertools.combinations(block, t - 1):
                    ts = tuple(sorted(rest + (p,)))
                    idx = tsub_to_idx[ts]
                    if not covered[idx]:
                        gain += 1
                
                if gain > best_gain:
                    best_gain = gain
                    best_point = p
            
            if best_point >= 0:
                block.append(best_point)
        
        block.sort()
        
        # Check if this block is redundant (all its t-subsets already covered)
        block_ts = _block_tsubsets(block, t)
        new_covered = sum(1 for ts in block_ts if not covered[tsub_to_idx[ts]])
        if new_covered == 0:
            continue
        
        blocks.append(block)
        
        # Update coverage
        for ts in block_ts:
            covered[tsub_to_idx[ts]] = True
        
        # Time check
        if time.time() > deadline:
            break
    
    return blocks


def _remove_redundant_blocks(blocks, v, k, t, deadline):
    """Remove blocks that don't contribute any unique t-subset coverage."""
    if not blocks:
        return blocks
    
    # Count how many blocks cover each t-subset
    all_tsubs = list(itertools.combinations(range(v), t))
    tsub_to_idx = {ts: i for i, ts in enumerate(all_tsubs)}
    n_tsubs = len(all_tsubs)
    
    # Count coverage per t-subset
    coverage_count = np.zeros(n_tsubs, dtype=int)
    for block in blocks:
        for ts in itertools.combinations(block, t):
            coverage_count[tsub_to_idx[ts]] += 1
    
    # A block is redundant if all its t-subsets are covered by other blocks
    # i.e., for every t-subset in the block, coverage_count > 1
    new_blocks = []
    for block in blocks:
        redundant = True
        for ts in itertools.combinations(block, t):
            if coverage_count[tsub_to_idx[ts]] == 1:
                redundant = False
                break
        if not redundant:
            new_blocks.append(block)
    
    return new_blocks


def _local_search(blocks, v, k, t, deadline, max_iterations=100):
    """Try to remove blocks or swap them to reduce count."""
    if not blocks:
        return blocks
    
    all_tsubs = list(itertools.combinations(range(v), t))
    tsub_to_idx = {ts: i for i, ts in enumerate(all_tsubs)}
    n_tsubs = len(all_tsubs)
    
    current_blocks = list(blocks)
    best_blocks = list(current_blocks)
    best_count = len(current_blocks)
    
    for iteration in range(max_iterations):
        if time.time() > deadline:
            break
        
        # Try removing each block and checking if coverage is maintained
        improved = False
        for i in range(len(current_blocks)):
            if time.time() > deadline:
                break
            
            # Check if removing block i maintains full coverage
            # Compute coverage without block i
            coverage = np.zeros(n_tsubs, dtype=bool)
            for j, block in enumerate(current_blocks):
                if j == i:
                    continue
                for ts in itertools.combinations(block, t):
                    coverage[tsub_to_idx[ts]] = True
            
            if coverage.all():
                # Block i is redundant
                current_blocks.pop(i)
                improved = True
                if len(current_blocks) < best_count:
                    best_count = len(current_blocks)
                    best_blocks = list(current_blocks)
                break
        
        if not improved:
            break
    
    return best_blocks


def construct(v, k, t):
    """Construct a covering design C(v, k, t)."""
    start_time = time.time()
    deadline = start_time + 140  # 140s deadline, 10s safety margin
    
    # Run multiple greedy restarts with different random seeds
    best_blocks = None
    best_count = float('inf')
    
    max_restarts = 50
    for seed in range(max_restarts):
        if time.time() > deadline - 30:  # Leave 30s for cleanup
            break
        
        blocks = _greedy_construct(v, k, t, deadline, random_seed=seed)
        
        if not blocks:
            continue
        
        # Remove redundant blocks
        blocks = _remove_redundant_blocks(blocks, v, k, t, deadline)
        
        if len(blocks) < best_count:
            best_count = len(blocks)
            best_blocks = blocks
        
        # If we've hit a good solution, do local search
        if best_blocks is not None and time.time() < deadline - 20:
            improved = _local_search(best_blocks, v, k, t, deadline - 5)
            if len(improved) < best_count:
                best_count = len(improved)
                best_blocks = improved
    
    if best_blocks is None:
        # Fallback: single greedy run
        best_blocks = _greedy_construct(v, k, t, deadline, random_seed=0)
    
    # Final check: ensure we return a valid covering if possible
    # If time is running out, just return what we have
    if time.time() > deadline - 5:
        return best_blocks
    
    # One more attempt at redundancy removal on the best solution
    final_blocks = _remove_redundant_blocks(best_blocks, v, k, t, deadline)
    
    return final_blocks
# EVOLVE-BLOCK-END
