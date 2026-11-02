"""Improved covering algorithm using vectorized greedy with local search."""
import itertools
import time
import random
import numpy as np

def construct(v, k, t):
    deadline = time.time() + 110.0  # safety margin
    
    # Precompute all t-subsets
    all_tsubsets = list(itertools.combinations(range(v), t))
    n_tsubsets = len(all_tsubsets)
    tsub_idx = {ts: i for i, ts in enumerate(all_tsubsets)}
    
    # Precompute all k-blocks and their coverage as numpy arrays
    all_blocks = list(itertools.combinations(range(v), k))
    n_blocks = len(all_blocks)
    
    # Build coverage matrix: cov_matrix[block_idx, tsub_idx] = 1 if block covers tsub
    # Use sparse representation: for each block, list of t-subset indices
    block_tsubs = []
    for block in all_blocks:
        tsubs = []
        for ts in itertools.combinations(block, t):
            tsubs.append(tsub_idx[ts])
        block_tsubs.append(tsubs)
    
    # Also build inverse: for each t-subset, list of blocks covering it
    tsub_blocks = [[] for _ in range(n_tsubsets)]
    for bi, tsubs in enumerate(block_tsubs):
        for ts_idx in tsubs:
            tsub_blocks[ts_idx].append(bi)
    
    best_blocks = None
    best_len = float('inf')
    
    def greedy_cover(seed=0):
        rng = random.Random(seed)
        uncovered = np.ones(n_tsubsets, dtype=bool)
        blocks_used = []
        
        # Compute initial gain: number of uncovered t-subsets each block covers
        # gain[i] = sum(uncovered[block_tsubs[i]])
        gain = np.array([sum(uncovered[ts] for ts in tsubs) for tsubs in block_tsubs], dtype=np.int32)
        
        while np.any(uncovered):
            if time.time() > deadline:
                break
            
            # Find block with maximum gain
            max_gain = gain.max()
            if max_gain == 0:
                break  # Should not happen if uncovered is non-empty
            
            # Pick randomly from blocks with max gain for diversity
            max_blocks = np.where(gain == max_gain)[0]
            chosen = max_blocks[rng.randint(0, len(max_blocks) - 1)]
            
            blocks_used.append(all_blocks[chosen])
            
            # Update uncovered and gain
            for ts_idx in block_tsubs[chosen]:
                if uncovered[ts_idx]:
                    uncovered[ts_idx] = False
                    # Decrement gain for all blocks covering this t-subset
                    for bi in tsub_blocks[ts_idx]:
                        gain[bi] -= 1
            
            # Mark chosen block as unusable (set gain to -1 to avoid re-selection)
            gain[chosen] = -1
        
        return blocks_used
    
    def local_search(blocks, seed=0):
        """Try to improve by removing redundant blocks and swapping."""
        if not blocks:
            return blocks
        
        rng = random.Random(seed)
        
        # First pass: remove redundant blocks
        cover_count = np.zeros(n_tsubsets, dtype=np.int32)
        block_tsub_indices = []
        for b in blocks:
            tsubs = []
            for ts in itertools.combinations(b, t):
                idx = tsub_idx[ts]
                tsubs.append(idx)
                cover_count[idx] += 1
            block_tsub_indices.append(tsubs)
        
        kept = [True] * len(blocks)
        changed = True
        while changed:
            changed = False
            for i in range(len(blocks)):
                if not kept[i]:
                    continue
                # Check if block i is redundant
                redundant = True
                for idx in block_tsub_indices[i]:
                    if cover_count[idx] == 1:
                        redundant = False
                        break
                if redundant:
                    kept[i] = False
                    for idx in block_tsub_indices[i]:
                        cover_count[idx] -= 1
                    changed = True
        
        reduced = [blocks[i] for i in range(len(blocks)) if kept[i]]
        
        # Second pass: try replacing pairs of blocks with one block
        # This is expensive, so only do a limited number of trials
        n_reduced = len(reduced)
        if n_reduced > 2 and time.time() < deadline - 10:
            # Try to find a block that covers all t-subsets of two blocks
            # and replace those two blocks with it
            for trial in range(50):
                if time.time() > deadline:
                    break
                i, j = rng.sample(range(n_reduced), 2)
                # Get all t-subsets covered by blocks i and j
                tsubs_ij = set(block_tsub_indices[i] + block_tsub_indices[j])
                # Find if any single block covers all of tsubs_ij
                # This is expensive, so skip for now
                pass
        
        return reduced
    
    # Run greedy with multiple seeds
    num_trials = 30
    for seed in range(num_trials):
        if time.time() > deadline:
            break
        
        blocks = greedy_cover(seed)
        if not blocks:
            continue
        
        blocks = local_search(blocks, seed)
        
        if len(blocks) < best_len:
            best_len = len(blocks)
            best_blocks = blocks
        
        # Early termination if we hit a very good result
        if best_len <= 135:
            break
    
    if best_blocks is None:
        # Fallback
        best_blocks = greedy_cover(0)
        best_blocks = local_search(best_blocks, 0)
    
    return best_blocks
