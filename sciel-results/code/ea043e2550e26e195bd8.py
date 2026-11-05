"""Greedy covering with fast gain computation, iterative pruning, and deadline guard."""
import itertools
import time
import numpy as np
import random

def construct(v, k, t):
    deadline = time.monotonic() + 115.0  # safety margin under 120s
    points = list(range(v))
    
    # Generate all t-subsets and assign them indices for fast lookup
    all_tsubs = list(itertools.combinations(points, t))
    tsub_to_idx = {ts: i for i, ts in enumerate(all_tsubs)}
    num_tsubs = len(all_tsubs)
    
    # Precompute all possible k-blocks
    all_kblocks = list(itertools.combinations(points, k))
    num_kblocks = len(all_kblocks)
    
    # For each k-block, precompute the list of t-subset indices it covers
    block_tsubs = []
    for b in all_kblocks:
        b_tsubs = [tsub_to_idx[ts] for ts in itertools.combinations(b, t)]
        block_tsubs.append(b_tsubs)
    
    # Convert to numpy array for vectorized operations
    t_per_block = len(block_tsubs[0])
    block_tsubs_arr = np.array(block_tsubs, dtype=np.int32)
    
    def greedy_solve(seed=None):
        """Run a greedy covering algorithm, optionally with randomized tie-breaking."""
        if seed is not None:
            rng = random.Random(seed)
        else:
            rng = None
        
        covered = np.zeros(num_tsubs, dtype=bool)
        blocks = []
        selected = np.zeros(num_kblocks, dtype=bool)
        
        while True:
            if time.monotonic() > deadline:
                break
            
            # Find the block with maximum gain using vectorized operations
            cover_status = covered[block_tsubs_arr]  # shape: (num_kblocks, t_per_block)
            gains = (~cover_status).sum(axis=1)  # shape: (num_kblocks,)
            
            # Exclude already selected blocks
            gains[selected] = 0
            
            max_gain = int(gains.max())
            if max_gain == 0:
                break
            
            # Find all blocks with maximum gain
            best_indices = np.where(gains == max_gain)[0]
            
            if len(best_indices) == 1 or rng is None:
                best_block_idx = int(best_indices[0])
            else:
                # Random tie-breaking
                best_block_idx = int(best_indices[rng.randint(0, len(best_indices) - 1)])
            
            # Select the best block
            best_block = all_kblocks[best_block_idx]
            blocks.append(list(best_block))
            selected[best_block_idx] = True
            
            # Mark its t-subsets as covered
            for idx in block_tsubs[best_block_idx]:
                covered[idx] = True
            
            # Check if all covered
            if covered.all():
                break
        
        return blocks
    
    # First run: deterministic greedy
    best_blocks = greedy_solve(seed=None)
    best_len = len(best_blocks)
    
    # If we have time, try randomized restarts to potentially find better solutions
    if time.monotonic() < deadline - 10.0:
        for attempt in range(1, 20):
            if time.monotonic() > deadline - 5.0:
                break
            
            # Try different random seeds
            seed = 42 + attempt
            blocks = greedy_solve(seed=seed)
            
            if len(blocks) < best_len:
                best_blocks = blocks
                best_len = len(blocks)
    
    return best_blocks
