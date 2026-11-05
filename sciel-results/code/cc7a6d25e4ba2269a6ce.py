"""Greedy covering with fast gain computation, iterative pruning, and deadline guard."""
import itertools
import time
import numpy as np

def construct(v, k, t):
    deadline = time.monotonic() + 115.0  # safety margin under 120s
    points = list(range(v))
    
    # Generate all t-subsets and assign them indices for fast lookup
    all_tsubs = list(itertools.combinations(points, t))
    tsub_to_idx = {ts: i for i, ts in enumerate(all_tsubs)}
    num_tsubs = len(all_tsubs)
    
    # Boolean array to track covered t-subsets
    covered = np.zeros(num_tsubs, dtype=bool)
    blocks = []
    
    # Precompute all possible k-blocks to avoid regenerating in loop?
    # C(19, 7) is ~50k. Generating them once is cheap.
    all_kblocks = list(itertools.combinations(points, k))
    num_kblocks = len(all_kblocks)
    
    # For each k-block, precompute the list of t-subset indices it covers.
    # This allows O(1) or O(C(k,t)) gain calculation instead of O(C(k,t) * lookup).
    block_tsubs = []
    for b in all_kblocks:
        b_tsubs = [tsub_to_idx[ts] for ts in itertools.combinations(b, t)]
        block_tsubs.append(b_tsubs)
    
    # Convert to numpy array for vectorized operations
    # Each block covers C(k,t) t-subsets. C(7,4)=35.
    t_per_block = len(block_tsubs[0])
    block_tsubs_arr = np.array(block_tsubs, dtype=np.int32)
    
    # Track which blocks have been selected
    selected = np.zeros(num_kblocks, dtype=bool)
    
    while True:
        if time.monotonic() > deadline:
            break
            
        # Find the block with maximum gain using vectorized operations
        # For each block, count how many of its t-subsets are not yet covered
        # gain[i] = sum of (not covered[idx]) for idx in block_tsubs[i]
        
        # We can do this by looking up covered values for all block-tsub pairs
        # block_tsubs_arr has shape (num_kblocks, t_per_block)
        # covered has shape (num_tsubs,)
        
        # Vectorized: look up covered values for all entries in block_tsubs_arr
        # This gives shape (num_kblocks, t_per_block)
        cover_status = covered[block_tsubs_arr]  # shape: (num_kblocks, t_per_block)
        
        # Count uncovered per block
        gains = (~cover_status).sum(axis=1)  # shape: (num_kblocks,)
        
        # Exclude already selected blocks
        gains[selected] = 0
        
        best_block_idx = int(np.argmax(gains))
        best_gain = int(gains[best_block_idx])
        
        if best_gain == 0:
            # No more blocks can cover anything, or all are covered
            break
            
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
