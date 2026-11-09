import numpy as np
from itertools import combinations

def construct(v, k, t):
    # Precompute all t-subsets and map them to indices
    t_subs = list(combinations(range(v), t))
    t_sub_idx = {sub: i for i, sub in enumerate(t_subs)}
    n_t = len(t_subs)
    
    # Precompute all k-subsets
    k_subs = list(combinations(range(v), k))
    n_k = len(k_subs)
    
    # Build mapping from k-subset to its t-subset indices
    # For each k-subset, find the indices of all its t-subsets
    c_k_t = len(list(combinations(range(k), t)))
    block_t_indices = np.zeros((n_k, c_k_t), dtype=np.int32)
    for i, ks in enumerate(k_subs):
        for j, ts in enumerate(combinations(ks, t)):
            block_t_indices[i, j] = t_sub_idx[ts]
    
    # --- Phase 1: Greedy Construction ---
    uncovered = np.ones(n_t, dtype=bool)
    n_uncovered = n_t
    active_blocks = []
    active_indices = []
    
    while n_uncovered > 0:
        # Compute gain for each block: number of currently uncovered t-subsets it covers
        # We only need to consider blocks not already selected
        # For efficiency, we can mask out already selected blocks, 
        # but since we are adding, we just pick the best from all, 
        # and if we pick a duplicate (which we shouldn't in greedy if we update uncovered), 
        # it's fine. But we must not pick the same block twice.
        
        # The seed code picks from ALL k_subs. 
        # If a block is already picked, its gain is 0 because its t-subsets are covered.
        # So np.argmax will naturally skip them if there are any uncovered left.
        
        gains = uncovered[block_t_indices].sum(axis=1)
        
        # Pick the block with maximum gain
        best_idx = np.argmax(gains)
        best_gain = gains[best_idx]
        
        if best_gain == 0:
            break
        
        active_indices.append(best_idx)
        
        # Update uncovered mask
        uncovered[block_t_indices[best_idx]] = False
        n_uncovered -= best_gain
    
    # --- Phase 2: Local Search Minimization ---
    # Initialize cover_count
    cover_count = np.zeros(n_t, dtype=np.int32)
    for idx in active_indices:
        cover_count[block_t_indices[idx]] += 1
        
    # is_active mask for the candidate pool (all k_subs)
    # But we only care about the ones in active_indices.
    # Let's work with the list active_indices.
    
    # We will iteratively remove blocks if they are redundant.
    # A block is redundant if all its t-subsets are covered by >1 block.
    
    # To speed up, we maintain a list of active block indices.
    current_active = list(active_indices)
    
    # Pre-compute for each block its t-subset indices for quick access
    # block_t_indices is already there.
    
    # Loop for minimization
    # We use a while loop that runs until no more blocks can be removed.
    # To avoid infinite loops or excessive time, we can limit iterations, 
    # but the number of blocks is small (< 500), so it should be fast.
    
    # Optimization: 
    # Instead of checking all blocks every time, we can just check.
    # With ~300 blocks, checking all is 300 * 56 = 16800 ops.
    # Even 300 iterations is ~5M ops. Very fast.
    
    while True:
        if not current_active:
            break
            
        # Calculate min cover count for each active block
        # Get indices of active blocks
        # block_t_indices[active_block_idx] gives (num_active, c_k_t)
        # We want min over axis 1 for each block.
        
        # Extract t-indices for current active blocks
        active_t_inds = block_t_indices[current_active]
        
        # Get cover counts for these t-subsets
        # cover_count[active_t_inds] -> shape (num_active, c_k_t)
        counts = cover_count[active_t_inds]
        
        # Min count for each block
        min_counts = counts.min(axis=1)
        
        # Find blocks where min_count > 1 (redundant)
        redundant_mask = min_counts > 1
        
        if not np.any(redundant_mask):
            break
        
        # Pick one redundant block to remove
        # Random choice to help escape local minima of the removal order
        redundant_indices = np.where(redundant_mask)[0]
        remove_pos = np.random.choice(redundant_indices)
        
        # The actual block index in k_subs
        block_idx = current_active[remove_pos]
        
        # Remove it
        # Update cover_count
        cover_count[block_t_indices[block_idx]] -= 1
        
        # Remove from current_active
        current_active.pop(remove_pos)
        
    # Return the remaining blocks
    return [list(k_subs[i]) for i in current_active]
