import itertools
import random
import time
import numpy as np

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 140.0  # Leave 10s margin for safety
    
    # Precompute all t-subsets and map them to indices
    t_subsets = list(itertools.combinations(range(v), t))
    t_sub_to_idx = {ts: i for i, ts in enumerate(t_subsets)}
    num_t_subsets = len(t_subsets)
    
    # Precompute all k-subsets (blocks)
    # For v=19, k=7, there are 50388 blocks. This is manageable.
    all_blocks = list(itertools.combinations(range(v), k))
    
    # Map each block to the list of t-subset indices it covers
    block_to_tsubs = []
    for b in all_blocks:
        indices = []
        for ts in itertools.combinations(b, t):
            indices.append(t_sub_to_idx[ts])
        block_to_tsubs.append(indices)
        
    # Helper to update cover counts
    # cover_count is a numpy array for speed
    cover_count = np.zeros(num_t_subsets, dtype=np.int32)
    
    def add_block(block_idx, count=1):
        for idx in block_to_tsubs[block_idx]:
            cover_count[idx] += count
            
    def remove_block(block_idx, count=1):
        for idx in block_to_tsubs[block_idx]:
            cover_count[idx] -= count
            
    def is_valid():
        # Check if any t-subset is uncovered
        return np.all(cover_count > 0)
        
    def is_redundant(block_idx):
        # A block is redundant if every t-subset it covers is covered by at least one other block
        # i.e., cover_count[idx] > 1 for all idx in block_to_tsubs[block_idx]
        # We can check this quickly
        for idx in block_to_tsubs[block_idx]:
            if cover_count[idx] == 1:
                return False
        return True

    # 1. Greedy Initialization
    # We want to pick blocks that cover the most uncovered t-subsets
    # We maintain a set of uncovered t-subsets for speed, or just use cover_count
    # Greedy:
    # Start with empty set. 
    # While not valid:
    #   Find block that maximizes number of t-subsets with cover_count == 0
    
    # Optimization: Keep track of uncovered count
    # Actually, let's just use the cover_count array.
    
    # To speed up greedy, we can pre-sort blocks or use a heap, but for v=19, 
    # simple iteration is likely fast enough for initialization.
    
    blocks_in_cover = []
    
    # Initial greedy
    while not is_valid():
        best_block_idx = -1
        max_gain = 0
        
        # Heuristic: Prioritize blocks that cover "rare" uncovered subsets?
        # Simple greedy: Maximize number of newly covered t-subsets (cover_count == 0)
        
        # We can speed this up by only checking blocks that have potential
        # But 50k blocks * 15 t-subsets * ~100 iterations = 75M ops. Might be slow in Python.
        
        # Let's use a smarter greedy:
        # Pick a random uncovered t-subset, then pick the block covering it 
        # that covers the most *other* uncovered t-subsets.
        
        # Find an uncovered t-subset
        uncovered_indices = np.where(cover_count == 0)[0]
        if len(uncovered_indices) == 0:
            break
            
        # Pick one randomly or the one that is "hardest" (fewest covering blocks)
        # For now, pick first one for simplicity, or random.
        target_idx = uncovered_indices[0]
        
        # Find all blocks covering this target
        # We need a reverse map: t_sub_idx -> list of block indices covering it
        # Let's build this reverse map once.
        
    # Re-do initialization with reverse map for speed
    
    # Build reverse map: t_sub_idx -> list of block indices
    t_sub_to_blocks = [[] for _ in range(num_t_subsets)]
    for b_idx, t_indices in enumerate(block_to_tsubs):
        for t_idx in t_indices:
            t_sub_to_blocks[t_idx].append(b_idx)
            
    # Greedy Initialization
    blocks_in_cover = []
    cover_count = np.zeros(num_t_subsets, dtype=np.int32)
    
    while not np.all(cover_count > 0):
        uncovered_indices = np.where(cover_count == 0)[0]
        if len(uncovered_indices) == 0:
            break
        
        # Pick a random uncovered t-subset
        target_idx = random.choice(uncovered_indices)
        
        # Candidates are blocks covering this target
        candidates = t_sub_to_blocks[target_idx]
        
        best_block_idx = -1
        max_gain = -1
        
        for b_idx in candidates:
            # Gain is number of t-subsets in this block with cover_count == 0
            gain = 0
            for t_idx in block_to_tsubs[b_idx]:
                if cover_count[t_idx] == 0:
                    gain += 1
            if gain > max_gain:
                max_gain = gain
                best_block_idx = b_idx
                
        if best_block_idx != -1:
            blocks_in_cover.append(best_block_idx)
            add_block(best_block_idx)
            
    # 2. Local Search
    # State: blocks_in_cover (list of block indices)
    # We will try to reduce the size.
    
    # Helper to check if we can remove a block
    def try_remove_block(b_idx):
        # Check if redundant
        if not is_redundant(b_idx):
            return False
        
        # Remove it
        remove_block(b_idx)
        blocks_in_cover.remove(b_idx)
        return True
        
    # Phase 1: Aggressive Deletion
    changed = True
    while changed:
        changed = False
        # Iterate through a copy of the list
        for b_idx in list(blocks_in_cover):
            if try_remove_block(b_idx):
                changed = True
                # Restart loop as indices might change? No, we use indices into all_blocks.
                # But removing one might make others redundant.
                # So we continue scanning.
                
    # Phase 2: Swap and Delete
    # We try to swap blocks to create redundancy.
    
    # To speed up swap evaluation:
    # We need to check if replacing B_old with B_new keeps cover_count > 0.
    # This is equivalent to: for all t_idx in B_old, if cover_count[t_idx] == 1, 
    # then B_new MUST cover t_idx.
    
    def can_swap(b_old_idx, b_new_idx):
        # Check if B_old has any "unique" t-subsets that B_new does not cover
        for t_idx in block_to_tsubs[b_old_idx]:
            if cover_count[t_idx] == 1:
                # This t-subset is only covered by B_old
                # So B_new must cover it
                if t_idx not in block_to_tsubs[b_new_idx]:
                    return False
        return True

    # We can precompute set of t-subsets for each block for faster membership check
    # block_to_tsub_set = [set(indices) for indices in block_to_tsubs]
    block_to_tsub_set = [set(indices) for indices in block_to_tsubs]
    
    def can_swap_fast(b_old_idx, b_new_idx):
        # If B_old has any unique t-subset, it must be in B_new
        # unique_tsubs = {t_idx for t_idx in block_to_tsubs[b_old_idx] if cover_count[t_idx] == 1}
        # return unique_tsubs.issubset(block_to_tsub_set[b_new_idx])
        
        # This is still somewhat slow. 
        # Let's stick to the loop but break early.
        for t_idx in block_to_tsubs[b_old_idx]:
            if cover_count[t_idx] == 1:
                if t_idx not in block_to_tsub_set[b_new_idx]:
                    return False
        return True

    # Optimization: Precompute "critical" t-subsets for each block?
    # cover_count changes, so it's dynamic.
    
    # Let's run the swap loop
    # We will try random swaps.
    # To be more effective, we can bias towards swapping blocks that are "heavy" 
    # (cover many unique subsets) with blocks that might cover them better?
    # Or just random.
    
    # We also need to ensure we don't swap a block with itself or a duplicate.
    # blocks_in_cover contains unique indices.
    
    while time.time() < deadline:
        # 1. Try to delete
        # We do a pass of deletion first
        deleted = False
        for b_idx in list(blocks_in_cover):
            if is_redundant(b_idx):
                remove_block(b_idx)
                blocks_in_cover.remove(b_idx)
                deleted = True
        
        if deleted:
            continue # Restart pass as state changed
            
        # 2. Try a swap
        if len(blocks_in_cover) < 2:
            break
            
        b_old_idx = random.choice(blocks_in_cover)
        
        # Choose b_new_idx
        # Random from all blocks, or biased?
        # Let's try random first.
        b_new_idx = random.randint(0, len(all_blocks) - 1)
        
        if b_new_idx == b_old_idx:
            continue
            
        # Check if b_new_idx is already in the cover
        # If it is, swapping is a no-op (remove one, add same one). 
        # Actually, if it's already there, we are just removing b_old and adding b_new.
        # If b_new is already in the list, we are effectively just removing b_old 
        # and keeping b_new (which was already there). 
        # This is equivalent to removing b_old if b_new covers the unique parts of b_old.
        # So the logic holds.
        
        if can_swap_fast(b_old_idx, b_new_idx):
            # Perform swap
            remove_block(b_old_idx)
            add_block(b_new_idx)
            
            # Update list
            blocks_in_cover.remove(b_old_idx)
            if b_new_idx not in blocks_in_cover:
                blocks_in_cover.append(b_new_idx)
            # else: it was already there, so we just removed b_old.
            # Wait, if b_new was already there, we removed b_old and added b_new (which is already there).
            # So the list size decreases by 1 if b_new was already there?
            # No.
            # List: [..., b_old, ..., b_new, ...]
            # Remove b_old: [..., b_new, ...]
            # Add b_new: [..., b_new, ...] (duplicate check fails, so not added)
            # So size decreases by 1. This is a valid deletion!
            
            # So the swap logic naturally handles deletion if the new block is already present.
            
            # After a swap, we might have created new redundancies.
            # The loop will go back to "Try to delete".

    # Convert block indices back to lists of points
    final_blocks = [list(all_blocks[i]) for i in blocks_in_cover]
    
    # Sort blocks for deterministic output (optional but good)
    final_blocks.sort()
    return final_blocks
