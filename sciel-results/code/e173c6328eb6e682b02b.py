"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import itertools
import time
import random
from collections import defaultdict

def construct(v, k, t):
    deadline = time.time() + 145  # 145s deadline for 150s limit
    
    def time_left():
        return deadline - time.time()
    
    # Pre-compute all t-subsets and map them to indices
    all_tsubs = list(itertools.combinations(range(v), t))
    num_tsubs = len(all_tsubs)
    tsub_to_idx = {tsub: i for i, tsub in enumerate(all_tsubs)}
    
    # Pre-compute all k-subsets (blocks) and their coverage masks
    # Note: For v=19, k=7, there are 50388 blocks. 
    # We can represent the set of covered t-subsets as an integer bitmask.
    # Python integers are arbitrary precision, so we can use them as bitsets.
    
    all_blocks = list(itertools.combinations(range(v), k))
    block_masks = []
    for block in all_blocks:
        mask = 0
        # For each t-subset within this block, set the corresponding bit
        # To speed this up, we can iterate over t-subsets of the block
        for tsub in itertools.combinations(block, t):
            mask |= (1 << tsub_to_idx[tsub])
        block_masks.append(mask)
    
    # To speed up greedy, we might want to sort blocks or just iterate.
    # Iterating 50k blocks * ~150 steps * 150s budget is feasible if operations are fast.
    # popcount is fast in Python 3.8+ via int.bit_count()
    
    best_blocks = None
    best_count = float('inf')
    
    # We will run multiple randomized greedy passes
    # Randomness comes from tie-breaking or slightly permuting the order of blocks considered
    # Or simply re-initializing and using random tie-breaking.
    
    max_restarts = 20
    for restart in range(max_restarts):
        if time_left() < 10:
            break
            
        random.seed(restart) # Reproducible restarts for debugging, or just use global random
        
        uncovered_mask = (1 << num_tsubs) - 1
        selected_blocks = []
        selected_indices = set()
        
        # Greedy Step
        while uncovered_mask != 0:
            if time_left() < 5:
                break
                
            best_gain = -1
            best_block_idx = -1
            best_block = None
            
            # Iterate over all possible blocks
            # Optimization: We can skip blocks that are subsets of already selected blocks? No, blocks are same size.
            # We can skip blocks that don't add new coverage.
            
            # To add randomness and escape local optima, we can shuffle the block order occasionally
            # But shuffling 50k items every step is expensive.
            # Instead, we can just pick the best one. Ties are broken by index, which is deterministic.
            # To introduce randomness, we can add a small random weight to the gain?
            # Or simply: if gain == best_gain, pick randomly among candidates?
            # That requires storing candidates.
            
            # Let's stick to deterministic greedy first, but use different "perturbations"
            # Actually, pure greedy often gets stuck. 
            # A common trick: if multiple blocks have the same max gain, pick one at random.
            
            candidates = []
            for i, block in enumerate(all_blocks):
                if i in selected_indices:
                    continue
                mask = block_masks[i]
                # Count how many uncovered t-subsets this block covers
                # gain = popcount(mask & uncovered_mask)
                gain = (mask & uncovered_mask).bit_count()
                
                if gain > best_gain:
                    best_gain = gain
                    candidates = [i]
                elif gain == best_gain and gain > 0:
                    candidates.append(i)
            
            if not candidates:
                # Should not happen if a solution exists
                break
            
            # Pick a random candidate from those with max gain
            chosen_idx = random.choice(candidates)
            selected_blocks.append(all_blocks[chosen_idx])
            selected_indices.add(chosen_idx)
            uncovered_mask &= ~block_masks[chosen_idx]
            
        if not selected_blocks:
            continue
            
        # Local Search: Remove redundant blocks
        # A block is redundant if all t-subsets it covers are also covered by other selected blocks.
        # We need to track coverage counts for the selected set.
        
        # Rebuild coverage map for the selected blocks
        # cover_count[tsub_idx] = number of selected blocks covering it
        cover_counts = [0] * num_tsubs
        for block in selected_blocks:
            # We need the mask for this block to update counts?
            # It's faster to iterate tsubs of the block
            for tsub in itertools.combinations(block, t):
                cover_counts[tsub_to_idx[tsub]] += 1
        
        changed = True
        while changed:
            if time_left() < 5:
                break
            changed = False
            for i in range(len(selected_blocks) - 1, -1, -1):
                block = selected_blocks[i]
                redundant = True
                for tsub in itertools.combinations(block, t):
                    idx = tsub_to_idx[tsub]
                    if cover_counts[idx] == 1:
                        redundant = False
                        break
                if redundant:
                    # Remove block i
                    # Update cover_counts
                    for tsub in itertools.combinations(block, t):
                        idx = tsub_to_idx[tsub]
                        cover_counts[idx] -= 1
                    selected_blocks.pop(i)
                    changed = True
                    break # Restart scan as indices shifted
        
        if len(selected_blocks) < best_count:
            best_count = len(selected_blocks)
            best_blocks = list(selected_blocks)
            
            # Try further optimization?
            # Simple swap: try replacing a block with another that covers the unique parts
            # This is complex to implement efficiently in the remaining time.
            # The redundancy removal is the main gain.
            
    if best_blocks is None:
        # Fallback: return empty or a trivial invalid result? 
        # The problem guarantees a solution exists.
        # If we timed out before finishing even one greedy, we might have an incomplete cover.
        # The contract says "Valid when every t-subset ... lies in at least one block".
        # If we return an incomplete cover, it's invalid (score 0).
        # We must ensure we return a *complete* cover.
        # The greedy loop above runs until uncovered_mask == 0.
        # If time runs out *during* greedy, uncovered_mask != 0.
        # In that case, we should complete the cover greedily without time checks?
        # Or just return what we have? No, it must be valid.
        
        # Let's add a safety net: if best_blocks is None, do a final fast greedy without local search
        if time_left() > 1:
            uncovered_mask = (1 << num_tsubs) - 1
            selected_blocks = []
            while uncovered_mask != 0:
                best_gain = -1
                best_idx = -1
                for i, block in enumerate(all_blocks):
                    if i in selected_indices: # selected_indices from last loop is stale, use set
                        continue
                    gain = (block_masks[i] & uncovered_mask).bit_count()
                    if gain > best_gain:
                        best_gain = gain
                        best_idx = i
                if best_idx == -1:
                    break
                selected_blocks.append(all_blocks[best_idx])
                uncovered_mask &= ~block_masks[best_idx]
            best_blocks = selected_blocks
        else:
            # Extreme timeout, return empty? No, must be valid.
            # This shouldn't happen if we have 145s.
            best_blocks = []
            
    return best_blocks

# EVOLVE-BLOCK-END
