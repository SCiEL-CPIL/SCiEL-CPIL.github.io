"""Covering design C(v,k,t) via optimized greedy set cover with precomputed indices."""
# EVOLVE-BLOCK-START
import itertools
import time
import random
import numpy as np


def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 120s limit

    def time_ok():
        return time.time() < deadline

    all_points = list(range(v))
    
    # Precompute all t-subsets and their indices
    all_t_subs = list(itertools.combinations(all_points, t))
    n_t = len(all_t_subs)
    t_sub_to_idx = {sub: i for i, sub in enumerate(all_t_subs)}
    
    # Precompute all k-blocks
    all_blocks = list(itertools.combinations(all_points, k))
    n_blocks = len(all_blocks)
    
    # Precompute for each block, the list of t-subset indices it covers
    # And for each t-subset, the list of block indices that cover it
    block_cover_indices = []
    t_sub_to_blocks = [[] for _ in range(n_t)]
    
    for b_idx, block in enumerate(all_blocks):
        covered_indices = []
        for t_sub in itertools.combinations(block, t):
            idx = t_sub_to_idx[t_sub]
            covered_indices.append(idx)
            t_sub_to_blocks[idx].append(b_idx)
        block_cover_indices.append(covered_indices)
    
    covered = np.zeros(n_t, dtype=np.uint8)
    uncovered_count = n_t
    selected_blocks = []
    
    # Greedy Phase
    while uncovered_count > 0 and time_ok():
        # Find an uncovered t-subset
        # Using np.where is fast enough for this size
        uncovered_indices = np.where(covered == 0)[0]
        if len(uncovered_indices) == 0:
            break
        
        # Pick a random uncovered t-subset
        # To improve quality, we could pick the one with fewest candidate blocks,
        # but random is standard and fast.
        anchor_idx = uncovered_indices[random.randint(0, len(uncovered_indices) - 1)]
        
        # Get all blocks covering this anchor t-subset
        candidates = t_sub_to_blocks[anchor_idx]
        
        best_block_idx = -1
        best_gain = -1
        
        # Evaluate gain for each candidate block
        for b_idx in candidates:
            gain = 0
            # Check how many of the t-subsets in this block are still uncovered
            for idx in block_cover_indices[b_idx]:
                if covered[idx] == 0:
                    gain += 1
            
            if gain > best_gain:
                best_gain = gain
                best_block_idx = b_idx
        
        if best_block_idx == -1:
            break
            
        # Add best block to solution
        selected_blocks.append(best_block_idx)
        
        # Update coverage
        for idx in block_cover_indices[best_block_idx]:
            if covered[idx] == 0:
                covered[idx] = 1
                uncovered_count -= 1
                
    # Convert block indices to actual blocks
    blocks = [list(all_blocks[i]) for i in selected_blocks]
    
    # Reduction Phase: Remove redundant blocks
    if blocks and time_ok():
        # We need the t-subset indices for the selected blocks
        selected_block_indices = [block_cover_indices[i] for i in selected_blocks]
        
        cover_count = np.zeros(n_t, dtype=np.int32)
        for subs in selected_block_indices:
            cover_count[subs] += 1
        
        changed = True
        while changed and time_ok():
            changed = False
            new_blocks = []
            new_selected_indices = []
            new_block_indices = []
            
            for i, subs in enumerate(selected_block_indices):
                # Check if any t-subset in this block has cover_count == 1
                keep = False
                for idx in subs:
                    if cover_count[idx] == 1:
                        keep = True
                        break
                if keep:
                    new_blocks.append(blocks[i])
                    new_selected_indices.append(selected_blocks[i])
                    new_block_indices.append(subs)
                else:
                    # Remove block
                    changed = True
                    cover_count[subs] -= 1
            
            blocks = new_blocks
            selected_blocks = new_selected_indices
            selected_block_indices = new_block_indices
            
            # Optional: Local Search / Swap
            # Try to swap a pair of blocks to reduce count? 
            # Given time constraints, simple reduction is safer.
            # If time remains, we could try random restarts or local search.
            # For now, stick to reduction.
    
    return blocks
# EVOLVE-BLOCK-END
