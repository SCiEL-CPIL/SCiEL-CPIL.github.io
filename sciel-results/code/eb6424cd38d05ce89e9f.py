"""Covering design C(19,8,5) - Global Bitmask Greedy with Local Search."""
import itertools
import time
import random
import sys
import numpy as np

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin

    # 1. Generate all t-subsets and assign indices
    all_t_subs = list(itertools.combinations(range(v), t))
    n_subs = len(all_t_subs)
    
    # Map subset tuple to index
    sub_to_idx = {sub: i for i, sub in enumerate(all_t_subs)}

    # 2. Generate all k-blocks
    all_blocks = list(itertools.combinations(range(v), k))
    n_blocks = len(all_blocks)
    
    # 3. Precompute bitmask for each block
    block_masks = [0] * n_blocks
    for b_idx, block in enumerate(all_blocks):
        mask = 0
        for t_sub in itertools.combinations(block, t):
            idx = sub_to_idx[t_sub]
            mask |= (1 << idx)
        block_masks[b_idx] = mask

    # Precompute bit positions for each block mask (list of t-subset indices covered)
    block_bit_lists = []
    for b_idx in range(n_blocks):
        mask = block_masks[b_idx]
        bits = []
        m = mask
        while m:
            idx = (m & -m).bit_length() - 1
            bits.append(idx)
            m &= m - 1
        block_bit_lists.append(bits)

    best_blocks = None
    best_count = float('inf')
    
    rng = random.Random(42)
    max_restarts = 200
    
    for restart in range(max_restarts):
        if time.time() > deadline:
            break
            
        uncovered_mask = (1 << n_subs) - 1
        selected_blocks = []
        active_blocks = list(range(n_blocks))
        
        while uncovered_mask:
            if time.time() > deadline:
                break
                
            best_gain = 0
            best_block_candidates = []
            
            new_active_blocks = []
            for b_idx in active_blocks:
                mask = block_masks[b_idx]
                gain = (mask & uncovered_mask).bit_count()
                if gain > 0:
                    new_active_blocks.append(b_idx)
                    if gain > best_gain:
                        best_gain = gain
                        best_block_candidates = [b_idx]
                    elif gain == best_gain:
                        best_block_candidates.append(b_idx)
                        
            if not best_block_candidates:
                break
                
            chosen_block = rng.choice(best_block_candidates)
            selected_blocks.append(chosen_block)
            uncovered_mask &= ~block_masks[chosen_block]
            active_blocks = new_active_blocks
            
            if not active_blocks:
                break

        if uncovered_mask == 0:
            # Local Search Phase 1: Delete redundant blocks
            cover_counts = [0] * n_subs
            for b_idx in selected_blocks:
                for idx in block_bit_lists[b_idx]:
                    cover_counts[idx] += 1
            
            changed = True
            while changed:
                changed = False
                for i in range(len(selected_blocks)):
                    b_idx = selected_blocks[i]
                    is_redundant = True
                    for idx in block_bit_lists[b_idx]:
                        if cover_counts[idx] == 1:
                            is_redundant = False
                            break
                    if is_redundant:
                        for idx in block_bit_lists[b_idx]:
                            cover_counts[idx] -= 1
                        selected_blocks.pop(i)
                        changed = True
                        break

            # Local Search Phase 2: Try swapping blocks to reduce count
            # For each selected block, try replacing it with a non-selected block
            # that covers all currently-uniquely-covered subsets by the removed block
            # and potentially covers more, allowing further deletions later
            swap_improved = True
            max_swap_iters = 5
            for _ in range(max_swap_iters):
                if time.time() > deadline:
                    break
                swap_improved = False
                for i in range(len(selected_blocks)):
                    if time.time() > deadline:
                        break
                    b_idx = selected_blocks[i]
                    # Find subsets that would become uncovered if we remove b_idx
                    unique_to_block = []
                    for idx in block_bit_lists[b_idx]:
                        if cover_counts[idx] == 1:
                            unique_to_block.append(idx)
                    
                    if not unique_to_block:
                        # This block is redundant, remove it
                        for idx in block_bit_lists[b_idx]:
                            cover_counts[idx] -= 1
                        selected_blocks.pop(i)
                        swap_improved = True
                        break
                    
                    # Try to find a replacement block that covers all unique_to_block
                    # and has good gain on remaining uncovered
                    # We'll just try a few random candidates for speed
                    n_tries = min(20, n_blocks - len(selected_blocks))
                    for _ in range(n_tries):
                        if time.time() > deadline:
                            break
                        cand = rng.randint(0, n_blocks - 1)
                        if cand in selected_blocks:
                            continue
                        # Check if cand covers all unique_to_block
                        cand_mask = block_masks[cand]
                        covers_all = True
                        for idx in unique_to_block:
                            if not (cand_mask >> idx) & 1:
                                covers_all = False
                                break
                        if not covers_all:
                            continue
                        
                        # Simulate swap: remove b_idx, add cand
                        # New cover counts
                        new_cc = cover_counts[:]
                        for idx in block_bit_lists[b_idx]:
                            new_cc[idx] -= 1
                        for idx in block_bit_lists[cand]:
                            new_cc[idx] += 1
                        
                        # Check validity: no negative counts
                        valid = True
                        for idx in range(n_subs):
                            if new_cc[idx] < 0:
                                valid = False
                                break
                        if not valid:
                            continue
                        
                        # After swap, try to delete redundant blocks
                        potential_deletions = 0
                        for j, sj in enumerate(selected_blocks):
                            if j == i:
                                continue
                            is_red = True
                            for idx in block_bit_lists[sj]:
                                if new_cc[idx] == 1:
                                    is_red = False
                                    break
                            if is_red:
                                potential_deletions += 1
                        
                        # Net change: -1 (remove b_idx) + 1 (add cand) - potential_deletions
                        # We want net < 0, i.e., potential_deletions > 0
                        if potential_deletions > 0:
                            # Perform the swap
                            for idx in block_bit_lists[b_idx]:
                                cover_counts[idx] -= 1
                            for idx in block_bit_lists[cand]:
                                cover_counts[idx] += 1
                            selected_blocks[i] = cand
                            swap_improved = True
                            break
                        
                        # Even without immediate deletion, accept if it doesn't hurt
                        # and might enable future improvements (stochastic)
                        if rng.random() < 0.1:
                            for idx in block_bit_lists[b_idx]:
                                cover_counts[idx] -= 1
                            for idx in block_bit_lists[cand]:
                                cover_counts[idx] += 1
                            selected_blocks[i] = cand
                            swap_improved = True
                            break
                    
                    if swap_improved:
                        break
                
                if not swap_improved:
                    break

            if len(selected_blocks) < best_count:
                best_count = len(selected_blocks)
                best_blocks = [list(all_blocks[i]) for i in selected_blocks]
                
                if best_count <= 335:
                    break

    if best_blocks is None:
        return []
        
    return best_blocks
