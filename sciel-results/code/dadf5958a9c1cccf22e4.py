import itertools
import random
import time
import math
import numpy as np

def construct(v, k, t):
    start = time.time()
    deadline = start + 118  # Safety margin for 120s limit
    
    rng = random.Random(1234567890)

    # Precompute all k-subsets (blocks) and t-subsets
    all_blocks = list(itertools.combinations(range(v), k))
    n_blocks = len(all_blocks)
    t_subsets = list(itertools.combinations(range(v), t))
    n_subsets = len(t_subsets)
    sub_idx = {s: i for i, s in enumerate(t_subsets)}

    # Precompute block masks (bitsets of t-subsets covered)
    block_masks = [0] * n_blocks
    for bi, blk in enumerate(all_blocks):
        mask = 0
        for sub in itertools.combinations(blk, t):
            si = sub_idx[sub]
            mask |= 1 << si
        block_masks[bi] = mask

    # Greedy Set Cover Initialization
    def greedy_init():
        uncovered = (1 << n_subsets) - 1
        chosen = []
        chosen_set = set()
        while uncovered:
            best_bi = -1
            best_gain = 0
            # Optimization: We can vectorize this if we use numpy, but for 50k blocks, 
            # a python loop with bit_count is often fast enough for the first few iterations.
            # However, let's use a simple loop.
            for bi in range(n_blocks):
                if bi in chosen_set:
                    continue
                gain = (block_masks[bi] & uncovered).bit_count()
                if gain > best_gain:
                    best_gain = gain
                    best_bi = bi
            if best_bi < 0:
                break
            chosen.append(best_bi)
            chosen_set.add(best_bi)
            uncovered &= ~block_masks[best_bi]
        return chosen

    # Prune redundant blocks
    def prune(blocks_list):
        chosen = list(blocks_list)
        counts = [0] * n_subsets
        for bi in chosen:
            mask = block_masks[bi]
            while mask:
                low = mask & -mask
                si = low.bit_length() - 1
                counts[si] += 1
                mask ^= low
        
        changed = True
        while changed:
            changed = False
            for i in range(len(chosen)):
                bi = chosen[i]
                mask = block_masks[bi]
                redundant = True
                while mask:
                    low = mask & -mask
                    si = low.bit_length() - 1
                    if counts[si] == 1:
                        redundant = False
                        break
                    mask ^= low
                if redundant:
                    mask = block_masks[bi]
                    while mask:
                        low = mask & -mask
                        si = low.bit_length() - 1
                        counts[si] -= 1
                        mask ^= low
                    chosen.pop(i)
                    changed = True
                    break
        return chosen

    # 2-Exchange Local Search
    # State: list of block indices, counts of coverage for each subset
    # Move: Remove one block, add one block (or just remove if valid, but we want to minimize size)
    # Actually, since we start with a valid covering, we can try to remove a block and see if it's still valid.
    # If not, we try to swap it with a different block that covers the lost subsets better.
    
    def local_search_2_exchange(initial_blocks, time_budget):
        current_blocks = list(initial_blocks)
        current_set = set(current_blocks)
        
        # Calculate initial coverage counts
        counts = [0] * n_subsets
        for bi in current_blocks:
            mask = block_masks[bi]
            while mask:
                low = mask & -mask
                si = low.bit_length() - 1
                counts[si] += 1
                mask ^= low
        
        best_blocks = list(current_blocks)
        best_size = len(current_blocks)
        
        t_start = time.time()
        
        # To speed up, we can precompute for each block, the list of subsets it covers.
        # But we already have masks.
        
        # Strategy: 
        # 1. Try to remove a random block. If valid, keep it.
        # 2. If not valid, try to find a replacement block that covers the uncovered subsets 
        #    introduced by the removal, while minimizing the "cost" (which is 0 if we just swap, 
        #    but we want to reduce size, so we only accept swaps if they allow further removals? 
        #    No, 2-exchange usually means: remove A, add B. Size stays same. 
        #    But we want to reduce size. 
        #    So we should primarily try removals. 
        #    If a removal makes it invalid, we can try to "repair" by adding a block that covers 
        #    the critical subsets, but that increases size. 
        #    So, let's stick to: 
        #    - Try removals. If valid, accept.
        #    - If stuck (no valid removals), try swaps to change the configuration to enable 
        #      future removals. A swap is (remove A, add B). We accept a swap if it doesn't 
        #      increase the size (it doesn't) and potentially improves "potential" for removals.
        #      How to measure potential? 
        #      Maybe: count how many subsets are covered exactly once (critical). 
        #      We want to minimize critical subsets? Or maximize flexibility?
        #      Actually, a common heuristic is to try to replace a block A with B such that 
        #      B covers more subsets that are currently covered only by A, or B covers 
        #      subsets that are covered by many blocks (to free up those blocks for removal).
        
        # Let's implement a simpler but effective search:
        # 1. Randomly pick a block A.
        # 2. Remove A.
        # 3. If valid, accept.
        # 4. If invalid, identify the set of subsets that became uncovered (or dropped to 0).
        #    Let U be the set of subsets with count 0 after removal.
        #    Find a block B that covers all of U.
        #    If such B exists, add B (swap). Accept the swap with some probability or if it improves a heuristic.
        #    Heuristic: Sum of counts of subsets covered by B? Or number of subsets covered by B 
        #    that are currently covered by exactly 1 block (other than A)?
        
        # Let's try:
        # - Try removals first.
        # - If no removals possible, try swaps.
        #   For a swap (remove A, add B):
        #   - B must cover all subsets that A uniquely covered.
        #   - We prefer B that covers many subsets that are currently covered by multiple blocks, 
        #     as this might allow those other blocks to be removed later.
        
        while time.time() - t_start < time_budget:
            # 1. Try random removals
            removed_any = False
            for _ in range(100): # Try up to 100 random removals per iteration
                if len(current_blocks) == 0:
                    break
                idx = rng.randrange(len(current_blocks))
                bi_out = current_blocks[idx]
                
                # Check if removal is valid
                # We need to know if any subset becomes 0.
                # We can check by iterating through the mask of bi_out.
                valid = True
                mask = block_masks[bi_out]
                while mask:
                    low = mask & -mask
                    si = low.bit_length() - 1
                    if counts[si] == 1:
                        valid = False
                        break
                    mask ^= low
                
                if valid:
                    # Perform removal
                    mask = block_masks[bi_out]
                    while mask:
                        low = mask & -mask
                        si = low.bit_length() - 1
                        counts[si] -= 1
                        mask ^= low
                    current_blocks.pop(idx)
                    current_set.discard(bi_out)
                    removed_any = True
                    break
            if removed_any:
                # Update best if size decreased
                if len(current_blocks) < best_size:
                    best_size = len(current_blocks)
                    best_blocks = list(current_blocks)
                continue

            # 2. No removals worked. Try a swap.
            # Pick a random block A to remove.
            if not current_blocks:
                break
            idx = rng.randrange(len(current_blocks))
            bi_out = current_blocks[idx]
            
            # Find subsets that would become uncovered (count 1)
            critical_subsets = []
            mask = block_masks[bi_out]
            while mask:
                low = mask & -mask
                si = low.bit_length() - 1
                if counts[si] == 1:
                    critical_subsets.append(si)
                mask ^= low
            
            if not critical_subsets:
                # This shouldn't happen if we checked removals, but just in case.
                # If no critical subsets, we can just remove it? 
                # Wait, if counts[si] > 1 for all si in mask, then removal is valid.
                # So this branch is unreachable if the removal check above was thorough.
                # But maybe we only checked 100 random ones. 
                # Let's just try to find a swap anyway.
                pass

            # We need to find a block B that covers all critical_subsets.
            # And we want B to be "good".
            # Score for B: Number of subsets in B that are currently covered by > 1 blocks.
            # This suggests B is "redundant" in a way that might help.
            
            best_b = -1
            best_score = -1
            
            # Optimization: Only check blocks that cover the first critical subset.
            if critical_subsets:
                first_crit = critical_subsets[0]
                # Precompute list of blocks covering each subset? 
                # That's 3876 subsets * ~1000 blocks each = ~4M entries. 
                # Might be too much memory or time to build.
                # Let's just iterate all blocks and check if they cover critical_subsets.
                # But we can filter quickly.
                
                for bi_in in range(n_blocks):
                    if bi_in in current_set:
                        continue
                    
                    # Check if bi_in covers all critical subsets
                    mask_in = block_masks[bi_in]
                    covers_all = True
                    for si in critical_subsets:
                        if not (mask_in & (1 << si)):
                            covers_all = False
                            break
                    if not covers_all:
                        continue
                    
                    # Calculate score: How many subsets does it cover that are currently redundant (count > 1)?
                    score = 0
                    m = mask_in
                    while m:
                        low = m & -m
                        si = low.bit_length() - 1
                        if counts[si] > 1:
                            score += 1
                        m ^= low
                    
                    if score > best_score:
                        best_score = score
                        best_b = bi_in
            
            if best_b != -1:
                # Perform swap
                # Remove bi_out
                mask_out = block_masks[bi_out]
                while mask_out:
                    low = mask_out & -mask_out
                    si = low.bit_length() - 1
                    counts[si] -= 1
                    mask_out ^= low
                current_set.discard(bi_out)
                
                # Add best_b
                mask_in = block_masks[best_b]
                while mask_in:
                    low = mask_in & -mask_in
                    si = low.bit_length() - 1
                    counts[si] += 1
                    mask_in ^= low
                current_set.add(best_b)
                
                current_blocks[idx] = best_b
                
                # We don't update best_blocks here because size didn't change.
                # But we continue searching.
                
        return best_blocks

    # Main Search
    init_blocks = greedy_init()
    pruned_init = prune(init_blocks)
    
    best_solution = pruned_init
    best_size = len(pruned_init)
    
    # Run local search
    if time.time() < deadline - 5:
        ls_result = local_search_2_exchange(pruned_init, time_budget=deadline - time.time() - 5)
        pruned_ls = prune(ls_result)
        if len(pruned_ls) < best_size:
            best_solution = pruned_ls
            best_size = len(pruned_ls)
    
    # Final Prune
    final_solution = prune(best_solution)
    
    return [list(all_blocks[bi]) for bi in final_solution]
