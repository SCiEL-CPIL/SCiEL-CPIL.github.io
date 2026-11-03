import itertools
import time
import random
import numpy as np

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 100.0  # Safety margin
    
    # 1. Precompute indices for t-subsets
    t_subsets = list(itertools.combinations(range(v), t))
    N = len(t_subsets)
    subset_to_idx = {s: i for i, s in enumerate(t_subsets)}
    
    # Full mask of all t-subsets
    full_mask = (1 << N) - 1
    
    # 2. Precompute coverage masks for all k-blocks
    k_blocks = list(itertools.combinations(range(v), k))
    num_blocks = len(k_blocks)
    
    # For each block, compute the bitset of t-subsets it covers
    block_masks = []
    for block in k_blocks:
        mask = 0
        for sub in itertools.combinations(block, t):
            mask |= (1 << subset_to_idx[sub])
        block_masks.append(mask)
        
    # 3. Greedy Construction with Randomization
    # We will run multiple greedy passes with different random seeds/orderings
    # to escape local minima.
    
    best_selected = None
    best_count = float('inf')
    
    # Number of restarts based on time available
    # Each restart is cheap if we just re-run greedy with a different tie-breaking.
    # Since greedy is deterministic if we pick the first max, we can randomize the order of blocks
    # or add small noise to gains.
    
    # Strategy:
    # 1. Standard Greedy (deterministic)
    # 2. Randomized Greedy: When multiple blocks have max gain, pick one randomly.
    #    Or: Add small random noise to gain.
    
    def run_greedy(noise_scale=0.0, seed=0):
        if noise_scale > 0:
            rng = random.Random(seed)
        else:
            rng = None
            
        uncovered = full_mask
        selected = []
        
        while uncovered:
            if time.time() > deadline:
                break
                
            best_gain = -1
            best_blocks = []
            
            # Find all blocks with max gain (or near max with noise)
            # To be efficient, we can just find the max and collect candidates.
            # But collecting all maxes might be expensive if many ties.
            # Better: iterate, keep track of best_gain, and a list of candidates.
            
            for i, mask in enumerate(block_masks):
                if not mask:
                    continue
                newly_covered = mask & uncovered
                gain = newly_covered.bit_count()
                
                if noise_scale > 0:
                    # Add noise to break ties or slightly perturb
                    gain += rng.uniform(0, noise_scale)
                
                if gain > best_gain:
                    best_gain = gain
                    best_blocks = [i]
                elif gain == best_gain:
                    # If exact tie (without noise), keep both.
                    # With noise, ties are rare.
                    if noise_scale == 0:
                        best_blocks.append(i)
            
            if not best_blocks:
                break
                
            if len(best_blocks) > 1:
                chosen = rng.choice(best_blocks) if rng else best_blocks[0]
            else:
                chosen = best_blocks[0]
                
            selected.append(chosen)
            uncovered &= ~block_masks[chosen]
            
        return selected, uncovered

    # Run 1: Deterministic
    sel1, unc1 = run_greedy(noise_scale=0.0)
    if not unc1:
        best_selected = sel1
        best_count = len(sel1)
    else:
        # If it didn't finish, it's invalid, but we keep it as a fallback?
        # No, must be valid. If deadline hit, we might have partial.
        # The problem says "Valid when every t-subset... lies in at least one block".
        # So we must ensure validity.
        pass

    # Run 2-5: Randomized with small noise to break ties differently
    # This helps explore different local optima.
    num_restarts = 5
    for i in range(num_restarts):
        if time.time() > deadline:
            break
            
        # Small noise to perturb the greedy choice
        # Gain is integer, so noise < 0.5 doesn't change order unless gain is equal.
        # So this effectively randomizes tie-breaking.
        sel, unc = run_greedy(noise_scale=0.1, seed=i+1)
        
        if not unc: # Valid covering
            if len(sel) < best_count:
                best_count = len(sel)
                best_selected = sel

    # If we didn't find a valid covering in time (unlikely for this size),
    # we might need to continue? Or return best effort?
    # The contract says "Valid when...". If we return invalid, score is 0.
    # Given 110s limit and greedy is fast, we should always finish.
    
    if best_selected is None:
        # Fallback: Return the last one we had, hoping it was valid?
        # Or re-run without time check?
        # For safety, let's assume the first deterministic one finished.
        # If not, we have a problem.
        # Let's just use sel1 if it was valid, otherwise we are in trouble.
        # But sel1 might have been cut off.
        # Let's re-check sel1 validity explicitly if best_selected is None.
        # Actually, run_greedy returns unc. If unc != 0, it's invalid.
        # We only updated best_selected if unc == 0.
        # So if best_selected is None, we have no valid covering.
        # This should not happen.
        return []

    # 4. Post-hoc Redundancy Removal (Local Search Polish)
    selected_blocks = best_selected
    
    # Count how many times each t-subset is covered
    counts = [0] * N
    
    for idx in selected_blocks:
        mask = block_masks[idx]
        m = mask
        while m:
            lsb = m & -m
            idx_bit = lsb.bit_length() - 1
            counts[idx_bit] += 1
            m ^= lsb
            
    # Identify unique bits (count == 1)
    unique_mask = 0
    for i in range(N):
        if counts[i] == 1:
            unique_mask |= (1 << i)
            
    # Remove redundant blocks
    changed = True
    while changed:
        changed = False
        new_selected = []
        for idx in selected_blocks:
            mask = block_masks[idx]
            # If this block covers any unique bit, it's essential
            if mask & unique_mask:
                new_selected.append(idx)
            else:
                # Redundant: remove it
                changed = True
                # Update counts
                m = mask
                while m:
                    lsb = m & -m
                    idx_bit = lsb.bit_length() - 1
                    counts[idx_bit] -= 1
                    if counts[idx_bit] == 1:
                        unique_mask |= (1 << idx_bit)
                    elif counts[idx_bit] == 0:
                        unique_mask &= ~(1 << idx_bit)
                    m ^= lsb
        
        selected_blocks = new_selected
        
    # 5. Additional Polish: Try swapping blocks?
    # If we have time, we can try to replace pairs of blocks with one block?
    # Or replace a block with a different one that covers the same unique constraints?
    # This is a local search step.
    
    # Given the tight deadline, simple redundancy removal is the most robust gain.
    # The randomized greedy should find a good initial set.
    
    # 6. Convert indices back to blocks
    result = [list(k_blocks[i]) for i in selected_blocks]
    
    return result
