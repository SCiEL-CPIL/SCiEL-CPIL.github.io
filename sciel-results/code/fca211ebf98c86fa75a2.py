import itertools
import time
import random
import numpy as np

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin for 120s limit (150s kill)

    # 1. Generate all t-subsets and map them to indices
    all_tsubs = list(itertools.combinations(range(v), t))
    n_tsubs = len(all_tsubs)
    tsub_to_idx = {sub: i for i, sub in enumerate(all_tsubs)}

    # 2. Generate all k-subsets (candidate blocks)
    all_ksubs = list(itertools.combinations(range(v), k))
    n_ksubs = len(all_ksubs)

    # 3. Precompute coverage mask for each k-subset as a Python int bitset
    ksub_masks = [0] * n_ksubs
    for j, ks in enumerate(all_ksubs):
        mask = 0
        for tsub in itertools.combinations(ks, t):
            idx = tsub_to_idx[tsub]
            mask |= (1 << idx)
        ksub_masks[j] = mask

    # 4. Initialize uncovered bitset (all 1s for n_tsubs bits)
    uncovered_mask = (1 << n_tsubs) - 1
    remaining = n_tsubs

    blocks = []
    used_blocks = set()
    selected_indices = []

    # 5. Greedy selection loop
    while remaining > 0:
        if time.time() > deadline:
            break

        best_j = -1
        best_gain = 0

        # Scan all candidates to find the one with max gain
        for j in range(n_ksubs):
            if j in used_blocks:
                continue
            
            # Calculate gain: number of newly covered t-subsets
            common = uncovered_mask & ksub_masks[j]
            gain = common.bit_count()
            
            if gain > best_gain:
                best_gain = gain
                best_j = j

        if best_j == -1 or best_gain == 0:
            # No more progress possible
            break

        # Add best block
        blocks.append(list(all_ksubs[best_j]))
        used_blocks.add(best_j)
        selected_indices.append(best_j)
        
        # Update uncovered mask
        uncovered_mask &= ~ksub_masks[best_j]
        remaining -= best_gain

    # 6. Post-processing: Remove redundant blocks
    #    This is the refinement machinery.
    #    A block is redundant if all its t-subsets are covered by other selected blocks.
    
    if time.time() < deadline - 5.0 and len(blocks) > 0:
        # Try to remove redundant blocks
        # We maintain a mask of what is covered by the *other* selected blocks.
        # For each block i, check if ksub_masks[i] is a subset of (union of masks of j != i).
        
        active_indices = list(selected_indices)
        
        changed = True
        while changed and time.time() < deadline - 2.0:
            changed = False
            new_active = []
            for i_idx, bi in enumerate(active_indices):
                # Calculate union of all OTHER active blocks
                other_mask = 0
                for j_idx, bj in enumerate(active_indices):
                    if i_idx != j_idx:
                        other_mask |= ksub_masks[bj]
                
                # Check if bi is redundant
                # If all bits in bi are in other_mask, then bi is redundant.
                # i.e., (ksub_masks[bi] & ~other_mask) == 0
                if (ksub_masks[bi] & ~other_mask) == 0:
                    # Redundant, do not add to new_active
                    changed = True
                else:
                    new_active.append(bi)
            
            active_indices = new_active
            
            # Early stop if no change
            if not changed:
                break

        # Reconstruct blocks from active_indices
        if active_indices:
            final_blocks = [list(all_ksubs[idx]) for idx in active_indices]
            # Verify validity
            final_mask = 0
            for idx in active_indices:
                final_mask |= ksub_masks[idx]
            if final_mask == (1 << n_tsubs) - 1:
                blocks = final_blocks
            # else keep original blocks (should not happen if logic is correct)

    return blocks
