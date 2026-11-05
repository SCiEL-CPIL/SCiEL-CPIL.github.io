import itertools
import time
import random
import numpy as np
from scipy.optimize import linear_sum_assignment

# Global Precomputation
V = 19
K = 7
T = 4

# Map 4-subsets to indices
subsets = list(itertools.combinations(range(V), T))
subset_to_idx = {s: i for i, s in enumerate(subsets)}
N_SUBSETS = len(subsets)

# Generate all 7-subsets and their masks
all_blocks = []
all_masks = []
blocks_containing_subset = [[] for _ in range(N_SUBSETS)]

for block in itertools.combinations(range(V), K):
    mask = 0
    for sub in itertools.combinations(block, T):
        idx = subset_to_idx[sub]
        mask |= 1 << idx
        blocks_containing_subset[idx].append(len(all_blocks))
    all_blocks.append(block)
    all_masks.append(mask)

N_BLOCKS = len(all_blocks)

def construct(v, k, t):
    if v != 19 or k != 7 or t != 4:
        # Fallback for other parameters (not expected in this task)
        uncovered = set(itertools.combinations(range(v), t))
        blocks = []
        all_k_subsets = list(itertools.combinations(range(v), k))
        random.shuffle(all_k_subsets)
        for b in all_k_subsets:
            if not uncovered:
                break
            covered = set(itertools.combinations(b, t))
            if uncovered & covered:
                blocks.append(list(b))
                uncovered -= covered
        return blocks

    start_time = time.time()
    deadline = start_time + 118  # Safety margin

    # 1. Greedy Initialization
    counts = [0] * N_SUBSETS
    current_blocks = []
    current_masks = []
    
    def add_block(idx):
        current_blocks.append(idx)
        current_masks.append(all_masks[idx])
        m = all_masks[idx]
        mm = m
        while mm:
            lsb = mm & -mm
            bit_idx = lsb.bit_length() - 1
            counts[bit_idx] += 1
            mm ^= lsb

    def remove_block(i):
        m = current_masks[i]
        mm = m
        while mm:
            lsb = mm & -mm
            bit_idx = lsb.bit_length() - 1
            counts[bit_idx] -= 1
            mm ^= lsb
        current_blocks.pop(i)
        current_masks.pop(i)

    def is_redundant(i):
        m = current_masks[i]
        mm = m
        while mm:
            lsb = mm & -mm
            bit_idx = lsb.bit_length() - 1
            if counts[bit_idx] == 1:
                return False
            mm ^= lsb
        return True

    # Greedy
    uncovered_mask = (1 << N_SUBSETS) - 1
    while uncovered_mask:
        best_idx = -1
        best_gain = 0
        # Use a heuristic to speed up: pick block with max new coverage
        # Randomize ties to avoid bias
        for i in range(N_BLOCKS):
            gain = (all_masks[i] & uncovered_mask).bit_count()
            if gain > best_gain:
                best_gain = gain
                best_idx = i
        if best_idx == -1:
            break
        add_block(best_idx)
        uncovered_mask &= ~all_masks[best_idx]
        
    best_len = len(current_blocks)
    best_blocks = current_blocks.copy()
    
    # 2. Local Search with Perturbation (ILS style)
    # We alternate between:
    # 1. Pruning redundant blocks
    # 2. Swapping blocks to enable further pruning
    # 3. Perturbing if stuck
    
    no_improve_count = 0
    max_no_improve = 50 # Lower threshold to trigger perturbation faster
    
    while time.time() < deadline:
        # Phase 1: Prune
        removed_any = False
        indices = list(range(len(current_blocks)))
        random.shuffle(indices)
        
        for i in indices:
            if time.time() > deadline:
                break
            if is_redundant(i):
                remove_block(i)
                removed_any = True
                if len(current_blocks) < best_len:
                    best_len = len(current_blocks)
                    best_blocks = current_blocks.copy()
                # Restarting the loop is safer after removal
                break
        
        if removed_any:
            no_improve_count = 0
            continue
            
        # Phase 2: Swap
        if len(current_blocks) == 0:
            break
            
        # Pick a random block to remove
        i = random.randint(0, len(current_blocks) - 1)
        
        # Find subsets that would become uncovered
        m_i = current_masks[i]
        uncovered_bits = []
        mm = m_i
        while mm:
            lsb = mm & -mm
            bit_idx = lsb.bit_length() - 1
            if counts[bit_idx] == 1:
                uncovered_bits.append(bit_idx)
            mm ^= lsb
            
        if not uncovered_bits:
            # It was redundant, remove it
            remove_block(i)
            if len(current_blocks) < best_len:
                best_len = len(current_blocks)
                best_blocks = current_blocks.copy()
            no_improve_count = 0
            continue
            
        # If too many bits to cover, this swap is likely hard.
        # We can still try if count is small, otherwise perturb.
        if len(uncovered_bits) > 4:
            no_improve_count += 1
            if no_improve_count > max_no_improve:
                # Perturbation: Add a random block, then prune
                # We add a block that covers some currently well-covered areas?
                # Or just any random block.
                # To be effective, maybe pick a block that covers many bits
                # with count 2 or 3?
                # For simplicity, random block.
                c = random.randint(0, N_BLOCKS - 1)
                if c not in current_blocks:
                    add_block(c)
                    no_improve_count = 0
                    max_no_improve = min(200, max_no_improve * 2) # Annealing
                continue
        else:
            # Find a block that covers all uncovered_bits
            # Sort by frequency to speed up intersection
            uncovered_bits.sort(key=lambda b: len(blocks_containing_subset[b]))
            
            # Start with the smallest list
            candidates = blocks_containing_subset[uncovered_bits[0]]
            
            # Intersect with others
            for b in uncovered_bits[1:]:
                if not candidates:
                    break
                # Optimization: If the list is small, use set intersection
                # Otherwise, filter
                set_cands = set(candidates)
                set_b = set(blocks_containing_subset[b])
                candidates = list(set_cands.intersection(set_b))
                
            if candidates:
                # Pick a random candidate
                # Ensure it's not the block we're removing (though it shouldn't be if it covers the unique bits)
                # Actually, if i covered unique bits, and j covers them, j != i is not guaranteed if i is in candidates?
                # Wait, i covers the bits. So i is in blocks_containing_subset for all these bits.
                # So i IS in candidates. We must exclude i.
                valid_candidates = [c for c in candidates if c != i]
                if valid_candidates:
                    j = random.choice(valid_candidates)
                    # Perform swap
                    remove_block(i)
                    add_block(j)
                    no_improve_count = 0
                    continue
            else:
                no_improve_count += 1
                if no_improve_count > max_no_improve:
                    # Perturbation
                    c = random.randint(0, N_BLOCKS - 1)
                    if c not in current_blocks:
                        add_block(c)
                        no_improve_count = 0
                        max_no_improve = min(200, max_no_improve * 2)
                    continue

    # Final cleanup: ensure we return the best found
    # Note: best_blocks is a list of indices into all_blocks
    # We need to return the actual blocks
    return [list(all_blocks[i]) for i in best_blocks]
