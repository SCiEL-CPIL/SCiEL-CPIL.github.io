import itertools
import random
import time
import math
import numpy as np
from collections import defaultdict

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

# Precompute for faster gain calculation
# For each block, store the list of subset indices it covers
block_subsets = []
for mask in all_masks:
    subs = []
    mm = mask
    while mm:
        lsb = mm & -mm
        bit_idx = lsb.bit_length() - 1
        subs.append(bit_idx)
        mm ^= lsb
    block_subsets.append(subs)

# Precompute point-to-subset mappings for column generation
# For each point, list of 4-subsets containing that point
point_to_subsets = [[] for _ in range(V)]
for idx, sub in enumerate(subsets):
    for p in sub:
        point_to_subsets[p].append(idx)

# For each pair of points, list of 4-subsets containing both
pair_to_subsets = defaultdict(list)
for idx, sub in enumerate(subsets):
    for i in range(4):
        for j in range(i+1, 4):
            pair = (sub[i], sub[j])
            pair_to_subsets[pair].append(idx)

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
    deadline = start_time + 115  # Safety margin
    
    # 1. Column-Generation Inspired Greedy with Weighted Pricing
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
        # Remove from lists
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

    # Weighted greedy: prioritize blocks that cover rare subsets
    # Compute frequency of each subset across all blocks
    subset_freq = [0] * N_SUBSETS
    for idx in range(N_BLOCKS):
        for s in block_subsets[idx]:
            subset_freq[s] += 1
    
    # Weight for each subset: inverse frequency (rarer subsets get higher weight)
    subset_weight = [1.0 / max(1, subset_freq[s]) for s in range(N_SUBSETS)]
    
    # Greedy with weighted gain
    uncovered_mask = (1 << N_SUBSETS) - 1
    while uncovered_mask:
        best_idx = -1
        best_gain = 0
        for i in range(N_BLOCKS):
            # Compute weighted gain
            gain = 0
            m = all_masks[i] & uncovered_mask
            mm = m
            while mm:
                lsb = mm & -mm
                bit_idx = lsb.bit_length() - 1
                gain += subset_weight[bit_idx]
                mm ^= lsb
            if gain > best_gain:
                best_gain = gain
                best_idx = i
        if best_idx == -1:
            break
        add_block(best_idx)
        uncovered_mask &= ~all_masks[best_idx]
        
    best_len = len(current_blocks)
    best_blocks = current_blocks.copy()
    
    # 2. Local Search with Column Generation Pricing
    no_improve_count = 0
    max_no_improve = 150
    
    while time.time() < deadline:
        # Try to remove redundant blocks
        removed_any = False
        indices = list(range(len(current_blocks)))
        random.shuffle(indices)
        
        for i in indices:
            if time.time() > deadline: break
            if is_redundant(i):
                remove_block(i)
                removed_any = True
                if len(current_blocks) < best_len:
                    best_len = len(current_blocks)
                    best_blocks = current_blocks.copy()
                break
        
        if removed_any:
            no_improve_count = 0
            continue
            
        # If no redundant block found, try a swap using column generation pricing
        if len(current_blocks) == 0: break
        
        # Pick a random block to remove
        i = random.randint(0, len(current_blocks) - 1)
        
        # Tentatively remove
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
            remove_block(i)
            if len(current_blocks) < best_len:
                best_len = len(current_blocks)
                best_blocks = current_blocks.copy()
            no_improve_count = 0
            continue
            
        # We need to cover uncovered_bits
        if len(uncovered_bits) > 4:
            no_improve_count += 1
            if no_improve_count > max_no_improve:
                # Perturbation: Use column generation to find a good block to add
                # Find the most "expensive" uncovered subset and find blocks covering it
                max_weight_idx = max(uncovered_bits, key=lambda b: subset_weight[b])
                candidates = blocks_containing_subset[max_weight_idx]
                if candidates:
                    c = random.choice(candidates)
                    if c not in current_blocks:
                        add_block(c)
                        no_improve_count = 0
                continue
        else:
            # Find a block that covers all uncovered_bits using column generation pricing
            uncovered_bits.sort(key=lambda b: len(blocks_containing_subset[b]))
            candidates = blocks_containing_subset[uncovered_bits[0]]
            
            for b in uncovered_bits[1:]:
                if not candidates: break
                set_cands = set(candidates)
                set_b = set(blocks_containing_subset[b])
                candidates = list(set_cands.intersection(set_b))
                
            if candidates:
                valid_candidates = [c for c in candidates if c != i]
                if valid_candidates:
                    # Choose candidate that maximizes weighted coverage of other uncovered subsets
                    best_j = valid_candidates[0]
                    best_score = -1
                    for j in valid_candidates:
                        score = 0
                        m_j = all_masks[j]
                        mm_j = m_j
                        while mm_j:
                            lsb = mm_j & -mm_j
                            bit_idx = lsb.bit_length() - 1
                            if counts[bit_idx] == 1:  # This subset is only covered by block i
                                score += subset_weight[bit_idx]
                            mm_j ^= lsb
                        if score > best_score:
                            best_score = score
                            best_j = j
                    remove_block(i)
                    add_block(best_j)
                    no_improve_count = 0
                    continue
            else:
                no_improve_count += 1
                if no_improve_count > max_no_improve:
                    # Column generation perturbation: find a block that covers many rare subsets
                    best_c = -1
                    best_c_score = 0
                    for c in range(N_BLOCKS):
                        if c in current_blocks:
                            continue
                        score = 0
                        m_c = all_masks[c]
                        mm_c = m_c
                        while mm_c:
                            lsb = mm_c & -mm_c
                            bit_idx = lsb.bit_length() - 1
                            if counts[bit_idx] == 0:  # Uncovered subset
                                score += subset_weight[bit_idx]
                            mm_c ^= lsb
                        if score > best_c_score:
                            best_c_score = score
                            best_c = c
                    if best_c != -1:
                        add_block(best_c)
                    no_improve_count = 0
                    continue

    # Final cleanup: remove any remaining redundant blocks from best solution
    # Reconstruct counts for best_blocks
    final_counts = [0] * N_SUBSETS
    for idx in best_blocks:
        m = all_masks[idx]
        mm = m
        while mm:
            lsb = mm & -mm
            bit_idx = lsb.bit_length() - 1
            final_counts[bit_idx] += 1
            mm ^= lsb
    
    # Try to remove redundant blocks from best solution
    final_blocks = best_blocks.copy()
    for i in range(len(final_blocks) - 1, -1, -1):
        m = all_masks[final_blocks[i]]
        mm = m
        redundant = True
        while mm:
            lsb = mm & -mm
            bit_idx = lsb.bit_length() - 1
            if final_counts[bit_idx] == 1:
                redundant = False
                break
            mm ^= lsb
        if redundant:
            # Remove it
            mm = m
            while mm:
                lsb = mm & -mm
                bit_idx = lsb.bit_length() - 1
                final_counts[bit_idx] -= 1
                mm ^= lsb
            final_blocks.pop(i)
    
    return [list(all_blocks[i]) for i in final_blocks]
