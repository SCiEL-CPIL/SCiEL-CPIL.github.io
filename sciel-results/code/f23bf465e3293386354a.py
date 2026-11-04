import itertools
import random
import time
import math
import numpy as np

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
subset_frequencies = [0] * N_SUBSETS

for block in itertools.combinations(range(V), K):
    mask = 0
    for sub in itertools.combinations(block, T):
        idx = subset_to_idx[sub]
        mask |= 1 << idx
        blocks_containing_subset[idx].append(len(all_blocks))
        subset_frequencies[idx] += 1
    all_blocks.append(block)
    all_masks.append(mask)

N_BLOCKS = len(all_blocks)

# Precompute weights for weighted greedy
# Weight is inverse of frequency, normalized
max_freq = max(subset_frequencies)
weights = [1.0 / f for f in subset_frequencies]

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
    
    # 1. Weighted Greedy Initialization
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

    # Weighted Greedy
    uncovered_mask = (1 << N_SUBSETS) - 1
    while uncovered_mask:
        best_idx = -1
        best_gain = 0.0
        for i in range(N_BLOCKS):
            m = all_masks[i]
            gain = 0.0
            mm = m & uncovered_mask
            while mm:
                lsb = mm & -mm
                bit_idx = lsb.bit_length() - 1
                gain += weights[bit_idx]
                mm ^= lsb
            if gain > best_gain:
                best_gain = gain
                best_idx = i
        if best_gain == 0:
            break # Should not happen if uncovered_mask is not 0
        add_block(best_idx)
        uncovered_mask &= ~all_masks[best_idx]
        
    best_len = len(current_blocks)
    best_blocks = current_blocks.copy()
    
    # 2. Local Search with Perturbation
    no_improve_count = 0
    max_no_improve = 500
    
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
            
        # If no redundant block found, try a swap
        if len(current_blocks) == 0: break
        
        # --- NEW: Try 2-for-1 Exchange ---
        # Remove b1, b2 and add b3 such that coverage is maintained.
        # This reduces block count by 1.
        found_2_1 = False
        if len(current_blocks) >= 2:
            # Try a few random pairs
            attempts = 50
            for _ in range(attempts):
                if time.time() > deadline: break
                i1 = random.randint(0, len(current_blocks) - 1)
                i2 = random.randint(0, len(current_blocks) - 1)
                if i1 == i2: continue
                
                b1_idx = current_blocks[i1]
                b2_idx = current_blocks[i2]
                
                # Calculate subsets that would be uncovered if we remove both
                # We need to check counts.
                # If a subset is covered only by b1 or b2 (and not by any other block), it becomes uncovered.
                # Actually, if count[s] == 1 and s is in b1, it's unique to b1.
                # If count[s] == 1 and s is in b2, it's unique to b2.
                # If count[s] == 2 and s is in both b1 and b2, it becomes uncovered (count drops to 0).
                
                uncovered_bits = set()
                
                # Check subsets in b1
                m1 = all_masks[b1_idx]
                mm = m1
                while mm:
                    lsb = mm & -mm
                    bit_idx = lsb.bit_length() - 1
                    if counts[bit_idx] == 1:
                        uncovered_bits.add(bit_idx)
                    elif counts[bit_idx] == 2:
                        # Check if b2 also covers this bit
                        if all_masks[b2_idx] & (1 << bit_idx):
                            uncovered_bits.add(bit_idx)
                    mm ^= lsb
                
                # Check subsets in b2 (that are not already in uncovered_bits from b1 logic? 
                # Actually, if count==1, it was added. If count==2 and in both, added.
                # What if count==2, in b2 but NOT in b1? Then removing b2 makes count 1, so still covered by b1? 
                # No, if it's in b2 and not b1, and count is 2, then there is one other block. 
                # Removing b2 leaves count 1. So it is NOT uncovered.
                # So we only care about bits that become count 0.
                # Bits in b1:
                #   count 1 -> becomes 0 (Uncovered)
                #   count 2 -> if in b2, becomes 0 (Uncovered); if not in b2, becomes 1 (Covered)
                # Bits in b2:
                #   count 1 -> becomes 0 (Uncovered)
                #   count 2 -> if in b1, becomes 0 (Uncovered); if not in b1, becomes 1 (Covered)
                
                # My loop above handled b1.
                # Now handle b2 bits that might be unique to b2 or shared with b1.
                m2 = all_masks[b2_idx]
                mm = m2
                while mm:
                    lsb = mm & -mm
                    bit_idx = lsb.bit_length() - 1
                    if counts[bit_idx] == 1:
                        uncovered_bits.add(bit_idx)
                    elif counts[bit_idx] == 2:
                        # If it was in b1, it was already added in the first loop.
                        # If it is NOT in b1, removing b2 makes count 1, so it is covered by b1? 
                        # No, if count is 2, and it's in b2, and NOT in b1, then the other cover is some b3.
                        # Removing b2 leaves b3. Count becomes 1. It is covered.
                        # So we don't add it.
                        pass
                    mm ^= lsb

                if not uncovered_bits:
                    # Both are redundant together? If no bits are uncovered, we can just remove one?
                    # If no bits are uncovered, it means every bit in b1 or b2 is covered by at least one other block (or the other one).
                    # If they are disjoint in their unique coverage, we might be able to remove both?
                    # No, if we remove both, and no bits become uncovered, then all bits were covered by >=2 other blocks (or the other of the pair).
                    # This implies we can remove both and coverage is maintained.
                    # However, we want to replace with 1 block. If we can remove 2 and add 0, that's even better.
                    # But usually, we can't remove 2 and add 0 because the blocks are not redundant individually.
                    # If uncovered_bits is empty, it means we can remove both?
                    # Let's check: if I remove b1, does anything become uncovered? 
                    # If a bit in b1 has count 1, it becomes 0. So it would be in uncovered_bits.
                    # If uncovered_bits is empty, then no bit in b1 has count 1, and no bit in b2 has count 1.
                    # And no bit in both has count 2.
                    # This means we can remove b1? 
                    # If we remove b1, bits in b1 with count 2 (and not in b2) become 1. Covered.
                    # Bits in b1 with count 2 (and in b2) become 1 (from b2). Covered.
                    # So b1 is redundant? 
                    # is_redundant(b1) checks if any bit in b1 has count 1.
                    # If uncovered_bits is empty, it implies no bit in b1 has count 1 (from first loop) 
                    # AND no bit in b2 has count 1 (from second loop).
                    # So b1 is redundant. Why didn't the pruning phase catch it?
                    # Maybe b1 was not redundant BEFORE removing b2? 
                    # If b1 has a bit with count 1, it would be in uncovered_bits.
                    # So if uncovered_bits is empty, b1 is redundant in the current state.
                    # This case should have been caught by the pruning loop.
                    # So this is unlikely.
                    continue

                # We have a set of bits to cover: uncovered_bits
                # Find a block b3 that covers all of them.
                # Sort by frequency of the subset to speed up intersection
                bits_list = list(uncovered_bits)
                bits_list.sort(key=lambda b: len(blocks_containing_subset[b]))
                
                candidates = set(blocks_containing_subset[bits_list[0]])
                for b in bits_list[1:]:
                    if not candidates: break
                    candidates &= set(blocks_containing_subset[b])
                
                if candidates:
                    # Exclude b1 and b2
                    candidates.discard(b1_idx)
                    candidates.discard(b2_idx)
                    if candidates:
                        b3_idx = random.choice(list(candidates))
                        # Perform swap: remove i1, i2, add b3_idx
                        # Note: remove_block changes indices. 
                        # We should remove the higher index first to keep the other index valid.
                        if i1 > i2:
                            remove_block(i1)
                            remove_block(i2)
                        else:
                            remove_block(i2)
                            remove_block(i1)
                        add_block(b3_idx)
                        
                        if len(current_blocks) < best_len:
                            best_len = len(current_blocks)
                            best_blocks = current_blocks.copy()
                        
                        found_2_1 = True
                        no_improve_count = 0
                        break
        
        if found_2_1:
            continue

        # --- End 2-for-1 ---

        # Try 1-for-1 Swap (Neutral move for diversification)
        i = random.randint(0, len(current_blocks) - 1)
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
            
        if len(uncovered_bits) <= 4:
            uncovered_bits.sort(key=lambda b: len(blocks_containing_subset[b]))
            candidates = set(blocks_containing_subset[uncovered_bits[0]])
            for b in uncovered_bits[1:]:
                if not candidates: break
                candidates &= set(blocks_containing_subset[b])
            
            if candidates:
                candidates.discard(current_blocks[i])
                if candidates:
                    j = random.choice(list(candidates))
                    remove_block(i)
                    add_block(j)
                    no_improve_count = 0
                    continue
        
        no_improve_count += 1
        if no_improve_count > max_no_improve:
            # Perturbation: Add a random block, then try to prune
            c = random.randint(0, N_BLOCKS - 1)
            if c not in current_blocks:
                add_block(c)
            no_improve_count = 0
            max_no_improve = min(500, max_no_improve * 2)

    # Final cleanup
    return [list(all_blocks[i]) for i in best_blocks]
