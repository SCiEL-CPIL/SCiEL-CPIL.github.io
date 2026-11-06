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
all_block_subsets = []
blocks_containing_subset = [[] for _ in range(N_SUBSETS)]
subset_frequencies = [0] * N_SUBSETS

for block in itertools.combinations(range(V), K):
    mask = 0
    sub_list = []
    for sub in itertools.combinations(block, T):
        idx = subset_to_idx[sub]
        mask |= 1 << idx
        sub_list.append(idx)
        blocks_containing_subset[idx].append(len(all_blocks))
        subset_frequencies[idx] += 1
    all_blocks.append(block)
    all_masks.append(mask)
    all_block_subsets.append(sub_list)

N_BLOCKS = len(all_blocks)

# Precompute weights for weighted greedy
# Weight is inverse of frequency, normalized
max_freq = max(subset_frequencies)
weights = np.array([1.0 / f for f in subset_frequencies])

# Precompute bit counts for each block (number of subsets it covers)
# For greedy, we need efficient gain calculation

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
    
    best_len = float('inf')
    best_blocks = []
    
    def time_ok():
        return time.time() < deadline

    # Run multiple iterations with different random seeds or perturbations
    for restart in range(30):  # Up to 30 restarts if time permits
        if not time_ok():
            break
            
        current_blocks = []
        current_masks = []
        counts = np.zeros(N_SUBSETS, dtype=np.int32)
        
        def add_block(idx):
            current_blocks.append(idx)
            current_masks.append(all_masks[idx])
            for s in all_block_subsets[idx]:
                counts[s] += 1

        def remove_block(i):
            m = current_masks[i]
            for s in all_block_subsets[current_blocks[i]]:
                counts[s] -= 1
            current_blocks.pop(i)
            current_masks.pop(i)

        def is_redundant(i):
            for s in all_block_subsets[current_blocks[i]]:
                if counts[s] == 1:
                    return False
            return True

        # 1. Weighted Greedy Initialization
        uncovered_mask = (1 << N_SUBSETS) - 1
        # Convert to numpy boolean array for faster operations
        uncovered_arr = np.zeros(N_SUBSETS, dtype=bool)
        for i in range(N_SUBSETS):
            uncovered_arr[i] = True
        
        while np.any(uncovered_arr):
            best_idx = -1
            best_gain = 0.0
            
            # Use numpy for faster gain calculation
            # For each block, sum weights of uncovered subsets
            # Precompute weight array
            gains = np.zeros(N_BLOCKS)
            
            # This is still slow in pure Python loop. Let's try a different approach.
            # For each subset that is uncovered, add its weight to all blocks containing it.
            
            # Get indices of uncovered subsets
            uncov_indices = np.where(uncovered_arr)[0]
            
            if len(uncov_indices) == 0:
                break
            
            # For each uncovered subset, distribute weight to containing blocks
            # This is O(sum of frequencies) which might be better
            gains.fill(0)
            for idx in uncov_indices:
                w = weights[idx]
                for b in blocks_containing_subset[idx]:
                    gains[b] += w
            
            best_idx = int(np.argmax(gains))
            best_gain = gains[best_idx]
            
            if best_gain == 0:
                break
                
            add_block(best_idx)
            # Update uncovered
            for s in all_block_subsets[best_idx]:
                uncovered_arr[s] = False
            uncovered_mask &= ~all_masks[best_idx]
            
        if len(current_blocks) < best_len:
            best_len = len(current_blocks)
            best_blocks = current_blocks.copy()
            
        if not time_ok():
            break

        # 2. Local Search
        no_improve_count = 0
        max_no_improve = 50
        
        while time_ok():
            # Try to remove redundant blocks
            removed_any = False
            indices = list(range(len(current_blocks)))
            random.shuffle(indices)
            
            for i in indices:
                if not time_ok(): break
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
                
            # If no redundant block found, try a 2-for-1 swap
            found_swap = False
            if len(current_blocks) >= 2:
                # Random pairs
                attempts = 50
                for _ in range(attempts):
                    if not time_ok(): break
                    i1 = random.randint(0, len(current_blocks) - 1)
                    i2 = random.randint(0, len(current_blocks) - 1)
                    if i1 == i2: continue
                    
                    b1_idx = current_blocks[i1]
                    b2_idx = current_blocks[i2]
                    
                    # Calculate uncovered bits if we remove both
                    uncovered_bits = set()
                    
                    for s in all_block_subsets[b1_idx]:
                        if counts[s] == 1:
                            uncovered_bits.add(s)
                        elif counts[s] == 2 and s in all_block_subsets[b2_idx]:
                            uncovered_bits.add(s)
                    
                    for s in all_block_subsets[b2_idx]:
                        if counts[s] == 1:
                            uncovered_bits.add(s)

                    if not uncovered_bits:
                        continue

                    # Find a block that covers all uncovered_bits
                    bits_list = list(uncovered_bits)
                    bits_list.sort(key=lambda b: len(blocks_containing_subset[b]))
                    
                    candidates = set(blocks_containing_subset[bits_list[0]])
                    for b in bits_list[1:]:
                        if not candidates: break
                        candidates.intersection_update(blocks_containing_subset[b])
                    
                    if candidates:
                        candidates.discard(b1_idx)
                        candidates.discard(b2_idx)
                        if candidates:
                            b3_idx = random.choice(list(candidates))
                            # Perform swap
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
                            
                            found_swap = True
                            no_improve_count = 0
                            break
                            
            if found_swap:
                continue

            # Try 1-for-1 Swap
            i = random.randint(0, len(current_blocks) - 1)
            uncovered_bits = []
            for s in all_block_subsets[current_blocks[i]]:
                if counts[s] == 1:
                    uncovered_bits.append(s)
                
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
                    candidates.intersection_update(blocks_containing_subset[b])
                
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
                # Perturbation
                c = random.randint(0, N_BLOCKS - 1)
                if c not in current_blocks:
                    add_block(c)
                no_improve_count = 0
                max_no_improve = min(200, max_no_improve * 2)

    # Final cleanup
    return [list(all_blocks[i]) for i in best_blocks]
