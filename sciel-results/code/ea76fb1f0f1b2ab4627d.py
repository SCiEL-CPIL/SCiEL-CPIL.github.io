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
weights_np = np.array(weights)

# Precompute block masks as numpy array of bit positions for vectorized gain computation
# For each block, store the list of subset indices it covers
block_subset_indices = []
for mask in all_masks:
    indices = []
    m = mask
    while m:
        lsb = m & -m
        bit_idx = lsb.bit_length() - 1
        indices.append(bit_idx)
        m ^= lsb
    block_subset_indices.append(indices)

# Precompute as a numpy array with -1 padding for blocks with fewer subsets
max_subsets_per_block = max(len(x) for x in block_subset_indices)
block_indices_np = np.full((N_BLOCKS, max_subsets_per_block), -1, dtype=np.int32)
block_lengths = np.array([len(x) for x in block_subset_indices], dtype=np.int32)
for i, indices in enumerate(block_subset_indices):
    if indices:
        block_indices_np[i, :len(indices)] = indices

# Precompute block masks as Python ints for fast bit operations (already done above)
# Precompute for each block, the list of subset indices as a tuple for faster access
block_indices_tuples = [tuple(x) for x in block_subset_indices]

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
    
    # 1. Weighted Greedy Initialization (optimized)
    counts = [0] * N_SUBSETS
    current_blocks = []
    current_masks = []
    current_set = set()
    
    def add_block(idx):
        current_blocks.append(idx)
        current_masks.append(all_masks[idx])
        current_set.add(idx)
        for bit_idx in block_indices_tuples[idx]:
            counts[bit_idx] += 1

    def remove_block(i):
        for bit_idx in block_indices_tuples[current_blocks[i]]:
            counts[bit_idx] -= 1
        current_set.discard(current_blocks[i])
        current_blocks.pop(i)
        current_masks.pop(i)

    def is_redundant(i):
        for bit_idx in block_indices_tuples[current_blocks[i]]:
            if counts[bit_idx] == 1:
                return False
        return True

    # Vectorized Weighted Greedy with early termination optimization
    uncovered_np = np.ones(N_SUBSETS, dtype=bool)
    n_uncovered = N_SUBSETS
    
    while n_uncovered > 0:
        # Compute gain for each block: sum of weights for uncovered subsets it covers
        # Use the precomputed block_indices_np
        gains = np.zeros(N_BLOCKS, dtype=np.float64)
        for i in range(N_BLOCKS):
            idxs = block_indices_np[i]
            valid = idxs >= 0
            if valid.any():
                idxs_valid = idxs[valid]
                mask_uncovered = uncovered_np[idxs_valid]
                if mask_uncovered.any():
                    gains[i] = weights_np[idxs_valid][mask_uncovered].sum()
        
        best_idx = int(np.argmax(gains))
        if gains[best_idx] <= 0:
            break
        
        add_block(best_idx)
        # Update uncovered mask
        for bit_idx in block_indices_tuples[best_idx]:
            if uncovered_np[bit_idx]:
                uncovered_np[bit_idx] = False
                n_uncovered -= 1
        
    best_len = len(current_blocks)
    best_blocks = current_blocks.copy()
    best_set = set(current_blocks)
    
    # 2. Local Search with Perturbation
    no_improve_count = 0
    max_no_improve = 300
    
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
                    best_set = set(current_blocks)
                break
        
        if removed_any:
            no_improve_count = 0
            continue
            
        # If no redundant block found, try a swap
        if len(current_blocks) == 0: break
        
        # Try 2-for-1 swap: remove two blocks, add one
        if len(current_blocks) >= 2 and no_improve_count > 50:
            i1 = random.randint(0, len(current_blocks) - 1)
            i2 = random.randint(0, len(current_blocks) - 1)
            if i1 == i2:
                i2 = (i2 + 1) % len(current_blocks)
            
            # Compute what would be uncovered if we remove both
            m1 = current_masks[i1]
            m2 = current_masks[i2]
            uncovered_bits = []
            mm = m1 | m2
            while mm:
                lsb = mm & -mm
                bit_idx = lsb.bit_length() - 1
                if counts[bit_idx] <= 2:
                    uncovered_bits.append(bit_idx)
                mm ^= lsb
            
            if uncovered_bits:
                # Find a block that covers all uncovered_bits
                uncovered_bits.sort(key=lambda b: len(blocks_containing_subset[b]))
                candidates = blocks_containing_subset[uncovered_bits[0]]
                
                for b in uncovered_bits[1:]:
                    if not candidates: break
                    set_cands = set(candidates)
                    set_b = set(blocks_containing_subset[b])
                    candidates = list(set_cands.intersection(set_b))
                
                if candidates:
                    valid_candidates = [c for c in candidates if c != i1 and c != i2 and c not in current_set]
                    if valid_candidates:
                        j = random.choice(valid_candidates)
                        # Remove the larger index first to keep indices valid
                        if i1 > i2:
                            i1, i2 = i2, i1
                        remove_block(i1)
                        remove_block(i2)
                        add_block(j)
                        no_improve_count = 0
                        continue
        
        # Pick a random block to remove
        i = random.randint(0, len(current_blocks) - 1)
        
        # Tentatively remove
        m_i = current_masks[i]
        uncovered_bits = []
        for bit_idx in block_indices_tuples[current_blocks[i]]:
            if counts[bit_idx] == 1:
                uncovered_bits.append(bit_idx)
            
        if not uncovered_bits:
            remove_block(i)
            if len(current_blocks) < best_len:
                best_len = len(current_blocks)
                best_blocks = current_blocks.copy()
                best_set = set(current_blocks)
            no_improve_count = 0
            continue
            
        # We need to cover uncovered_bits
        # Try 1-exchange: find a block covering all uncovered_bits
        success = False
        if len(uncovered_bits) <= 5: # Slightly relaxed threshold
            uncovered_bits.sort(key=lambda b: len(blocks_containing_subset[b]))
            candidates = blocks_containing_subset[uncovered_bits[0]]
            
            for b in uncovered_bits[1:]:
                if not candidates: break
                set_cands = set(candidates)
                set_b = set(blocks_containing_subset[b])
                candidates = list(set_cands.intersection(set_b))
                
            if candidates:
                valid_candidates = [c for c in candidates if c != i and c not in current_set]
                if valid_candidates:
                    j = random.choice(valid_candidates)
                    remove_block(i)
                    add_block(j)
                    no_improve_count = 0
                    success = True
        
        if success:
            continue

        # Try 2-exchange: remove i, add two blocks (perturbation)
        if not success and no_improve_count > 100:
            for _ in range(15):
                if time.time() > deadline: break
                j1 = random.randint(0, N_BLOCKS - 1)
                j2 = random.randint(0, N_BLOCKS - 1)
                if j1 == j2 or j1 == i or j2 == i:
                    continue
                if j1 in current_set or j2 in current_set:
                    continue
                
                # Check if j1 and j2 cover all uncovered_bits
                combined_mask = all_masks[j1] | all_masks[j2]
                covers_all = True
                for bit_idx in uncovered_bits:
                    if not (combined_mask & (1 << bit_idx)):
                        covers_all = False
                        break
                
                if covers_all:
                    remove_block(i)
                    add_block(j1)
                    add_block(j2)
                    no_improve_count = 0
                    success = True
                    break
        
        if not success:
            no_improve_count += 1
            if no_improve_count > max_no_improve:
                # Perturbation: Add a random block, then try to prune
                c = random.randint(0, N_BLOCKS - 1)
                if c not in current_set:
                    add_block(c)
                no_improve_count = 0

    # Final cleanup: remove any remaining redundant blocks from best state
    # Reconstruct state from best_blocks
    counts_final = [0] * N_SUBSETS
    for idx in best_blocks:
        for bit_idx in block_indices_tuples[idx]:
            counts_final[bit_idx] += 1
    
    # Try to prune best_blocks
    i = len(best_blocks) - 1
    while i >= 0:
        redundant = True
        for bit_idx in block_indices_tuples[best_blocks[i]]:
            if counts_final[bit_idx] == 1:
                redundant = False
                break
        if redundant:
            # Remove it
            for bit_idx in block_indices_tuples[best_blocks[i]]:
                counts_final[bit_idx] -= 1
            best_blocks.pop(i)
        i -= 1
    
    return [list(all_blocks[i]) for i in best_blocks]
