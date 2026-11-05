import itertools
import random
import time
import numpy as np

# Global Precomputation
V = 19
K = 7
T = 4

# Map 4-subsets to indices
subsets = list(itertools.combinations(range(V), T))
subset_to_idx = {s: i for i, s in enumerate(subsets)}
N_SUBSETS = len(subsets)

# Generate all 7-subsets
all_blocks = list(itertools.combinations(range(V), K))
N_BLOCKS = len(all_blocks)

# Precompute masks as Python ints for fast bit_count
# Also precompute list of subsets for each block
# And list of blocks containing each subset
all_masks = [0] * N_BLOCKS
blocks_containing_subset = [[] for _ in range(N_SUBSETS)]
# Precompute list of subset indices for each block (for faster gain calc if needed)
block_subset_indices = [[] for _ in range(N_BLOCKS)]

for bi, block in enumerate(all_blocks):
    mask = 0
    for sub in itertools.combinations(block, T):
        idx = subset_to_idx[sub]
        mask |= 1 << idx
        blocks_containing_subset[idx].append(bi)
        block_subset_indices[bi].append(idx)
    all_masks[bi] = mask

def construct(v, k, t):
    if v != 19 or k != 7 or t != 4:
        # Fallback for other parameters
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
    deadline = start_time + 115  # Safety margin before 120s limit
    
    # State
    counts = np.zeros(N_SUBSETS, dtype=np.int32)
    current_blocks = []
    current_masks = []
    
    def add_block(idx):
        current_blocks.append(idx)
        m = all_masks[idx]
        current_masks.append(m)
        # Update counts
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
        if i == len(current_blocks) - 1:
            current_blocks.pop()
            current_masks.pop()
        else:
            current_blocks[i] = current_blocks[-1]
            current_masks[i] = current_masks[-1]
            current_blocks.pop()
            current_masks.pop()

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

    # 1. Greedy Initialization
    uncovered_mask = (1 << N_SUBSETS) - 1
    while uncovered_mask:
        best_idx = -1
        best_gain = 0
        # Optimization: Precompute gains using numpy? 
        # all_masks is a list of big ints. 
        # We can convert uncovered_mask to a numpy array of bytes? No, bit_count is fast.
        # Let's stick to the loop but maybe skip blocks that are already in current_blocks?
        # No, greedy doesn't add duplicates.
        
        # Vectorized gain calculation:
        # We can't easily vectorize big int AND + bit_count in numpy without converting to arrays of bytes, which is slow.
        # Let's try to speed up the loop.
        
        for i in range(N_BLOCKS):
            gain = (all_masks[i] & uncovered_mask).bit_count()
            if gain > best_gain:
                best_gain = gain
                best_idx = i
                if best_gain == 35:
                    break
        if best_idx == -1:
            break
        add_block(best_idx)
        uncovered_mask &= ~all_masks[best_idx]
        
    best_len = len(current_blocks)
    best_blocks = current_blocks.copy()
    
    # 2. Local Search with Perturbation
    no_improve_count = 0
    max_no_improve = 50
    # To allow more search, we can increase max_no_improve or improve perturbation.
    # Let's try a slightly more aggressive perturbation: remove 2 random blocks and add 2 random blocks?
    # Or just stick to single block swap but with better candidate selection.
    
    while time.time() < deadline:
        # Phase 1: Aggressive Pruning
        removed_any = False
        indices = list(range(len(current_blocks)))
        random.shuffle(indices)
        
        for i in indices:
            if time.time() > deadline:
                break
            if i >= len(current_blocks):
                continue
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
            
        # Phase 2: Swap Search
        if len(current_blocks) == 0:
            break
            
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
            
        if len(uncovered_bits) > 5:
            no_improve_count += 1
            if no_improve_count > max_no_improve:
                # Perturbation
                c = random.randint(0, N_BLOCKS - 1)
                if c not in current_blocks:
                    add_block(c)
                    no_improve_count = 0
                    max_no_improve = 20
                continue
        else:
            uncovered_bits.sort(key=lambda b: len(blocks_containing_subset[b]))
            
            candidates = blocks_containing_subset[uncovered_bits[0]]
            
            for b in uncovered_bits[1:]:
                if not candidates:
                    break
                set_cands = set(candidates)
                set_b = set(blocks_containing_subset[b])
                candidates = list(set_cands.intersection(set_b))
                
            if candidates:
                valid_candidates = [c for c in candidates if c != i]
                if valid_candidates:
                    j = random.choice(valid_candidates)
                    remove_block(i)
                    add_block(j)
                    no_improve_count = 0
                    continue
            else:
                no_improve_count += 1
                if no_improve_count > max_no_improve:
                    c = random.randint(0, N_BLOCKS - 1)
                    if c not in current_blocks:
                        add_block(c)
                        no_improve_count = 0
                        max_no_improve = 20
                    continue

    return [list(all_blocks[i]) for i in best_blocks]
