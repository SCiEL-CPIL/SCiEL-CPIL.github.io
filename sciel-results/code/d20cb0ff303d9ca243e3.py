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
all_masks = []
blocks_containing_subset = [[] for _ in range(N_SUBSETS)]

for bi, block in enumerate(all_blocks):
    mask = 0
    for sub in itertools.combinations(block, T):
        idx = subset_to_idx[sub]
        mask |= 1 << idx
        blocks_containing_subset[idx].append(bi)
    all_masks.append(mask)

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
    counts = [0] * N_SUBSETS
    current_blocks = []
    current_masks = []
    
    def add_block(idx):
        current_blocks.append(idx)
        m = all_masks[idx]
        current_masks.append(m)
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

    # 1. Greedy Initialization (with randomization for diversity)
    uncovered_mask = (1 << N_SUBSETS) - 1
    while uncovered_mask:
        best_idx = -1
        best_gain = 0
        # Sample a subset of blocks to speed up greedy if needed, but N_BLOCKS=50388 is manageable
        # Let's do a full scan but with a bit of randomization in tie-breaking
        for i in range(N_BLOCKS):
            gain = (all_masks[i] & uncovered_mask).bit_count()
            if gain > best_gain or (gain == best_gain and random.random() < 0.05):
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
    max_no_improve = 30 # Threshold to trigger perturbation
    temperature = 1.0
    
    while time.time() < deadline:
        # Phase 1: Aggressive Pruning
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
                # Restarting the loop is safer after removal to check new redundancies
                break
        
        if removed_any:
            no_improve_count = 0
            continue
            
        # Phase 2: Swap Search
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
        if len(uncovered_bits) > 3:
            no_improve_count += 1
            if no_improve_count > max_no_improve:
                # Perturbation: Remove a random block and add a random different one?
                # Or just add a random block to create new redundancy opportunities?
                # Let's try: Remove 2 random blocks, then we MUST cover their unique parts.
                # Simpler perturbation: Add a random block that covers a lot of 'count=1' bits?
                # Actually, just a random addition is a valid perturbation to break symmetry.
                c = random.randint(0, N_BLOCKS - 1)
                if c not in current_blocks:
                    add_block(c)
                    no_improve_count = 0
                    # Reset max_no_improve to allow quick re-perturbation if it fails
                    max_no_improve = 10 
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
                set_cands = set(candidates)
                set_b = set(blocks_containing_subset[b])
                candidates = list(set_cands.intersection(set_b))
                
            if candidates:
                # Pick a random candidate
                # Exclude i if it's in candidates (it covers the bits, so it should be, unless we removed it? No, we haven't removed it yet)
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
                        max_no_improve = 10
                    continue

    # Final cleanup: ensure we return the best found
    return [list(all_blocks[i]) for i in best_blocks]
