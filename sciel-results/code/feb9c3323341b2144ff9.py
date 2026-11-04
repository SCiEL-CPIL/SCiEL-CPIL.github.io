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

# Precompute subset sizes for blocks (number of 4-subsets covered)
block_sizes = [len(s) for s in block_subsets]

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
    
    def greedy_construct(alpha=1.0):
        """Greedy construction with logarithmic gain when alpha > 1."""
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

        # Greedy with log-gain: maximize log(gain + 1) / cost
        # This gives diminishing returns, favoring blocks that cover new subsets
        uncovered_mask = (1 << N_SUBSETS) - 1
        n_uncovered = N_SUBSETS
        
        while uncovered_mask:
            best_idx = -1
            best_score = -1.0
            for i in range(N_BLOCKS):
                gain = (all_masks[i] & uncovered_mask).bit_count()
                if gain == 0:
                    continue
                if alpha > 1.0:
                    # Logarithmic gain: log(gain + 1)
                    score = math.log(gain + 1) / block_sizes[i]
                else:
                    # Linear gain
                    score = gain / block_sizes[i]
                if score > best_score:
                    best_score = score
                    best_idx = i
            if best_idx == -1:
                break
            add_block(best_idx)
            uncovered_mask &= ~all_masks[best_idx]
        
        return current_blocks, current_masks, counts

    def local_search(current_blocks, current_masks, counts, deadline, best_len):
        """Local search: remove redundant blocks, then try swaps."""
        best_blocks = current_blocks.copy()
        no_improve_count = 0
        max_no_improve = 50
        
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

        def try_remove_redundant():
            nonlocal best_blocks
            # Try to find and remove a redundant block
            for i in range(len(current_blocks)):
                if time.time() > deadline:
                    return False
                if is_redundant(i):
                    # Remove block i
                    m = current_masks[i]
                    mm = m
                    while mm:
                        lsb = mm & -mm
                        bit_idx = lsb.bit_length() - 1
                        counts[bit_idx] -= 1
                        mm ^= lsb
                    current_blocks.pop(i)
                    current_masks.pop(i)
                    if len(current_blocks) < len(best_blocks):
                        best_blocks = current_blocks.copy()
                    return True
            return False

        def try_swap():
            nonlocal best_blocks
            if len(current_blocks) == 0:
                return False
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
                # Block is redundant, remove it
                mm = m_i
                while mm:
                    lsb = mm & -mm
                    bit_idx = lsb.bit_length() - 1
                    counts[bit_idx] -= 1
                    mm ^= lsb
                current_blocks.pop(i)
                current_masks.pop(i)
                if len(current_blocks) < len(best_blocks):
                    best_blocks = current_blocks.copy()
                return True
            
            # Need to cover uncovered_bits
            if len(uncovered_bits) > 4:
                return False
            
            # Find blocks that cover all uncovered_bits
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
                    # Remove i
                    mm = m_i
                    while mm:
                        lsb = mm & -mm
                        bit_idx = lsb.bit_length() - 1
                        counts[bit_idx] -= 1
                        mm ^= lsb
                    current_blocks.pop(i)
                    current_masks.pop(i)
                    # Add j
                    current_blocks.append(j)
                    current_masks.append(all_masks[j])
                    m_j = all_masks[j]
                    mm = m_j
                    while mm:
                        lsb = mm & -mm
                        bit_idx = lsb.bit_length() - 1
                        counts[bit_idx] += 1
                        mm ^= lsb
                    return True
            return False

        while time.time() < deadline:
            # Phase 1: Remove redundant blocks
            found_redundant = False
            for _ in range(20):
                if time.time() > deadline:
                    break
                if try_remove_redundant():
                    found_redundant = True
                else:
                    break
            
            if found_redundant:
                no_improve_count = 0
                continue
            
            # Phase 2: Try swaps
            if try_swap():
                no_improve_count = 0
                continue
            
            # Phase 3: Perturbation
            no_improve_count += 1
            if no_improve_count > max_no_improve:
                # Add a random block
                c = random.randint(0, N_BLOCKS - 1)
                if c not in current_blocks:
                    current_blocks.append(c)
                    current_masks.append(all_masks[c])
                    m = all_masks[c]
                    mm = m
                    while mm:
                        lsb = mm & -mm
                        bit_idx = lsb.bit_length() - 1
                        counts[bit_idx] += 1
                        mm ^= lsb
                no_improve_count = 0
        
        return best_blocks

    # Try multiple greedy strategies with different alpha values
    best_overall = None
    best_overall_len = float('inf')
    
    for alpha in [1.0, 1.5, 2.0]:
        if time.time() > deadline:
            break
        current_blocks, current_masks, counts = greedy_construct(alpha)
        if len(current_blocks) < best_overall_len:
            best_overall_len = len(current_blocks)
            best_overall = current_blocks.copy()
        
        # Local search
        improved = local_search(current_blocks, current_masks, counts, deadline, best_overall_len)
        if len(improved) < best_overall_len:
            best_overall_len = len(improved)
            best_overall = improved.copy()
    
    if best_overall is None:
        return []
    
    # Final cleanup: remove any remaining redundant blocks
    final_counts = [0] * N_SUBSETS
    for idx in best_overall:
        m = all_masks[idx]
        mm = m
        while mm:
            lsb = mm & -mm
            bit_idx = lsb.bit_length() - 1
            final_counts[bit_idx] += 1
            mm ^= lsb
    
    final_blocks = best_overall.copy()
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
            mm = m
            while mm:
                lsb = mm & -mm
                bit_idx = lsb.bit_length() - 1
                final_counts[bit_idx] -= 1
                mm ^= lsb
            final_blocks.pop(i)
    
    return [list(all_blocks[i]) for i in final_blocks]
