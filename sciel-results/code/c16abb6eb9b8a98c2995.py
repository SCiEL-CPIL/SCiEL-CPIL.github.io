import itertools
import time
import random
import math

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
        # Simple greedy
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
    
    # 1. Greedy Initialization with tie-breaking by total coverage
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

    # Greedy with tie-breaking: prefer blocks that cover more already-covered subsets
    # (this tends to produce more redundant blocks that can be pruned later)
    uncovered_mask = (1 << N_SUBSETS) - 1
    while uncovered_mask:
        best_idx = -1
        best_gain = 0
        best_total = 0
        for i in range(N_BLOCKS):
            new_cov = all_masks[i] & uncovered_mask
            gain = new_cov.bit_count()
            if gain > best_gain:
                best_gain = gain
                best_idx = i
                best_total = all_masks[i].bit_count()
            elif gain == best_gain and gain > 0:
                # Tie-break: prefer block with more total coverage (more redundancy)
                total = all_masks[i].bit_count()
                if total > best_total:
                    best_total = total
                    best_idx = i
        if best_idx == -1:
            break
        add_block(best_idx)
        uncovered_mask &= ~all_masks[best_idx]
        
    # Initial pruning pass after greedy
    for _ in range(3):
        removed = False
        for i in range(len(current_blocks) - 1, -1, -1):
            if is_redundant(i):
                remove_block(i)
                removed = True
        if not removed:
            break
    
    best_len = len(current_blocks)
    best_blocks = current_blocks.copy()
    
    # 2. Local Search with improved swap heuristics
    
    # Precompute point masks for faster pricing
    point_to_subsets = [[] for _ in range(V)]
    for sub_idx, sub in enumerate(subsets):
        for p in sub:
            point_to_subsets[p].append(sub_idx)
    
    # Precompute for each block: which subsets it uniquely covers (count==1)
    # This helps identify "critical" blocks
    
    no_improve_count = 0
    max_no_improve = 30
    consecutive_swaps = 0
    
    while time.time() < deadline:
        # Phase 1: Aggressive Prune - remove ALL redundant blocks
        removed_any = False
        for i in range(len(current_blocks) - 1, -1, -1):
            if time.time() > deadline:
                break
            if is_redundant(i):
                remove_block(i)
                removed_any = True
        
        if removed_any:
            if len(current_blocks) < best_len:
                best_len = len(current_blocks)
                best_blocks = current_blocks.copy()
            no_improve_count = 0
            consecutive_swaps = 0
            continue
            
        # Phase 2: Smart Swap
        if len(current_blocks) == 0:
            break
            
        # Find blocks with most "unique" coverage (hardest to replace)
        # and blocks with least unique coverage (easiest to remove)
        
        # Pick a block to remove: prefer blocks with fewer uniquely-covered subsets
        block_unique_counts = []
        for i in range(len(current_blocks)):
            m_i = current_masks[i]
            unique_count = 0
            mm = m_i
            while mm:
                lsb = mm & -mm
                bit_idx = lsb.bit_length() - 1
                if counts[bit_idx] == 1:
                    unique_count += 1
                mm ^= lsb
            block_unique_counts.append(unique_count)
        
        # Sort by unique count ascending (remove easiest to replace first)
        sorted_blocks = sorted(range(len(current_blocks)), key=lambda x: block_unique_counts[x])
        
        # Try removing from the easiest blocks first
        swap_done = False
        for i in sorted_blocks:
            if time.time() > deadline:
                break
            if block_unique_counts[i] == 0:
                # This block is completely redundant (shouldn't happen after prune, but check)
                remove_block(i)
                if len(current_blocks) < best_len:
                    best_len = len(current_blocks)
                    best_blocks = current_blocks.copy()
                no_improve_count = 0
                consecutive_swaps = 0
                swap_done = True
                break
        
        if swap_done:
            continue
        
        # Try a swap: remove one block, add another that covers its unique subsets
        # Pick from blocks with moderate unique count (not too hard to replace)
        candidate_removals = [i for i in sorted_blocks if block_unique_counts[i] <= 3]
        
        if not candidate_removals:
            # If all blocks have many unique subsets, just try random
            i = random.randint(0, len(current_blocks) - 1)
        else:
            i = random.choice(candidate_removals)
        
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
            remove_block(i)
            if len(current_blocks) < best_len:
                best_len = len(current_blocks)
                best_blocks = current_blocks.copy()
            no_improve_count = 0
            consecutive_swaps = 0
            continue
        
        if len(uncovered_bits) > 5:
            no_improve_count += 1
            consecutive_swaps += 1
            if no_improve_count > max_no_improve:
                # Perturbation: add a random block, then prune
                c = random.randint(0, N_BLOCKS - 1)
                if c not in current_blocks:
                    add_block(c)
                    no_improve_count = 0
                    max_no_improve = min(100, max_no_improve * 2)
                    consecutive_swaps = 0
                continue
        else:
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
                # Score candidates by how much additional coverage they provide
                best_j = -1
                best_score = -1
                current_set = set(current_blocks)
                
                for c in candidates:
                    if c == i or c in current_set:
                        continue
                    # Score: number of new subsets covered (not in current coverage)
                    score = 0
                    m_c = all_masks[c]
                    mm_c = m_c
                    while mm_c:
                        lsb = mm_c & -mm_c
                        bit_idx = lsb.bit_length() - 1
                        if counts[bit_idx] == 0:
                            score += 10  # High priority for uncovered
                        elif counts[bit_idx] == 1:
                            score += 1   # Low priority for barely covered
                        mm_c ^= lsb
                    
                    if score > best_score:
                        best_score = score
                        best_j = c
                
                if best_j != -1:
                    remove_block(i)
                    add_block(best_j)
                    no_improve_count = 0
                    consecutive_swaps = 0
                    continue
                else:
                    no_improve_count += 1
            else:
                no_improve_count += 1
        
        if no_improve_count > max_no_improve:
            # Perturbation: add a random block, then prune
            c = random.randint(0, N_BLOCKS - 1)
            if c not in current_blocks:
                add_block(c)
            no_improve_count = 0
            max_no_improve = min(100, max_no_improve * 2)
            consecutive_swaps = 0

    # Final aggressive pruning pass
    for _ in range(5):
        removed = False
        for i in range(len(current_blocks) - 1, -1, -1):
            if is_redundant(i):
                remove_block(i)
                removed = True
        if not removed:
            break
    
    if len(current_blocks) < best_len:
        best_len = len(current_blocks)
        best_blocks = current_blocks.copy()
    
    # Return the best found
    return [list(all_blocks[i]) for i in best_blocks]
