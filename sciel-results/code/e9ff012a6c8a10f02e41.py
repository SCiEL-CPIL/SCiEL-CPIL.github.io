import itertools
import time
import random
import sys
import math

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin

    # 1. Precomputation
    all_t_subsets = list(itertools.combinations(range(v), t))
    num_t_subsets = len(all_t_subsets)
    t_subset_to_idx = {s: i for i, s in enumerate(all_t_subsets)}
    
    all_blocks = list(itertools.combinations(range(v), k))
    num_blocks = len(all_blocks)
    
    # Precompute bitmask for each block
    block_masks = [0] * num_blocks
    # t_to_blocks[j] = list of block indices that cover t-subset j
    t_to_blocks = [[] for _ in range(num_t_subsets)]
    
    for b_idx, block in enumerate(all_blocks):
        mask = 0
        for sub in itertools.combinations(block, t):
            idx = t_subset_to_idx[sub]
            mask |= (1 << idx)
            t_to_blocks[idx].append(b_idx)
        block_masks[b_idx] = mask

    best_blocks = None
    best_count = float('inf')
    
    # Helper: Extract set bits from mask
    def extract_bits(mask):
        bits = []
        while mask:
            lsb = mask & -mask
            bits.append(lsb.bit_length() - 1)
            mask ^= lsb
        return bits

    # Helper: Prune redundant blocks
    # Returns list of remaining block indices
    def prune_redundant(selected_indices):
        if not selected_indices:
            return []
        
        # Count coverage for each t-subset
        cover_counts = [0] * num_t_subsets
        for b_idx in selected_indices:
            m = block_masks[b_idx]
            while m:
                lsb = m & -m
                idx = lsb.bit_length() - 1
                cover_counts[idx] += 1
                m ^= lsb
        
        improved = True
        current_blocks = list(selected_indices)
        
        while improved and time.time() < deadline:
            improved = False
            new_blocks = []
            for b_idx in current_blocks:
                # Check if redundant: all t-subsets in this block have count > 1
                redundant = True
                m = block_masks[b_idx]
                while m:
                    lsb = m & -m
                    idx = lsb.bit_length() - 1
                    if cover_counts[idx] == 1:
                        redundant = False
                        break
                    m ^= lsb
                
                if redundant:
                    # Remove it
                    m = block_masks[b_idx]
                    while m:
                        lsb = m & -m
                        idx = lsb.bit_length() - 1
                        cover_counts[idx] -= 1
                        m ^= lsb
                    improved = True
                else:
                    new_blocks.append(b_idx)
            current_blocks = new_blocks
            
        return current_blocks

    # Helper: Local Search with Swaps
    def local_search_improve(block_indices, max_time=5.0):
        current_blocks = list(block_indices)
        selected_arr = [False] * num_blocks
        for b_idx in current_blocks:
            selected_arr[b_idx] = True

        # Maintain cover_counts for efficiency
        cover_counts = [0] * num_t_subsets
        for b_idx in current_blocks:
            m = block_masks[b_idx]
            while m:
                lsb = m & -m
                idx = lsb.bit_length() - 1
                cover_counts[idx] += 1
                m ^= lsb

        improved = True
        last_change_time = time.time()

        while improved and time.time() < deadline and (time.time() - last_change_time) < max_time:
            improved = False
            random.shuffle(current_blocks)

            for b_old in current_blocks[:]:
                if time.time() > deadline:
                    return current_blocks

                # Find unique t-subsets covered by b_old
                unique_mask = 0
                m = block_masks[b_old]
                while m:
                    lsb = m & -m
                    idx = lsb.bit_length() - 1
                    if cover_counts[idx] == 1:
                        unique_mask |= lsb
                    m ^= lsb

                if unique_mask == 0:
                    # This block is redundant, remove it
                    m = block_masks[b_old]
                    while m:
                        lsb = m & -m
                        idx = lsb.bit_length() - 1
                        cover_counts[idx] -= 1
                        m ^= lsb
                    current_blocks.remove(b_old)
                    selected_arr[b_old] = False
                    improved = True
                    last_change_time = time.time()
                    continue

                # Try to find a block b_new that covers unique_mask
                # Collect all unique t-subset indices, sort by rarity
                unique_bits = []
                tmp = unique_mask
                while tmp:
                    lsb = tmp & -tmp
                    unique_bits.append(lsb.bit_length() - 1)
                    tmp ^= lsb
                unique_bits.sort(key=lambda x: cover_counts[x])

                # Try each unique t-subset as a guide (up to 3 rarest)
                found_swap = False
                for target_t_idx in unique_bits[:3]:
                    if found_swap:
                        break
                    candidates = t_to_blocks[target_t_idx]
                    random.shuffle(candidates)
                    for b_new in candidates:
                        if selected_arr[b_new]:
                            continue
                        if (block_masks[b_new] & unique_mask) == unique_mask:
                            # Valid swap
                            m = block_masks[b_old]
                            while m:
                                lsb = m & -m
                                idx = lsb.bit_length() - 1
                                cover_counts[idx] -= 1
                                m ^= lsb
                            m = block_masks[b_new]
                            while m:
                                lsb = m & -m
                                idx = lsb.bit_length() - 1
                                cover_counts[idx] += 1
                                m ^= lsb
                            current_blocks.remove(b_old)
                            current_blocks.append(b_new)
                            selected_arr[b_old] = False
                            selected_arr[b_new] = True
                            improved = True
                            last_change_time = time.time()
                            found_swap = True
                            break

        return current_blocks

    # Helper: 2-for-1 replacement — find pair of blocks whose union of unique
    # t-subsets is covered by a single unselected block, then swap 2 -> 1.
    def two_for_one_improve(block_indices, max_time=8.0):
        current_blocks = list(block_indices)
        selected_arr = [False] * num_blocks
        for b_idx in current_blocks:
            selected_arr[b_idx] = True

        cover_counts = [0] * num_t_subsets
        for b_idx in current_blocks:
            m = block_masks[b_idx]
            while m:
                lsb = m & -m
                idx = lsb.bit_length() - 1
                cover_counts[idx] += 1
                m ^= lsb

        improved = True
        last_change = time.time()
        while improved and time.time() < deadline and (time.time() - last_change) < max_time:
            improved = False
            n = len(current_blocks)
            for i in range(n):
                if time.time() > deadline:
                    return current_blocks
                b_i = current_blocks[i]
                if not selected_arr[b_i]:
                    continue
                # unique t-subsets for b_i
                uni_i = 0
                m = block_masks[b_i]
                while m:
                    lsb = m & -m
                    idx = lsb.bit_length() - 1
                    if cover_counts[idx] == 1:
                        uni_i |= lsb
                    m ^= lsb
                if uni_i == 0:
                    continue  # already redundant; prune handles it
                for j in range(i + 1, n):
                    b_j = current_blocks[j]
                    if not selected_arr[b_j]:
                        continue
                    uni_j = 0
                    m = block_masks[b_j]
                    while m:
                        lsb = m & -m
                        idx = lsb.bit_length() - 1
                        if cover_counts[idx] == 1:
                            uni_j |= lsb
                        m ^= lsb
                    if uni_j == 0:
                        continue
                    combined = uni_i | uni_j
                    if combined == 0:
                        continue
                    # Try to find one block covering 'combined'
                    # Pick the rarest t-subset in combined to limit candidates
                    bits = []
                    tmp = combined
                    while tmp:
                        lsb = tmp & -tmp
                        bits.append(lsb.bit_length() - 1)
                        tmp ^= lsb
                    # Sort by cover count (ascending) to pick rarest
                    bits.sort(key=lambda x: cover_counts[x])
                    target = bits[0]
                    candidates = t_to_blocks[target]
                    random.shuffle(candidates)
                    found = False
                    for b_new in candidates:
                        if selected_arr[b_new]:
                            continue
                        if (block_masks[b_new] & combined) == combined:
                            # Valid 2-for-1 swap
                            # Remove b_i and b_j
                            for b_old in (b_i, b_j):
                                m = block_masks[b_old]
                                while m:
                                    lsb = m & -m
                                    idx = lsb.bit_length() - 1
                                    cover_counts[idx] -= 1
                                    m ^= lsb
                                selected_arr[b_old] = False
                            # Add b_new
                            m = block_masks[b_new]
                            while m:
                                lsb = m & -m
                                idx = lsb.bit_length() - 1
                                cover_counts[idx] += 1
                                m ^= lsb
                            selected_arr[b_new] = True
                            current_blocks.remove(b_i)
                            current_blocks.remove(b_j)
                            current_blocks.append(b_new)
                            improved = True
                            last_change = time.time()
                            found = True
                            break
                    if found:
                        break  # restart pair scan with updated counts
        return current_blocks

    # Main Loop: Multiple Greedy Instances
    max_runs = 100 # Adjust based on speed

    for run_idx in range(max_runs):
        if time.time() > deadline:
            break

        random.seed(run_idx)

        # --- Phase 1: Fast Greedy ---
        uncovered_mask = (1 << num_t_subsets) - 1
        selected_indices = []
        selected_arr = [False] * num_blocks

        # Initial gain: C(k, t)
        init_gain = len(list(itertools.combinations(range(k), t)))
        gain = [init_gain] * num_blocks

        while uncovered_mask:
            if time.time() > deadline:
                break

            # Find max gain
            max_gain = -1
            best_candidates = []

            # Linear scan with boolean array (O(1) lookup vs O(1) set but faster in practice)
            for b_idx in range(num_blocks):
                if selected_arr[b_idx]:
                    continue
                g = gain[b_idx]
                # Add small noise for randomization
                g += random.random() * 0.01
                if g > max_gain:
                    max_gain = g
                    best_candidates = [b_idx]
                elif abs(g - max_gain) < 1e-9:
                    best_candidates.append(b_idx)

            if not best_candidates:
                break

            chosen_idx = random.choice(best_candidates)
            selected_indices.append(chosen_idx)
            selected_arr[chosen_idx] = True

            # Update uncovered and gains
            newly_covered = uncovered_mask & block_masks[chosen_idx]
            uncovered_mask &= ~block_masks[chosen_idx]

            # Update gains for affected blocks
            m = newly_covered
            while m:
                lsb = m & -m
                t_idx = lsb.bit_length() - 1
                m ^= lsb
                for b_prime in t_to_blocks[t_idx]:
                    if not selected_arr[b_prime]:
                        gain[b_prime] -= 1
                        if gain[b_prime] < 0:
                            gain[b_prime] = 0

        if uncovered_mask != 0:
            continue

        # --- Phase 2: Prune ---
        selected_indices = prune_redundant(selected_indices)

        # --- Phase 3: Local Search (Swaps) ---
        if time.time() < deadline - 15.0:
            selected_indices = local_search_improve(selected_indices)

        # --- Phase 4: 2-for-1 Replacement ---
        if time.time() < deadline - 5.0:
            selected_indices = two_for_one_improve(selected_indices)

        # Final prune after 2-for-1
        selected_indices = prune_redundant(selected_indices)

        if len(selected_indices) < best_count:
            best_count = len(selected_indices)
            best_blocks = [list(all_blocks[i]) for i in selected_indices]

    if best_blocks is None:
        return []
        
    return best_blocks
