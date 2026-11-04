import itertools
import time
import numpy as np

def construct(v, k, t):
    deadline = time.time() + 115  # Safety margin
    
    # 1. Precompute all t-subsets and map to index
    all_t_subs = list(itertools.combinations(range(v), t))
    n_t_subs = len(all_t_subs)
    sub_to_idx = {s: i for i, s in enumerate(all_t_subs)}
    
    # 2. Precompute all k-blocks
    all_k_blocks = list(itertools.combinations(range(v), k))
    n_k_blocks = len(all_k_blocks)
    
    # 3. Precompute coverage masks for blocks and reverse index for subsets
    # Use numpy arrays of uint64 for efficient bit operations
    # Each t-subset gets a bit position; we split into chunks of 64
    n_words = (n_t_subs + 63) // 64
    block_masks_np = np.zeros((n_k_blocks, n_words), dtype=np.uint64)
    blocks_containing_subset = [[] for _ in range(n_t_subs)]
    
    for i, block in enumerate(all_k_blocks):
        for t_sub in itertools.combinations(block, t):
            idx = sub_to_idx[t_sub]
            word = idx // 64
            bit = idx % 64
            block_masks_np[i, word] |= (1 << bit)
            blocks_containing_subset[idx].append(i)
    
    # 4. Greedy Construction with numpy for speed
    # uncovered as numpy array of uint64
    uncovered = np.ones(n_words, dtype=np.uint64)
    # Set bits properly: for the last word, mask off unused bits
    if n_t_subs % 64 != 0:
        uncovered[-1] = (1 << (n_t_subs % 64)) - 1
    
    selected_block_indices = []
    
    while True:
        if time.time() > deadline - 30:
            break
        
        # Check if all covered
        if not np.any(uncovered):
            break
        
        # Find block with maximum gain using numpy
        # For each block, count how many uncovered bits it covers
        # gain = sum over words of popcount(block_masks[i, w] & uncovered[w])
        # This is expensive in pure Python; let's do it in a loop but optimized
        
        best_idx = -1
        best_gain = -1
        
        # Precompute uncovered as list for faster access
        unc_list = uncovered.tolist()
        
        for i in range(n_k_blocks):
            bm = block_masks_np[i]
            gain = 0
            for w in range(n_words):
                gain += (bm[w] & unc_list[w]).bit_count()
            if gain > best_gain:
                best_gain = gain
                best_idx = i
        
        if best_idx == -1 or best_gain == 0:
            break
        
        selected_block_indices.append(best_idx)
        # Update uncovered: uncovered &= ~block_masks[best_idx]
        bm = block_masks_np[best_idx]
        for w in range(n_words):
            uncovered[w] &= ~bm[w]
    
    # 5. Local Search
    current_blocks = set(selected_block_indices)
    
    def get_unique_subsets(block_idx, cover_count):
        """Get t-subsets only covered by this block."""
        bm = block_masks_np[block_idx]
        uniques = []
        for w in range(n_words):
            m = bm[w]
            while m:
                lsb = m & -m
                bit_pos = lsb.bit_length() - 1
                idx = w * 64 + bit_pos
                if idx < n_t_subs and cover_count[idx] == 1:
                    uniques.append(idx)
                m ^= lsb
        return uniques
    
    def compute_cover_counts(block_set):
        """Compute how many times each t-subset is covered."""
        cover_count = np.zeros(n_t_subs, dtype=np.int32)
        for b_idx in block_set:
            bm = block_masks_np[b_idx]
            for w in range(n_words):
                m = bm[w]
                while m:
                    lsb = m & -m
                    bit_pos = lsb.bit_length() - 1
                    idx = w * 64 + bit_pos
                    if idx < n_t_subs:
                        cover_count[idx] += 1
                    m ^= lsb
        return cover_count
    
    improved = True
    while improved and time.time() < deadline - 10:
        improved = False
        
        # Compute cover counts
        cover_count = compute_cover_counts(current_blocks)
        
        # Pass 1: Redundancy Removal
        blocks_to_remove = []
        for b_idx in list(current_blocks):
            uniques = get_unique_subsets(b_idx, cover_count)
            if not uniques:
                blocks_to_remove.append(b_idx)
                
        if blocks_to_remove:
            for b_idx in blocks_to_remove:
                current_blocks.remove(b_idx)
            improved = True
            continue
        
        # Precompute unique subsets for all blocks
        unique_subsets_map = {}
        for b_idx in current_blocks:
            unique_subsets_map[b_idx] = get_unique_subsets(b_idx, cover_count)
        
        # Pass 2: 1-to-1 Swap
        # For each block i, find if there's a block C not in current_blocks
        # that covers all unique subsets of i, and C is "better" 
        # (covers more uncovered or is just valid)
        # Actually, 1-to-1 swap: replace block i with block C where C covers
        # all unique subsets of i. This doesn't reduce count but might enable
        # further reductions. Let's skip this for now and focus on 2-to-1.
        
        # Pass 3: 2-to-1 or 2-to-0 Merge
        blocks_list = list(current_blocks)
        n_blocks = len(blocks_list)
        
        found_merge = False
        for i_idx in range(n_blocks):
            if time.time() > deadline - 15:
                break
            b_i = blocks_list[i_idx]
            uniques_i = unique_subsets_map[b_i]
            
            for j_idx in range(i_idx + 1, n_blocks):
                b_j = blocks_list[j_idx]
                uniques_j = unique_subsets_map[b_j]
                
                union_uniques = set(uniques_i) | set(uniques_j)
                
                if not union_uniques:
                    current_blocks.remove(b_i)
                    current_blocks.remove(b_j)
                    improved = True
                    found_merge = True
                    break
                    
                # Find a block C covering all subsets in union_uniques
                sorted_uniques = sorted(union_uniques, key=lambda x: len(blocks_containing_subset[x]))
                
                # Start with the smallest list
                candidate_blocks = set(blocks_containing_subset[sorted_uniques[0]])
                
                for s_idx in sorted_uniques[1:]:
                    if not candidate_blocks:
                        break
                    candidate_blocks &= set(blocks_containing_subset[s_idx])
                    
                if candidate_blocks:
                    c_block = next(iter(candidate_blocks))
                    
                    if c_block in current_blocks:
                        current_blocks.remove(b_i)
                        current_blocks.remove(b_j)
                    else:
                        current_blocks.remove(b_i)
                        current_blocks.remove(b_j)
                        current_blocks.add(c_block)
                        
                    improved = True
                    found_merge = True
                    break
                    
            if found_merge:
                break
        
        # Pass 4: 1-to-1 Swap (replace block i with block C that covers i's uniques)
        if not improved and time.time() < deadline - 15:
            for b_i in list(current_blocks):
                if time.time() > deadline - 15:
                    break
                uniques_i = unique_subsets_map.get(b_i, get_unique_subsets(b_i, cover_count))
                if not uniques_i:
                    continue
                    
                sorted_uniques = sorted(uniques_i, key=lambda x: len(blocks_containing_subset[x]))
                candidate_blocks = set(blocks_containing_subset[sorted_uniques[0]])
                
                for s_idx in sorted_uniques[1:]:
                    if not candidate_blocks:
                        break
                    candidate_blocks &= set(blocks_containing_subset[s_idx])
                    
                # Remove b_i from candidates since we're replacing it
                candidate_blocks.discard(b_i)
                
                if candidate_blocks:
                    # Pick a candidate that is not already in current_blocks
                    # to avoid no-op
                    new_candidates = [c for c in candidate_blocks if c not in current_blocks]
                    if new_candidates:
                        c_block = new_candidates[0]
                        current_blocks.remove(b_i)
                        current_blocks.add(c_block)
                        improved = True
                        break
    
    # Final validity check and repair if needed
    blocks = [list(all_k_blocks[i]) for i in current_blocks]
    
    # Verify coverage
    covered = set()
    for block in blocks:
        for t_sub in itertools.combinations(block, t):
            covered.add(t_sub)
    
    if len(covered) < n_t_subs:
        # Repair: add blocks greedily for uncovered subsets
        uncovered_set = set(all_t_subs) - covered
        repair_deadline = deadline - 2
        while uncovered_set and time.time() < repair_deadline:
            best_idx = -1
            best_gain = 0
            for i in range(n_k_blocks):
                if i in current_blocks:
                    continue
                gain = 0
                for t_sub in itertools.combinations(all_k_blocks[i], t):
                    if t_sub in uncovered_set:
                        gain += 1
                if gain > best_gain:
                    best_gain = gain
                    best_idx = i
            if best_idx == -1 or best_gain == 0:
                break
            current_blocks.add(best_idx)
            for t_sub in itertools.combinations(all_k_blocks[best_idx], t):
                uncovered_set.discard(t_sub)
        
        blocks = [list(all_k_blocks[i]) for i in current_blocks]
    
    return blocks
