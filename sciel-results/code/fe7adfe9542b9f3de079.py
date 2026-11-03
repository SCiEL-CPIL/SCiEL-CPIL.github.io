import itertools
import time
import random
import heapq
import numpy as np

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin
    
    total_space = q ** n
    
    # Precompute word to index and index to word
    def word_to_idx(w):
        idx = 0
        for v in w:
            idx = idx * q + v
        return idx

    def idx_to_word(idx):
        w = []
        for i in range(n):
            w.append(idx % q)
            idx //= q
        w.reverse()
        return w

    # Precompute all words
    all_words = [idx_to_word(i) for i in range(total_space)]
    
    # Precompute cover masks
    # mask[i] is a bitmask of indices of words within Hamming distance <= radius of word i
    cover_masks = [0] * total_space
    for i in range(total_space):
        w = all_words[i]
        mask = 1 << i
        if radius >= 1:
            for pos in range(n):
                orig_val = w[pos]
                for val in range(q):
                    if val == orig_val:
                        continue
                    nw = w[:]
                    nw[pos] = val
                    j = word_to_idx(nw)
                    mask |= (1 << j)
        cover_masks[i] = mask
        
    full_mask = (1 << total_space) - 1
    
    # Helper to count bits
    # Using int.bit_count() is efficient
    
    def greedy_cover(alpha=0.0, randomize_ties=False):
        uncovered = full_mask
        covered_union = 0
        code = []
        used = [False] * total_space
        
        while uncovered:
            best_gain = -1
            best_overlap = 0
            best_idx = -1
            candidates = []
            
            # Iterate all candidates
            for i in range(total_space):
                if used[i]:
                    continue
                gain = (cover_masks[i] & uncovered).bit_count()
                if gain == 0:
                    continue
                overlap = (cover_masks[i] & covered_union).bit_count()
                score = gain - alpha * overlap
                
                # We want max score
                # For randomization, we might collect top candidates
                if score > best_gain:
                    best_gain = score
                    best_overlap = overlap
                    best_idx = i
                    candidates = [i]
                elif score == best_gain:
                    candidates.append(i)
            
            if not candidates:
                break
                
            if randomize_ties and len(candidates) > 1:
                best_idx = random.choice(candidates)
            
            code.append(best_idx)
            used[best_idx] = True
            uncovered &= ~cover_masks[best_idx]
            covered_union |= cover_masks[best_idx]
            
        return code

    def local_search_reduce(code):
        if not code:
            return code
        
        current_code = code[:]
        # Set for fast lookup
        code_set = set(current_code)
        
        # Build cover_count array
        cover_count = np.zeros(total_space, dtype=np.int32)
        for idx in current_code:
            m = cover_masks[idx]
            while m:
                low = m & (-m)
                w_idx = low.bit_length() - 1
                cover_count[w_idx] += 1
                m ^= low
        
        improved = True
        while improved:
            improved = False
            random.shuffle(current_code)
            n_cw = len(current_code)
            
            # Pass 1: Removal
            for i in range(n_cw):
                if time.time() > deadline:
                    return current_code
                
                idx = current_code[i]
                mask = cover_masks[idx]
                
                # Check if all words covered by idx are covered by at least one other codeword
                removable = True
                m = mask
                while m:
                    low = m & (-m)
                    w_idx = low.bit_length() - 1
                    if cover_count[w_idx] <= 1:
                        removable = False
                        break
                    m ^= low
                
                if removable:
                    # Decrement cover_count and remove
                    m = mask
                    while m:
                        low = m & (-m)
                        w_idx = low.bit_length() - 1
                        cover_count[w_idx] -= 1
                        m ^= low
                    current_code.pop(i)
                    code_set.remove(idx)
                    improved = True
                    break # Restart loop after modification
            
            if improved:
                continue
            
            # Pass 2: Swap to enable removal
            # Try to swap a codeword that has "unique" coverage with one that doesn't,
            # such that the new codeword covers the unique parts.
            
            # To be efficient, we only try swaps if we have time.
            if time.time() > deadline - 10.0:
                break
                
            found_swap = False
            # Shuffle to avoid bias
            for i in range(n_cw):
                if time.time() > deadline:
                    return current_code
                    
                idx = current_code[i]
                mask = cover_masks[idx]
                
                # Find words uniquely covered by idx
                unique_mask = 0
                m = mask
                while m:
                    low = m & (-m)
                    w_idx = low.bit_length() - 1
                    if cover_count[w_idx] == 1:
                        unique_mask |= low
                    m ^= low
                
                if unique_mask == 0:
                    # Should have been removed in Pass 1, but maybe counts changed?
                    # Or it's redundant but we didn't catch it? 
                    # Actually, if unique_mask is 0, it is removable.
                    # But Pass 1 should have caught it. 
                    # If we are here, it means we didn't find a removal in Pass 1.
                    # This implies unique_mask != 0 for all i?
                    # Wait, if unique_mask == 0, it IS removable.
                    # Why didn't Pass 1 remove it?
                    # Pass 1 checks `cover_count[w_idx] <= 1`.
                    # If unique_mask is 0, all `cover_count` in mask are > 1.
                    # So Pass 1 should have removed it.
                    # If we are in Pass 2, it means Pass 1 found NO removable codewords.
                    # Therefore, for all i, unique_mask != 0.
                    continue
                
                # Find a replacement j not in code_set such that cover_masks[j] covers unique_mask
                # i.e., (cover_masks[j] & unique_mask) == unique_mask
                replacement = -1
                for j in range(total_space):
                    if j in code_set:
                        continue
                    if (cover_masks[j] & unique_mask) == unique_mask:
                        replacement = j
                        break
                
                if replacement != -1:
                    # Perform swap
                    # 1. Remove idx
                    m = mask
                    while m:
                        low = m & (-m)
                        w_idx = low.bit_length() - 1
                        cover_count[w_idx] -= 1
                        m ^= low
                    code_set.remove(idx)
                    
                    # 2. Add replacement
                    m = cover_masks[replacement]
                    while m:
                        low = m & (-m)
                        w_idx = low.bit_length() - 1
                        cover_count[w_idx] += 1
                        m ^= low
                    code_set.add(replacement)
                    
                    current_code[i] = replacement
                    improved = True
                    found_swap = True
                    break # Restart loop after swap
            
            if not found_swap:
                break
                
        return current_code

    # Run initial greedy with different alphas and random restarts
    best_code = greedy_cover(alpha=0.0)
    best_code = local_search_reduce(best_code)
    best_size = len(best_code)
    
    # Randomized restarts
    # Vary alpha and use random tie-breaking
    alphas = [0.0, 0.1, 0.2, 0.5, 1.0]
    
    # Estimate time per restart
    # Greedy: ~0.02s
    # LS: ~0.05s
    # Total ~0.07s. 
    # 115s / 0.07s ~ 1600 restarts.
    # Let's run as many as possible.
    
    restart_count = 0
    while time.time() < deadline - 5.0:
        alpha = random.choice(alphas)
        # Randomize tie-breaking
        rand_ties = random.random() < 0.5
        
        cand_code = greedy_cover(alpha=alpha, randomize_ties=rand_ties)
        cand_code = local_search_reduce(cand_code)
        
        if len(cand_code) < best_size:
            best_size = len(cand_code)
            best_code = cand_code
            # If we hit the theoretical lower bound or known best, we can stop early?
            # Known best is 186. Lower bound 156.
            # If we hit 186, it's good. If we go lower, great.
            if best_size <= 186:
                # Continue searching for a while to see if we can go lower
                pass
        
        restart_count += 1
        
    return [all_words[i] for i in best_code]

# EVOLVE-BLOCK-END
