import time
import random
import numpy as np

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 118.0  # Safety margin (150s kill, 32s buffer)
    
    total_space = q ** n
    
    # Precompute word to index and index to word
    def idx_to_word(idx):
        w = []
        for i in range(n):
            w.append(idx % q)
            idx //= q
        w.reverse()
        return w

    # Precompute all words
    all_words = [idx_to_word(i) for i in range(total_space)]
    
    # Precompute cover masks as Python integers (bitsets)
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
                    power = q ** (n - 1 - pos)
                    j = i + (val - orig_val) * power
                    mask |= (1 << j)
        cover_masks[i] = mask
        
    full_mask = (1 << total_space) - 1
    
    # Precompute reverse cover: for each word, which codewords cover it
    reverse_cover = [[] for _ in range(total_space)]
    for cw_idx in range(total_space):
        m = cover_masks[cw_idx]
        while m:
            low = m & (-m)
            w_idx = low.bit_length() - 1
            reverse_cover[w_idx].append(cw_idx)
            m ^= low
    
    # Precompute cover sizes
    cover_sizes = [cover_masks[i].bit_count() for i in range(total_space)]
    
    # Greedy Set Cover with random tie-breaking
    def greedy_cover_randomized():
        uncovered = full_mask
        code = []
        while uncovered:
            best_gain = 0
            candidates = []
            for i in range(total_space):
                gain = (cover_masks[i] & uncovered).bit_count()
                if gain > best_gain:
                    best_gain = gain
                    candidates = [i]
                elif gain == best_gain and gain > 0:
                    candidates.append(i)
            if not candidates:
                break
            best_idx = random.choice(candidates)
            code.append(best_idx)
            uncovered &= ~cover_masks[best_idx]
        return code

    # Local Search: Remove redundant codewords and perform swaps
    def local_search_reduce(code):
        if not code:
            return code
        
        current_code = list(code)
        current_code_set = set(current_code)
        
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
        while improved and time.time() < deadline:
            improved = False
            random.shuffle(current_code)
            n_cw = len(current_code)
            
            # Phase 1: Removal - try removing each codeword
            for i in range(n_cw):
                if time.time() > deadline:
                    return current_code
                idx = current_code[i]
                mask = cover_masks[idx]
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
                    m = mask
                    while m:
                        low = m & (-m)
                        w_idx = low.bit_length() - 1
                        cover_count[w_idx] -= 1
                        m ^= low
                    current_code.pop(i)
                    current_code_set.remove(idx)
                    improved = True
                    break
            
            # Phase 2: 1-for-1 swap - replace codeword with better one
            if not improved:
                for i in range(n_cw):
                    if time.time() > deadline:
                        return current_code
                    idx = current_code[i]
                    mask = cover_masks[idx]
                    
                    # Find words uniquely covered by this codeword
                    unique_words = []
                    m = mask
                    while m:
                        low = m & (-m)
                        w_idx = low.bit_length() - 1
                        if cover_count[w_idx] == 1:
                            unique_words.append(w_idx)
                        m ^= low
                    
                    if not unique_words:
                        # Can remove entirely
                        m = mask
                        while m:
                            low = m & (-m)
                            w_idx = low.bit_length() - 1
                            cover_count[w_idx] -= 1
                            m ^= low
                        current_code.pop(i)
                        current_code_set.remove(idx)
                        improved = True
                        break
                    
                    if len(unique_words) > 5:
                        continue
                    
                    # Sort by rarity to find candidates faster
                    unique_words.sort(key=lambda w: len(reverse_cover[w]))
                    
                    # Find candidates that cover all unique words
                    candidates = set(reverse_cover[unique_words[0]])
                    for w in unique_words[1:]:
                        if not candidates:
                            break
                        candidates.intersection_update(reverse_cover[w])
                    
                    candidates.discard(idx)
                    candidates -= current_code_set
                    
                    if candidates:
                        # Score candidates by how much new coverage they provide
                        best_candidate = None
                        best_score = -1
                        cand_list = list(candidates)
                        random.shuffle(cand_list)
                        
                        for cand in cand_list:
                            new_cover = (cover_masks[cand] & ~mask).bit_count()
                            if new_cover > best_score:
                                best_score = new_cover
                                best_candidate = cand
                                if best_score >= cover_sizes[cand] - 2:
                                    break
                        
                        if best_candidate is not None:
                            # Perform swap
                            m = mask
                            while m:
                                low = m & (-m)
                                w_idx = low.bit_length() - 1
                                cover_count[w_idx] -= 1
                                m ^= low
                            
                            m = cover_masks[best_candidate]
                            while m:
                                low = m & (-m)
                                w_idx = low.bit_length() - 1
                                cover_count[w_idx] += 1
                                m ^= low
                            
                            current_code[i] = best_candidate
                            current_code_set.remove(idx)
                            current_code_set.add(best_candidate)
                            improved = True
                            break
            
            # Phase 3: 2-for-1 move - replace 2 codewords with 1
            if not improved and n_cw >= 2:
                for i in range(n_cw - 1):
                    if time.time() > deadline:
                        return current_code
                    for j in range(i + 1, n_cw):
                        if time.time() > deadline:
                            return current_code
                        
                        idx1 = current_code[i]
                        idx2 = current_code[j]
                        mask1 = cover_masks[idx1]
                        mask2 = cover_masks[idx2]
                        
                        # Find words that become uniquely uncovered
                        unique_mask = 0
                        m = mask1 | mask2
                        while m:
                            low = m & (-m)
                            w_idx = low.bit_length() - 1
                            if cover_count[w_idx] == 1:
                                unique_mask |= (1 << w_idx)
                            m ^= low
                        
                        if unique_mask == 0:
                            # Both redundant
                            for idx in [idx1, idx2]:
                                m = cover_masks[idx]
                                while m:
                                    low = m & (-m)
                                    w_idx = low.bit_length() - 1
                                    cover_count[w_idx] -= 1
                                    m ^= low
                            current_code.pop(j)
                            current_code.pop(i)
                            current_code_set.remove(idx1)
                            current_code_set.remove(idx2)
                            improved = True
                            break
                        
                        if unique_mask.bit_count() > 7:
                            continue
                        
                        # Find single codeword covering all unique words
                        words_in_unique = []
                        m = unique_mask
                        while m:
                            low = m & (-m)
                            w_idx = low.bit_length() - 1
                            words_in_unique.append(w_idx)
                            m ^= low
                        
                        words_in_unique.sort(key=lambda w: len(reverse_cover[w]))
                        
                        candidates = set(reverse_cover[words_in_unique[0]])
                        for w in words_in_unique[1:]:
                            if not candidates:
                                break
                            candidates.intersection_update(reverse_cover[w])
                        
                        candidates.discard(idx1)
                        candidates.discard(idx2)
                        candidates -= current_code_set
                        
                        if candidates:
                            best_candidate = None
                            best_score = -1
                            cand_list = list(candidates)
                            random.shuffle(cand_list)
                            
                            for cand in cand_list:
                                new_cover = (cover_masks[cand] & (mask1 | mask2)).bit_count()
                                if new_cover > best_score:
                                    best_score = new_cover
                                    best_candidate = cand
                            
                            if best_candidate is not None:
                                for idx in [idx1, idx2]:
                                    m = cover_masks[idx]
                                    while m:
                                        low = m & (-m)
                                        w_idx = low.bit_length() - 1
                                        cover_count[w_idx] -= 1
                                        m ^= low
                                
                                m = cover_masks[best_candidate]
                                while m:
                                    low = m & (-m)
                                    w_idx = low.bit_length() - 1
                                    cover_count[w_idx] += 1
                                    m ^= low
                                
                                current_code.pop(j)
                                current_code.pop(i)
                                current_code.append(best_candidate)
                                current_code_set.remove(idx1)
                                current_code_set.remove(idx2)
                                current_code_set.add(best_candidate)
                                improved = True
                                break
                    
                    if improved:
                        break
            
            # Phase 4: 2-for-2 swap - replace 2 codewords with 2 different ones
            if not improved and n_cw >= 2:
                for i in range(n_cw - 1):
                    if time.time() > deadline:
                        return current_code
                    for j in range(i + 1, n_cw):
                        if time.time() > deadline:
                            return current_code
                        
                        idx1 = current_code[i]
                        idx2 = current_code[j]
                        mask1 = cover_masks[idx1]
                        mask2 = cover_masks[idx2]
                        
                        # Find words uniquely covered by idx1 or idx2
                        unique_to_1 = mask1 & ~mask2
                        unique_to_2 = mask2 & ~mask1
                        
                        unique_words_1 = []
                        m = unique_to_1
                        while m:
                            low = m & (-m)
                            w_idx = low.bit_length() - 1
                            if cover_count[w_idx] == 1:
                                unique_words_1.append(w_idx)
                            m ^= low
                        
                        unique_words_2 = []
                        m = unique_to_2
                        while m:
                            low = m & (-m)
                            w_idx = low.bit_length() - 1
                            if cover_count[w_idx] == 1:
                                unique_words_2.append(w_idx)
                            m ^= low
                        
                        if len(unique_words_1) > 4 or len(unique_words_2) > 4:
                            continue
                        
                        # Find replacement for idx1
                        if unique_words_1:
                            unique_words_1.sort(key=lambda w: len(reverse_cover[w]))
                            cand1 = set(reverse_cover[unique_words_1[0]])
                            for w in unique_words_1[1:]:
                                if not cand1:
                                    break
                                cand1.intersection_update(reverse_cover[w])
                            cand1.discard(idx1)
                            cand1.discard(idx2)
                            cand1 -= current_code_set
                        else:
                            cand1 = set()
                        
                        # Find replacement for idx2
                        if unique_words_2:
                            unique_words_2.sort(key=lambda w: len(reverse_cover[w]))
                            cand2 = set(reverse_cover[unique_words_2[0]])
                            for w in unique_words_2[1:]:
                                if not cand2:
                                    break
                                cand2.intersection_update(reverse_cover[w])
                            cand2.discard(idx1)
                            cand2.discard(idx2)
                            cand2 -= current_code_set
                        else:
                            cand2 = set()
                        
                        if cand1 and cand2:
                            # Try combinations
                            best_pair = None
                            best_score = -1
                            list1 = list(cand1)
                            list2 = list(cand2)
                            random.shuffle(list1)
                            random.shuffle(list2)
                            
                            for c1 in list1[:20]:
                                for c2 in list2[:20]:
                                    if c1 == c2:
                                        continue
                                    # Check no overlap issues
                                    new_mask = cover_masks[c1] | cover_masks[c2]
                                    old_mask = mask1 | mask2
                                    gain = (new_mask & ~old_mask).bit_count()
                                    if gain > best_score:
                                        best_score = gain
                                        best_pair = (c1, c2)
                            
                            if best_pair is not None:
                                c1, c2 = best_pair
                                # Perform swap
                                for idx in [idx1, idx2]:
                                    m = cover_masks[idx]
                                    while m:
                                        low = m & (-m)
                                        w_idx = low.bit_length() - 1
                                        cover_count[w_idx] -= 1
                                        m ^= low
                                
                                for idx in [c1, c2]:
                                    m = cover_masks[idx]
                                    while m:
                                        low = m & (-m)
                                        w_idx = low.bit_length() - 1
                                        cover_count[w_idx] += 1
                                        m ^= low
                                
                                current_code[i] = c1
                                current_code[j] = c2
                                current_code_set.remove(idx1)
                                current_code_set.remove(idx2)
                                current_code_set.add(c1)
                                current_code_set.add(c2)
                                improved = True
                                break
                    
                    if improved:
                        break
                        
        return current_code
    
    # Initial greedy construction
    best_code = greedy_cover_randomized()
    best_code = local_search_reduce(best_code)
    best_size = len(best_code)
    
    # Run multiple restarts with different random seeds
    restart_count = 0
    while time.time() < deadline - 10.0:
        restart_count += 1
        cand_code = greedy_cover_randomized()
        cand_code = local_search_reduce(cand_code)
        if len(cand_code) < len(best_code):
            best_code = cand_code
            best_size = len(best_code)
        # Occasionally do a deeper local search on the best
        if restart_count % 5 == 0 and time.time() < deadline - 20.0:
            best_code = local_search_reduce(best_code)
    
    return [all_words[i] for i in best_code]
