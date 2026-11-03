import time
import random
import numpy as np

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 115.0
    
    total_space = q ** n
    
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

    all_words = [idx_to_word(i) for i in range(total_space)]
    
    # Build cover matrix using NumPy for fast vectorized operations
    cover_matrix = np.zeros((total_space, total_space), dtype=bool)
    np.fill_diagonal(cover_matrix, True)
    
    if radius >= 1:
        # Precompute powers for index arithmetic
        powers = [q ** (n - 1 - pos) for pos in range(n)]
        for i in range(total_space):
            w = all_words[i]
            for pos in range(n):
                orig_val = w[pos]
                for val in range(q):
                    if val == orig_val:
                        continue
                    j = i + (val - orig_val) * powers[pos]
                    cover_matrix[i, j] = True

    # Convert to int8 for fast dot products
    cm_int = cover_matrix.astype(np.int8)
    cm_float = cover_matrix.astype(np.float32)

    # Precompute reverse cover lists (who covers each word)
    reverse_cover = [[] for _ in range(total_space)]
    for w_idx in range(total_space):
        # Words that cover w_idx are those i where cover_matrix[i, w_idx] is True
        # This is column w_idx of cover_matrix
        cols = np.where(cover_matrix[:, w_idx])[0]
        reverse_cover[w_idx] = cols.tolist()

    def greedy_cover_randomized():
        uncovered = np.ones(total_space, dtype=bool)
        code = []
        
        while np.any(uncovered):
            unc_int = uncovered.astype(np.int8)
            gains = cm_int.dot(unc_int)
            
            max_gain = np.max(gains)
            if max_gain == 0:
                break
                
            candidates = np.where(gains == max_gain)[0]
            
            best_idx = candidates[random.randint(0, len(candidates)-1)]
            
            code.append(int(best_idx))
            uncovered[cover_matrix[best_idx]] = False
            
        return code

    def local_search_reduce(code):
        if not code:
            return code
        
        current_code = code[:]
        current_code_set = set(current_code)
        
        # Use int32 array for cover counts
        cover_count = np.zeros(total_space, dtype=np.int32)
        for idx in current_code:
            cover_count += cover_matrix[idx].astype(np.int32)
        
        improved = True
        while improved and time.time() < deadline:
            improved = False
            random.shuffle(current_code)
            n_cw = len(current_code)
            
            for i in range(n_cw):
                if time.time() > deadline:
                    return current_code
                
                idx = current_code[i]
                mask = cover_matrix[idx]
                
                # Check if removable: all covered words have count > 1
                counts_in_mask = cover_count[mask]
                if len(counts_in_mask) > 0 and np.min(counts_in_mask) > 1:
                    # Removable
                    cover_count[mask] -= 1
                    current_code.pop(i)
                    current_code_set.remove(idx)
                    improved = True
                    break
            
            if not improved:
                # Try to replace codewords to reduce size or improve structure
                # Find codewords with "unique" coverage (words covered only by them)
                # and see if we can replace them to reduce total unique words or size
                
                max_unique_to_cover = 3
                
                for i in range(n_cw):
                    if time.time() > deadline:
                        return current_code
                    
                    idx = current_code[i]
                    mask = cover_matrix[idx]
                    
                    # Find words uniquely covered by this codeword
                    unique_mask = mask & (cover_count == 1)
                    
                    if not np.any(unique_mask):
                        # Can remove
                        cover_count[mask] -= 1
                        current_code.pop(i)
                        current_code_set.remove(idx)
                        improved = True
                        break
                    
                    unique_words = np.where(unique_mask)[0]
                    
                    if len(unique_words) > max_unique_to_cover:
                        continue
                    
                    # Try to find a replacement codeword that covers these unique words
                    # and doesn't introduce too many new unique words
                    # Intersection of reverse covers of unique words
                    candidates = set(reverse_cover[unique_words[0]])
                    
                    for w in unique_words[1:]:
                        if not candidates:
                            break
                        candidates.intersection_update(set(reverse_cover[w]))
                    
                    candidates.discard(idx)
                    candidates -= current_code_set
                    
                    if candidates:
                        # Score candidates by how many new words they cover (that are currently uncovered by others)
                        # Actually, we want to minimize the "cost" of replacement
                        # Cost = words that become uniquely covered by the new codeword (that weren't before)
                        # Benefit = words that are no longer uniquely covered by the old codeword
                        
                        best_candidate = None
                        best_score = float('inf')
                        cand_list = list(candidates)
                        random.shuffle(cand_list)
                        
                        for cand in cand_list:
                            new_mask = cover_matrix[cand]
                            old_mask = cover_matrix[idx]
                            
                            # Words covered by new but not old
                            new_only = new_mask & ~old_mask
                            # Words covered by old but not new
                            old_only = old_mask & ~new_mask
                            
                            # After replacement:
                            # cover_count changes: -1 for old_only, +1 for new_only
                            # A word in new_only becomes uniquely covered if its current count is 1 (it's not in old_mask, so count is unchanged? No.)
                            # Wait. If word is in new_only, it's not in old_mask. So its count is currently C. After replacement, it becomes C+1.
                            # It becomes uniquely covered if C+1 == 1, i.e., C == 0. But if C==0, it was uncovered.
                            # We assume the code is valid, so all words have count >= 1.
                            # So new_only words have count >= 1. After +1, count >= 2. So they are NOT uniquely covered.
                            # Words in old_only have count C. After -1, count C-1.
                            # They become uniquely covered if C-1 == 1, i.e., C == 2.
                            # They become uncovered if C-1 == 0, i.e., C == 1. (Invalid!)
                            
                            # So we must ensure no word in old_only has count 1.
                            # And we want to minimize the number of words in old_only with count 2 (which become unique).
                            # Also, words in intersection (old & new) have count C. After -1+1, count C. No change in uniqueness.
                            
                            # Let's check validity first
                            old_only_counts = cover_count[old_only]
                            if len(old_only_counts) > 0 and np.min(old_only_counts) <= 1:
                                continue # Would leave some word with count 0 or 1 (if it was 2, it becomes 1, which is unique, but valid. If it was 1, it becomes 0, invalid)
                            
                            # Wait, if count was 2, it becomes 1. That's valid, but it becomes uniquely covered by the new codeword?
                            # No, it's in old_only, so it's NOT covered by the new codeword.
                            # So if count was 2, and we remove the old codeword, count becomes 1.
                            # Who covers it now? Only the other codeword that covered it.
                            # So it is NOT uniquely covered by the new codeword.
                            # It is uniquely covered by the OTHER codeword.
                            # Does that matter? We are trying to reduce the number of codewords.
                            # The "uniqueness" metric is a heuristic for which codewords are "hard" to remove.
                            
                            # Let's define score as: number of words that become uniquely covered by ANY codeword after replacement?
                            # That's complex.
                            
                            # Simpler heuristic:
                            # We want to replace idx with cand if:
                            # 1. All words in old_only have count > 1 (so they don't become uncovered).
                            # 2. The number of words in new_only with count == 1 is small? No, they become count 2.
                            # 3. Actually, the goal is to enable future removals.
                            #    A codeword is removable if it has NO unique words.
                            #    By replacing idx with cand, we change the unique status of words in old_only and new_only.
                            #    Words in old_only: count C -> C-1.
                            #       If C-1 == 1, they become uniquely covered by the remaining codeword.
                            #    Words in new_only: count C -> C+1.
                            #       If C+1 == 1, impossible since C>=1.
                            
                            # So the "cost" is the number of words in old_only with count == 2.
                            # These words will now be uniquely covered by some other codeword.
                            # This might make that other codeword harder to remove.
                            
                            # Let's use this score.
                            cost = np.sum(cover_count[old_only] == 2)
                            
                            if cost < best_score:
                                best_score = cost
                                best_candidate = cand
                                
                        if best_candidate is not None:
                            # Perform replacement
                            old_mask = cover_matrix[idx]
                            new_mask = cover_matrix[best_candidate]
                            
                            cover_count[old_mask] -= 1
                            cover_count[new_mask] += 1
                            
                            current_code[i] = best_candidate
                            current_code_set.remove(idx)
                            current_code_set.add(best_candidate)
                            
                            improved = True
                            break
                            
        return current_code

    def fix_coverage(code, uncovered_mask):
        new_code = code[:]
        code_set = set(new_code)
        while np.any(uncovered_mask):
            unc_int = uncovered_mask.astype(np.int8)
            gains = cm_int.dot(unc_int)
            # Exclude existing codewords
            for i in code_set:
                gains[i] = 0
                
            max_gain = np.max(gains)
            if max_gain == 0:
                break
                
            best_idx = np.argmax(gains)
            
            new_code.append(int(best_idx))
            code_set.add(best_idx)
            uncovered_mask[cover_matrix[best_idx]] = False
        return new_code

    # Initial construction
    best_code = greedy_cover_randomized()
    best_code = local_search_reduce(best_code)
    
    # Iterated Local Search
    if time.time() < deadline - 15.0:
        for _ in range(100):
            if time.time() > deadline - 5.0:
                break
            
            current_code = best_code[:]
            code_set = set(current_code)
            
            # Compute cover counts
            cover_count = np.zeros(total_space, dtype=np.int32)
            for idx in current_code:
                cover_count += cover_matrix[idx].astype(np.int32)
            
            # Perturb: remove a random codeword
            if current_code:
                rem_idx_pos = random.randint(0, len(current_code) - 1)
                rem_idx = current_code[rem_idx_pos]
                
                current_code.pop(rem_idx_pos)
                code_set.remove(rem_idx)
                
                # Identify newly uncovered words
                uncovered_mask = np.zeros(total_space, dtype=bool)
                mask = cover_matrix[rem_idx]
                cover_count[mask] -= 1
                uncovered_mask[cover_count == 0] = True
                
                # Fix coverage
                if np.any(uncovered_mask):
                    current_code = fix_coverage(current_code, uncovered_mask)
                    code_set = set(current_code)
                
                # Run Local Search
                cand_code = local_search_reduce(current_code)
                
                if len(cand_code) < len(best_code):
                    best_code = cand_code

    return [all_words[i] for i in best_code]
