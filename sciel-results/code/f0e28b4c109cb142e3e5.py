import numpy as np
import itertools
import time
import random

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 115  # 5s buffer for safety
    
    N = q ** n
    
    def word_to_idx(word):
        idx = 0
        for i in range(n):
            idx = idx * q + word[i]
        return idx
        
    def idx_to_word(idx):
        word = [0] * n
        for i in range(n-1, -1, -1):
            word[i] = idx % q
            idx //= q
        return word

    all_words = [idx_to_word(i) for i in range(N)]
    
    # Precompute neighbor masks and lists
    neighbor_masks = [0] * N
    neighbor_lists = [[] for _ in range(N)]
    
    for c_idx in range(N):
        c_word = all_words[c_idx]
        mask = 1 << c_idx
        neighbor_lists[c_idx].append(c_idx)
        
        if radius == 1:
            for i in range(n):
                orig = c_word[i]
                for new_val in range(q):
                    if new_val != orig:
                        new_word = c_word.copy()
                        new_word[i] = new_val
                        w_idx = word_to_idx(new_word)
                        mask |= (1 << w_idx)
                        neighbor_lists[c_idx].append(w_idx)
        else:
            for dist in range(1, radius + 1):
                for positions in itertools.combinations(range(n), dist):
                    for vals in itertools.product(range(q), repeat=dist):
                        if all(vals[j] == c_word[positions[j]] for j in range(dist)):
                            continue
                        new_word = c_word.copy()
                        for j, pos in enumerate(positions):
                            new_word[pos] = vals[j]
                        w_idx = word_to_idx(new_word)
                        mask |= (1 << w_idx)
                        neighbor_lists[c_idx].append(w_idx)
                        
        neighbor_masks[c_idx] = mask
    
    # Precompute: for each word, which codewords cover it (inverse mapping)
    # This helps with criticality-based greedy
    word_to_coders = [[] for _ in range(N)]
    for c_idx in range(N):
        for w in neighbor_lists[c_idx]:
            word_to_coders[w].append(c_idx)
    
    def greedy_cover_criticality():
        """Greedy that prioritizes uncovered words with fewest potential coders (criticality)"""
        uncovered = (1 << N) - 1
        code = []
        code_set = set()
        
        # Track coverage count for each word
        cover_count = np.zeros(N, dtype=np.int32)
        
        while uncovered:
            if time.time() > deadline:
                break
            
            # Find the uncovered word with minimum criticality (fewest potential coders)
            # Criticality = number of codewords that could cover this word
            # We want to cover words that are hardest to cover first
            
            min_crit = N + 1
            target_word = -1
            
            # Sample some uncovered words to find one with low criticality
            # For efficiency, we can check a random sample or all uncovered words
            # Since N=6561, checking all is feasible
            
            # Get list of uncovered word indices
            uncovered_words = []
            temp = uncovered
            while temp:
                lsb = temp & (-temp)
                w_idx = lsb.bit_length() - 1
                uncovered_words.append(w_idx)
                temp ^= lsb
            
            if not uncovered_words:
                break
            
            # Find word with minimum criticality among uncovered
            for w_idx in uncovered_words:
                crit = len(word_to_coders[w_idx])
                if crit < min_crit:
                    min_crit = crit
                    target_word = w_idx
                if min_crit == 1:  # Can't get better
                    break
            
            if target_word == -1:
                break
            
            # Find the best codeword that covers target_word and maximizes new coverage
            best_c = -1
            best_new_count = -1
            
            for c_idx in word_to_coders[target_word]:
                if c_idx in code_set:
                    continue
                
                # Count how many new words this codeword would cover
                new_count = (uncovered & neighbor_masks[c_idx]).bit_count()
                
                if new_count > best_new_count:
                    best_new_count = new_count
                    best_c = c_idx
            
            if best_c == -1 or best_new_count == 0:
                # Fallback: pick any codeword that covers target_word
                for c_idx in word_to_coders[target_word]:
                    if c_idx not in code_set:
                        best_c = c_idx
                        break
            
            if best_c == -1:
                break
            
            code.append(best_c)
            code_set.add(best_c)
            uncovered &= ~neighbor_masks[best_c]
            
        return code
    
    def local_search_remove(code):
        """Remove redundant codewords"""
        if not code:
            return code
            
        cover_counts = np.zeros(N, dtype=np.int32)
        for c in code:
            for w in neighbor_lists[c]:
                cover_counts[w] += 1
                
        improved = True
        while improved and time.time() < deadline:
            improved = False
            for i in range(len(code)):
                c = code[i]
                # Check if any word is uniquely covered by c
                has_unique = False
                for w in neighbor_lists[c]:
                    if cover_counts[w] == 1:
                        has_unique = True
                        break
                        
                if not has_unique:
                    code.pop(i)
                    for w in neighbor_lists[c]:
                        cover_counts[w] -= 1
                    improved = True
                    break
                    
        return code
    
    def two_opt_local_search(code):
        """2-opt: try to replace one codeword with another to enable removal of another"""
        code = list(code)
        code_set = set(code)
        
        cover_counts = np.zeros(N, dtype=np.int32)
        for c in code:
            for w in neighbor_lists[c]:
                cover_counts[w] += 1
        
        improved = True
        while improved and time.time() < deadline:
            improved = False
            
            # Try removing each codeword
            for i in range(len(code)):
                if time.time() > deadline:
                    break
                
                c_remove = code[i]
                
                # Find words that would become uncovered if we remove c_remove
                newly_uncovered = []
                for w in neighbor_lists[c_remove]:
                    if cover_counts[w] == 1:
                        newly_uncovered.append(w)
                
                if not newly_uncovered:
                    # Can remove without adding
                    code.pop(i)
                    for w in neighbor_lists[c_remove]:
                        cover_counts[w] -= 1
                    code_set.discard(c_remove)
                    improved = True
                    break
                
                # Try to find a single codeword that covers all newly_uncovered
                # and is not already in code
                best_add = -1
                best_gain = -1
                
                # For each newly uncovered word, find codewords that cover it
                # We need a codeword that covers ALL newly_uncovered words
                candidate_codewords = set()
                for w in newly_uncovered:
                    for c in word_to_coders[w]:
                        if c not in code_set:
                            candidate_codewords.add(c)
                
                # Check each candidate
                for c_add in candidate_codewords:
                    # Check if c_add covers all newly_uncovered
                    covers_all = True
                    for w in newly_uncovered:
                        if (neighbor_masks[c_add] >> w) & 1 == 0:
                            covers_all = False
                            break
                    
                    if not covers_all:
                        continue
                    
                    # Count how many new words c_add would cover (beyond newly_uncovered)
                    new_coverage = 0
                    for w in neighbor_lists[c_add]:
                        if cover_counts[w] == 0:
                            new_coverage += 1
                    
                    if new_coverage > best_gain:
                        best_gain = new_coverage
                        best_add = c_add
                
                if best_add != -1:
                    # Remove c_remove and add best_add
                    code.pop(i)
                    for w in neighbor_lists[c_remove]:
                        cover_counts[w] -= 1
                    code_set.discard(c_remove)
                    
                    code.append(best_add)
                    for w in neighbor_lists[best_add]:
                        cover_counts[w] += 1
                    code_set.add(best_add)
                    
                    improved = True
                    break
        
        return code
    
    def three_opt_local_search(code):
        """3-opt: try to remove two codewords and add one to improve coverage"""
        code = list(code)
        code_set = set(code)
        
        cover_counts = np.zeros(N, dtype=np.int32)
        for c in code:
            for w in neighbor_lists[c]:
                cover_counts[w] += 1
        
        improved = True
        while improved and time.time() < deadline:
            improved = False
            
            # Try removing pairs of codewords
            for i in range(len(code)):
                if time.time() > deadline:
                    break
                
                c1 = code[i]
                
                for j in range(i + 1, len(code)):
                    if time.time() > deadline:
                        break
                    
                    c2 = code[j]
                    
                    # Find words that would become uncovered if we remove both c1 and c2
                    newly_uncovered = []
                    for w in neighbor_lists[c1]:
                        if cover_counts[w] == 1:
                            newly_uncovered.append(w)
                        elif cover_counts[w] == 2:
                            # Check if c2 also covers w
                            if (neighbor_masks[c2] >> w) & 1:
                                newly_uncovered.append(w)
                    
                    # Also check words covered only by c2
                    for w in neighbor_lists[c2]:
                        if cover_counts[w] == 1:
                            if w not in newly_uncovered:
                                newly_uncovered.append(w)
                        elif cover_counts[w] == 2:
                            # Check if c1 also covers w
                            if (neighbor_masks[c1] >> w) & 1 and w not in newly_uncovered:
                                newly_uncovered.append(w)
                    
                    if not newly_uncovered:
                        # Can remove both without adding
                        code.pop(j)
                        code.pop(i)
                        for w in neighbor_lists[c1]:
                            cover_counts[w] -= 1
                        for w in neighbor_lists[c2]:
                            cover_counts[w] -= 1
                        code_set.discard(c1)
                        code_set.discard(c2)
                        improved = True
                        break
                    
                    # Try to find a single codeword that covers all newly_uncovered
                    best_add = -1
                    best_gain = -1
                    
                    candidate_codewords = set()
                    for w in newly_uncovered:
                        for c in word_to_coders[w]:
                            if c not in code_set and c != c1 and c != c2:
                                candidate_codewords.add(c)
                    
                    for c_add in candidate_codewords:
                        covers_all = True
                        for w in newly_uncovered:
                            if (neighbor_masks[c_add] >> w) & 1 == 0:
                                covers_all = False
                                break
                        
                        if not covers_all:
                            continue
                        
                        # Net gain: we remove 2 codewords, add 1, so net -1 if it works
                        # But we need to ensure coverage is maintained
                        best_add = c_add
                        break  # Take first valid one
                    
                    if best_add != -1:
                        # Remove c1 and c2, add best_add
                        code.pop(j)
                        code.pop(i)
                        for w in neighbor_lists[c1]:
                            cover_counts[w] -= 1
                        for w in neighbor_lists[c2]:
                            cover_counts[w] -= 1
                        code_set.discard(c1)
                        code_set.discard(c2)
                        
                        code.append(best_add)
                        for w in neighbor_lists[best_add]:
                            cover_counts[w] += 1
                        code_set.add(best_add)
                        
                        improved = True
                        break
                
                if improved:
                    break
        
        return code
    
    # Run criticality-based greedy
    best_code = greedy_cover_criticality()
    best_code = local_search_remove(best_code)
    best_code = two_opt_local_search(best_code)
    best_code = three_opt_local_search(best_code)
    
    # Try multiple random restarts if time permits
    best_size = len(best_code)
    
    for restart in range(1, 20):
        if time.time() > deadline - 10:
            break
        
        # Randomized greedy: shuffle the order of considering codewords
        random.seed(restart * 42)
        code = greedy_cover_criticality()
        code = local_search_remove(code)
        code = two_opt_local_search(code)
        code = three_opt_local_search(code)
        
        if len(code) < best_size:
            best_code = code
            best_size = len(code)
    
    # Final verification and fix if needed
    covered_mask = 0
    for c in best_code:
        covered_mask |= neighbor_masks[c]
        
    if covered_mask != (1 << N) - 1:
        uncovered = ((1 << N) - 1) & ~covered_mask
        while uncovered and time.time() < deadline:
            best_c = None
            best_count = -1
            for c in range(N):
                if c in best_code:
                    continue
                count = (uncovered & neighbor_masks[c]).bit_count()
                if count > best_count:
                    best_count = count
                    best_c = c
            if best_c is None:
                break
            best_code.append(best_c)
            covered_mask |= neighbor_masks[best_c]
            uncovered &= ~neighbor_masks[best_c]
    
    result = [all_words[i] for i in best_code]
    return result
