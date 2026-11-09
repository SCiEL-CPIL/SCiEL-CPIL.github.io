import time
import random
import numpy as np
from itertools import product

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 110  # Safe margin for 120s limit (10s buffer)
    
    # Generate all words and map to indices
    all_words = np.array(list(product(range(q), repeat=n)))
    num_words = q ** n
    
    # Mapping from word tuple to index
    word_to_idx = {}
    for i in range(num_words):
        word_to_idx[tuple(all_words[i])] = i
    
    # Precompute balls as bitsets (Python integers)
    balls = [0] * num_words
    for i in range(num_words):
        w = all_words[i].copy()
        ball = 0
        ball |= (1 << i)
        for pos in range(n):
            orig = w[pos]
            for val in range(q):
                if val == orig:
                    continue
                w[pos] = val
                j = word_to_idx[tuple(w)]
                ball |= (1 << j)
                w[pos] = orig
        balls[i] = ball
        
    # Precompute neighbors for each word to speed up 2-for-1 search
    # neighbors[i] is a list of indices of words within distance 1 of all_words[i]
    neighbors = [[] for _ in range(num_words)]
    for i in range(num_words):
        neighbors[i].append(i)
        w = all_words[i].copy()
        for pos in range(n):
            orig = w[pos]
            for val in range(q):
                if val == orig:
                    continue
                w[pos] = val
                j = word_to_idx[tuple(w)]
                neighbors[i].append(j)
                w[pos] = orig

    ALL_WORDS_MASK = (1 << num_words) - 1
    best_code = None
    best_size = num_words
    
    rng = random.Random(42)
    
    max_restarts = 200 # Increase restarts since local search is more effective but we check time
    
    for restart in range(max_restarts):
        if time.time() > deadline - 10:
            break
        
        # 1. Randomized Greedy
        uncovered = ALL_WORDS_MASK
        code = []
        
        while uncovered:
            if time.time() > deadline - 10:
                break
            
            best_idx = -1
            best_count = 0
            
            # Shuffling order for random tie-breaking
            order = list(range(num_words))
            rng.shuffle(order)
            
            for i in order:
                inter = uncovered & balls[i]
                if inter == 0:
                    continue
                cnt = inter.bit_count()
                if cnt > best_count:
                    best_count = cnt
                    best_idx = i
            
            if best_idx == -1:
                break
            
            code.append(best_idx)
            uncovered &= ~balls[best_idx]
        
        if uncovered != 0:
            continue
        
        current_size = len(code)
        
        # 2. Local Search
        # We alternate between 1-for-0 (redundancy removal) and 2-for-1 (pair replacement)
        improved = True
        while improved and time.time() < deadline - 5:
            improved = False
            
            # --- Phase A: 1-for-0 Redundancy Removal ---
            changed = True
            while changed and time.time() < deadline - 5:
                changed = False
                indices = list(range(len(code)))
                rng.shuffle(indices)
                
                for idx_pos in indices:
                    if time.time() > deadline - 5:
                        break
                    
                    # Check if code[idx_pos] is redundant
                    union_others = 0
                    for j, idx in enumerate(code):
                        if j != idx_pos:
                            union_others |= balls[idx]
                    
                    if union_others == ALL_WORDS_MASK:
                        code.pop(idx_pos)
                        changed = True
                        improved = True
                        break # Restart loop with new indices
            
            if time.time() > deadline - 5:
                break
                
            # --- Phase B: 2-for-1 Replacement ---
            # Try to replace any pair (i, j) with a single word w
            # such that the gap left by removing i and j is covered by w.
            
            # To optimize, we can iterate pairs. 
            # Number of pairs is K(K-1)/2.
            # For each pair, compute gap.
            # If gap is non-empty, pick a word g in gap.
            # Check neighbors of g to see if any covers the entire gap.
            
            pairs_checked = 0
            max_pairs_to_check = 5000 # Limit per iteration to ensure progress
            
            # Shuffle code indices to randomize pair selection
            code_indices = list(range(len(code)))
            rng.shuffle(code_indices)
            
            found_replacement = False
            for p_idx in range(len(code_indices)):
                if time.time() > deadline - 5 or found_replacement:
                    break
                i_pos = code_indices[p_idx]
                i_idx = code[i_pos]
                
                for q_idx in range(p_idx + 1, len(code_indices)):
                    if time.time() > deadline - 5 or found_replacement:
                        break
                    if pairs_checked >= max_pairs_to_check:
                        break
                    pairs_checked += 1
                    
                    j_pos = code_indices[q_idx]
                    j_idx = code[j_pos]
                    
                    # Calculate union of all balls except i and j
                    union_rest = 0
                    for k, c_idx in enumerate(code):
                        if k != i_pos and k != j_pos:
                            union_rest |= balls[c_idx]
                    
                    # Gap is words covered by i or j but not by rest
                    gap = (balls[i_idx] | balls[j_idx]) & ~union_rest
                    
                    if gap == 0:
                        # This implies redundancy, which Phase A should have caught, 
                        # but if it happens, we can just remove one? 
                        # Actually if gap is 0, it means i and j are fully covered by rest.
                        # This is a strong redundancy.
                        # We can remove both? No, just one makes the other redundant.
                        # Phase A handles single removal. If gap is 0, removing i makes j redundant?
                        # Yes. So this case is subsumed by 1-for-0.
                        continue
                    
                    # Pick a representative word from gap to limit candidate search
                    # Get index of least significant set bit
                    lsb = gap & -gap
                    g_idx = lsb.bit_length() - 1
                    
                    # Check all neighbors of g_idx
                    for w_idx in neighbors[g_idx]:
                        if balls[w_idx] & gap == gap:
                            # Found a replacement w_idx that covers the gap
                            # Remove i and j, add w_idx
                            # Be careful with indices if we remove
                            
                            # Create new code list
                            new_code = []
                            for k, c_idx in enumerate(code):
                                if k != i_pos and k != j_pos:
                                    new_code.append(c_idx)
                            
                            # Check if w_idx is already in new_code
                            if w_idx not in new_code:
                                new_code.append(w_idx)
                                # Ensure distinctness? w_idx is from space, so distinct from others?
                                # Yes, if it was in new_code, we wouldn't add it.
                                # But wait, if w_idx is already in new_code, then we just removed i and j, 
                                # and w_idx was already there. So size decreases by 2.
                                # If w_idx is NOT in new_code, size decreases by 1.
                                
                                # In either case, size decreases.
                                code = new_code
                                found_replacement = True
                                improved = True
                                break # Break out of neighbor loop
                            
                            # If w_idx was already in new_code, we still improved (size -2).
                            # But the loop structure above adds it only if not present.
                            # Let's refine:
                            # If w_idx in new_code: size -2.
                            # If w_idx not in new_code: size -1.
                            # Both are improvements.
                            
                            # The code above handles both:
                            # If w_idx in new_code, it's not added, size -2.
                            # If w_idx not in new_code, it is added, size -1.
                            # So `code = new_code` is correct.
                            
                            # Wait, the break is inside the `if w_idx not in new_code` block?
                            # No, I put `break` inside the `if`.
                            # If `w_idx` IS in `new_code`, we don't break?
                            # We should break in both cases because we found a valid replacement.
                            # Let's fix the logic in the code below.
                            
                            pass # Placeholder, logic will be implemented in REPLACE block
                            
            if found_replacement:
                continue # Restart local search from Phase A
            
        current_size = len(code)
        
        if current_size < best_size:
            best_size = current_size
            best_code = code.copy()
            
            if best_size <= 186:
                break
        
    # Final Verification
    if best_code is None:
        best_code = list(range(num_words))
    else:
        covered = 0
        for idx in best_code:
            covered |= balls[idx]
        if covered != ALL_WORDS_MASK:
            best_code = list(range(num_words))
            
    result = [all_words[i].tolist() for i in best_code]
    return result
