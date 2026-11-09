import time
import random
import itertools
import numpy as np

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 115  # Safety margin for 120s limit
    
    # Specific parameters for this task: q=3, n=8, radius=1
    if not (q == 3 and n == 8 and radius == 1):
        # Fallback for other parameters
        # For q=3, n >= 4, use tetracode construction
        if q == 3 and n >= 4:
            tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
            return [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]
        
        # Generic small case fallback
        centers = []
        covered = set()
        all_words = set(itertools.product(range(q), repeat=n))
        candidates = list(all_words)
        random.shuffle(candidates)
        
        while covered != all_words and time.time() < deadline:
            best_center = None
            max_new = 0
            for c in candidates:
                neigh = [c]
                for i in range(n):
                    for val in range(q):
                        if val == c[i]: continue
                        new_c = list(c)
                        new_c[i] = val
                        neigh.append(tuple(new_c))
                new_covered = [w for w in neigh if w not in covered]
                if len(new_covered) > max_new:
                    max_new = len(new_covered)
                    best_center = c
                    best_neigh = neigh
            if best_center is None or max_new == 0:
                break
            centers.append(list(best_center))
            for w in best_neigh:
                covered.add(w)
            candidates.remove(best_center)
        return centers

    # --- Optimized Path for q=3, n=8, radius=1 ---
    
    total_words = 3 ** 8  # 6561
    
    # 1. Pre-compute mapping and neighbors
    word_to_idx = {}
    idx_to_word = [None] * total_words
    
    # Generate all words and map to indices
    for word in itertools.product(range(3), repeat=8):
        idx = 0
        for digit in word:
            idx = idx * 3 + digit
        word_to_idx[word] = idx
        idx_to_word[idx] = word

    # Pre-compute neighbors for each word (including itself)
    # neighbors[i] is a list of indices of words within distance 1 of idx_to_word[i]
    neighbors = [[] for _ in range(total_words)]
    
    for i in range(total_words):
        center = idx_to_word[i]
        neighbors[i].append(i)
        for pos in range(8):
            old_val = center[pos]
            for new_val in range(3):
                if new_val == old_val:
                    continue
                neighbor = list(center)
                neighbor[pos] = new_val
                n_idx = word_to_idx[tuple(neighbor)]
                neighbors[i].append(n_idx)
    
    # 2. Initial Code: Direct Sum Construction
    # Tetracode: 9 codewords of length 4
    tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
    
    # Direct sum with length 4 all-3s? No, direct sum of tetracode with length 4 code?
    # The standard 729 code is Tetracode(4,9) x {0,1,2}^4
    # So we take every tetracode word and append every possible length-4 tail.
    
    initial_code_indices = []
    for t in tetracode:
        t_idx = 0
        for d in t:
            t_idx = t_idx * 3 + d
        # We need to find the indices of words starting with t
        # Instead of searching, we can just generate them
        for tail in itertools.product(range(3), repeat=4):
            full_word = t + list(tail)
            idx = word_to_idx[tuple(full_word)]
            initial_code_indices.append(idx)
            
    code_set = set(initial_code_indices)
    
    # 3. Local Search: Critical Word Swaps
    # counts[w] = number of centers in code_set covering w
    counts = np.zeros(total_words, dtype=np.int32)
    
    def update_counts(idx, delta):
        for n_idx in neighbors[idx]:
            counts[n_idx] += delta

    for idx in code_set:
        update_counts(idx, 1)
        
    best_len = len(code_set)
    best_code = set(code_set)
    
    # Pre-compute list of all indices for random sampling
    all_indices = list(range(total_words))
    
    # Local search loop
    while time.time() < deadline - 5:
        # Find a critical word: a word covered by exactly 1 center
        # We can sample random words to find critical ones faster than scanning all
        # Or scan if the code is small enough? 6561 is small.
        # But scanning 6561 ints is fast.
        
        # Let's try to find a critical word
        critical_word_idx = -1
        # Sample 1000 random words to find a critical one
        found_critical = False
        for _ in range(1000):
            w = random.choice(all_indices)
            if counts[w] == 1:
                critical_word_idx = w
                found_critical = True
                break
                
        if not found_critical:
            # If no critical word found in sample, maybe code is locally optimal or stuck
            # Try a random swap anyway? Or break?
            # If no critical words, every word is covered by >=2 centers.
            # We can still try to remove a center if it doesn't make any word count 0.
            # Let's try to remove a random center.
            removable = False
            for _ in range(100):
                c = random.choice(list(code_set))
                # Check if removing c makes any neighbor count 0
                can_remove = True
                for n_idx in neighbors[c]:
                    if counts[n_idx] == 1:
                        can_remove = False
                        break
                if can_remove:
                    # Remove it
                    update_counts(c, -1)
                    code_set.remove(c)
                    best_len -= 1
                    if len(code_set) < best_len:
                        best_len = len(code_set)
                        best_code = set(code_set)
                    removable = True
                    break # Restart search with new state
            if not removable:
                # Stuck? Try a random perturbation?
                # Or break to save time?
                # Let's try a random swap: remove one, add one
                # Pick a random center to remove
                c_remove = random.choice(list(code_set))
                # Pick a random word it covers that is critical?
                # Actually, just pick a random new center
                c_add = random.choice(all_indices)
                if c_add in code_set:
                    continue
                    
                # Check validity of swap:
                # Remove c_remove, add c_add
                # Must not leave any word uncovered.
                # Words only covered by c_remove must be covered by c_add.
                # i.e., for all w in N(c_remove) where counts[w]==1, w must be in N(c_add)
                
                valid_swap = True
                for w in neighbors[c_remove]:
                    if counts[w] == 1:
                        # w is only covered by c_remove
                        # c_add must cover w
                        # Check if w is in neighbors[c_add]
                        # neighbors[c_add] is a list. Membership check is O(18).
                        if w not in neighbors[c_add]:
                            valid_swap = False
                            break
                            
                if valid_swap:
                    update_counts(c_remove, -1)
                    code_set.remove(c_remove)
                    update_counts(c_add, 1)
                    code_set.add(c_add)
                    # Size unchanged, but structure changed. Might unlock further reductions.
                    # We don't update best_len here as size didn't change, but we accept the move.
                    # To avoid cycling, we could track visited states, but for now we just proceed.
                    # Actually, if size doesn't change, we should be careful not to oscillate.
                    # Let's only accept if it allows a subsequent reduction?
                    # For simplicity, we accept it and hope the next iteration finds a reduction.
                    # But if we oscillate, we waste time.
                    # Let's only do this if we are stuck on reductions for a while?
                    # For now, let's stick to reductions only to be safe.
                    # Revert
                    update_counts(c_add, -1)
                    code_set.remove(c_add)
                    update_counts(c_remove, 1)
                    code_set.add(c_remove)
                    
                    # If we can't reduce, we might be at a local minimum.
                    # Break to avoid wasting time?
                    # Or try harder?
                    # Let's break if we've been stuck for a while.
                    # Since we don't track "stuck" counter, let's just break if no critical word found in 1000 samples.
                    break
            continue

        # Found a critical word `critical_word_idx`
        # Find the center `c_remove` that covers it
        # We need to identify which center in code_set covers critical_word_idx
        # We can iterate through code_set and check if critical_word_idx is in neighbors[c]
        # But code_set can be large (up to 729).
        # Iterating 729 * 1000 samples * 18 checks is ~13M ops. Might be slow in Python.
        
        # Optimization: Maintain a map from word to list of centers covering it?
        # That's heavy to update.
        
        # Alternative: Since counts[w]==1, there is exactly one center.
        # We can find it by checking neighbors of w? No, neighbors of w are words.
        # We need the center.
        
        # Let's iterate through code_set.
        c_remove = -1
        for c in code_set:
            if critical_word_idx in neighbors[c]:
                c_remove = c
                break
                
        if c_remove == -1:
            # Should not happen if counts are consistent
            continue
            
        # Now, we want to find a replacement center `c_add` such that:
        # 1. c_add is not in code_set (or if it is, it's a different one? No, distinct centers).
        # 2. c_add covers critical_word_idx.
        # 3. For all w in N(c_remove) where counts[w] == 1, c_add must cover w.
        #    (This ensures no word becomes uncovered).
        # 4. Ideally, c_add covers more words or allows future reductions.
        #    Simplest: Just find ANY valid c_add.
        
        # Candidates for c_add:
        # Must be in neighbors[critical_word_idx].
        # There are 18 such candidates.
        
        found_replacement = False
        # Shuffle candidates to randomize
        candidates = list(neighbors[critical_word_idx])
        random.shuffle(candidates)
        
        for c_add in candidates:
            if c_add in code_set:
                continue
                
            # Check validity
            valid = True
            for w in neighbors[c_remove]:
                if counts[w] == 1:
                    if w not in neighbors[c_add]:
                        valid = False
                        break
                        
            if valid:
                # Perform swap
                update_counts(c_remove, -1)
                code_set.remove(c_remove)
                update_counts(c_add, 1)
                code_set.add(c_add)
                
                # Size unchanged. 
                # We accept this swap. It changes the structure.
                # We do NOT update best_len.
                # We continue the loop to try to find a reduction with the new structure.
                found_replacement = True
                break
                
        if not found_replacement:
            # No valid replacement for this critical word.
            # Try another critical word?
            # Or break?
            # If we can't swap any critical word, we are stuck.
            # Let's try to find another critical word in the same iteration?
            # The loop continues, so it will pick a new random word.
            # If we keep failing, we might loop forever.
            # Let's add a counter for failures.
            # For now, we just continue. If we time out, we return best.
            # To prevent infinite loop on same critical word, the random sampling will likely pick a different one.
            # But if only 1 critical word exists and we can't swap it, we are stuck.
            # In that case, we should break.
            # How to know if it's the only one? Hard.
            # Let's just break if we fail to swap a few times in a row?
            # Simple heuristic: if we fail 10 times in a row, break.
            # We don't have a counter here. Let's just break to be safe and fast.
            # Actually, if we break, we might miss a solution.
            # But if we can't swap, we can't reduce size from this state.
            # So breaking is reasonable.
            break

    # Convert best_code to list of lists
    result = [list(idx_to_word[idx]) for idx in best_code]
    
    return result
