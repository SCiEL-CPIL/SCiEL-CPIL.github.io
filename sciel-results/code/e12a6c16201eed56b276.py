import time
import numpy as np

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 110.0
    
    total_words = q ** n
    
    # 1. Precompute neighbors (balls) and coverage matrix
    def int_to_tuple(x, n, q):
        t = [0] * n
        for i in range(n):
            t[i] = x % q
            x //= q
        return t

    def tuple_to_int(t, q):
        x = 0
        for i in range(n-1, -1, -1):
            x = x * q + t[i]
        return x

    # Precompute balls
    balls = [None] * total_words
    for i in range(total_words):
        t_i = int_to_tuple(i, n, q)
        ball = [i]
        if radius >= 1:
            for pos in range(n):
                orig_val = t_i[pos]
                for new_val in range(q):
                    if new_val == orig_val:
                        continue
                    t_new = t_i[:]
                    t_new[pos] = new_val
                    ball.append(tuple_to_int(t_new, q))
        balls[i] = ball
        
    # Coverage matrix M: M[i, j] = 1 if j is in balls[i]
    M = np.zeros((total_words, total_words), dtype=np.uint8)
    for i in range(total_words):
        M[i, balls[i]] = 1

    # 2. Initial Solution: Greedy Set Cover
    covered = np.zeros(total_words, dtype=bool)
    codewords = []
    
    while True:
        if time.time() > deadline - 30:
            break
        uncovered_mask = ~covered
        if not np.any(uncovered_mask):
            break
        scores = M.dot(uncovered_mask.astype(np.uint8))
        best_idx = np.argmax(scores)
        if scores[best_idx] == 0:
            break
        codewords.append(best_idx)
        covered[balls[best_idx]] = True
        
    # Ensure validity
    def is_valid(codewords):
        if len(codewords) == 0:
            return total_words == 0
        coverage_counts = M[codewords, :].sum(axis=0)
        return np.all(coverage_counts >= 1)

    if not is_valid(codewords):
        while not is_valid(codewords) and time.time() < deadline - 10:
             uncovered = np.where(M[codewords, :].sum(axis=0) == 0)[0]
             if len(uncovered) == 0: break
             uncovered_mask = ~covered
             scores = M.dot(uncovered_mask.astype(np.uint8))
             best_idx = np.argmax(scores)
             if scores[best_idx] > 0:
                 codewords.append(best_idx)
                 covered[balls[best_idx]] = True
             else:
                 break

    best_codewords = codewords[:]
    best_size = len(best_codewords)
    
    # 3. Exhaustive Pairwise Exchange Local Search
    # Strategy: 
    # 1. Try to remove any redundant codeword (size reduction by 1)
    # 2. Try to replace any codeword with another to enable removal of another (size reduction by 1, but via swap)
    #    Actually, simpler: 
    #    A. Check if any codeword is redundant (can be removed). If yes, remove and restart.
    #    B. If no redundant codewords, try to swap one codeword c_out with a new codeword c_in 
    #       such that the new set is valid and has the same size (no immediate gain), 
    #       BUT this might allow a subsequent removal? 
    #       No, standard local search for set cover minimization usually focuses on:
    #       1. Removal: If c is redundant, remove it.
    #       2. Exchange: If removing c breaks coverage, find c' that covers the broken parts 
    #          and allows removal of some other c''? 
    #       Actually, a direct exchange that reduces size is impossible (swap keeps size same).
    #       So we rely on:
    #       1. Removals until local minimum.
    #       2. Perturbation: Randomly swap a codeword to escape local minimum.
    #       3. Removals again.
        
    # We will implement a loop that alternates between:
    # - Greedy Removal (find all redundant and remove them)
    # - Perturbation (swap one codeword to a random other, then repair if needed)
    # - Repeat until time limit.

    def greedy_removal(cw):
        changed = True
        while changed and time.time() < deadline - 2:
            changed = False
            if not cw: break
            cover_count = M[cw, :].sum(axis=0)
            # Check each codeword
            for i in range(len(cw)):
                c = cw[i]
                # If min cover count in balls[c] is > 1, c is redundant
                if np.min(cover_count[balls[c]]) > 1:
                    cw.pop(i)
                    changed = True
                    break
            return cw

    # Main Search Loop
    while time.time() < deadline - 5:
        # Phase 1: Remove all redundancies
        current = best_codewords[:]
        greedy_removal(current)
        
        if len(current) < best_size:
            best_size = len(current)
            best_codewords = current[:]
            # Reset perturbation counter? No, just continue.
            continue
            
        # Phase 2: Perturbation to escape local minimum
        # Pick a random codeword to swap out
        if not current:
            break
            
        # Try a few random swaps
        swap_found = False
        for _ in range(20): # Limit swaps per iteration to avoid wasting time
            if time.time() > deadline - 5: break
            
            idx_out = np.random.randint(len(current))
            c_out = current[idx_out]
            
            # Pick a random candidate to swap in
            # To be more effective, pick a candidate that covers uncovered parts? 
            # But we are at a local minimum, so nothing is uncovered.
            # We swap to change the structure.
            # Let's pick a random word from the whole space, or one that is "close" to c_out?
            # Random is simpler.
            c_in = np.random.randint(total_words)
            
            if c_in == c_out:
                continue
                
            new_set = current[:]
            new_set[idx_out] = c_in
            
            # Check validity
            if is_valid(new_set):
                # It's a valid set of same size. 
                # We accept it if it allows further removals? 
                # We don't know yet. Let's just accept it and let the next greedy_removal try to shrink it.
                # To avoid cycling, we can keep a history or just accept.
                # Since we always do greedy_removal next, if it doesn't shrink, we might cycle.
                # Simple anti-cycling: only accept if c_in is "better" in some metric? 
                # Or just accept and rely on randomness.
                # Let's accept.
                best_codewords = new_set
                swap_found = True
                break
            else:
                # Invalid, try to repair? 
                # Repairing adds codewords, which we don't want immediately.
                # Just skip this swap.
                pass
                
        if not swap_found:
            # If no valid swap found after 20 tries, maybe try a more directed search?
            # Or just break to avoid infinite loop of useless attempts?
            # Let's try a different perturbation: remove one and add two? (Size +1, then remove 2?)
            # That's complex. Let's just break and restart greedy? 
            # Actually, if we can't find a valid swap, we are stuck. 
            # Let's try to remove a random codeword and see if we can cover the gap with a smarter choice?
            # This is essentially the "Greedy-Remove + Random-Insert" from inspirations.
            
            # Let's implement a single step of that:
            # 1. Remove a random codeword c_out.
            # 2. Find the best c_in to cover the newly uncovered words.
            # 3. If valid, accept.
            
            idx_out = np.random.randint(len(current))
            c_out = current[idx_out]
            
            # Compute coverage without c_out
            # cover_count_without = M[current without c_out].sum(axis=0)
            # Uncovered = where cover_count_without == 0
            
            # To do this efficiently without full sum every time:
            # We know cover_count of current set.
            # Removing c_out subtracts M[c_out].
            
            # Let's just compute it.
            if len(current) > 1:
                others = [c for i, c in enumerate(current) if i != idx_out]
                if others:
                    cover_others = M[others, :].sum(axis=0)
                    uncovered = np.where(cover_others == 0)[0]
                    
                    if len(uncovered) > 0:
                        # Find best c_in to cover max uncovered
                        # scores = M.dot(uncovered_mask)
                        # But we want to cover the specific uncovered set.
                        # Since uncovered is a set of indices, we can sum M[:, uncovered]
                        # M[:, uncovered] is (total_words, len(uncovered))
                        # Sum over axis 1 gives (total_words,)
                        
                        # Optimization: if uncovered is small, this is fast.
                        # If uncovered is large, it's slower.
                        # But radius=1, ball size is 15. So removing one word uncovers at most 15 words.
                        # So len(uncovered) <= 15.
                        
                        if len(uncovered) > 0:
                            # Vectorized: M[:, uncovered].sum(axis=1)
                            # This creates a (2187, len(uncovered)) slice.
                            # Summing it is fast.
                            scores = M[:, uncovered].sum(axis=1)
                            best_in = np.argmax(scores)
                            
                            if scores[best_in] > 0:
                                new_set = current[:]
                                new_set[idx_out] = best_in
                                
                                # Check validity (should be valid if scores[best_in] == len(uncovered))
                                # But it might not cover ALL if best_in doesn't cover all.
                                # Wait, if scores[best_in] == len(uncovered), it covers all.
                                # If not, it's invalid.
                                # We can check validity quickly.
                                if is_valid(new_set):
                                    best_codewords = new_set
                                    # Do not break, let the loop continue to greedy_removal
                                    break
                                
    # Final Pruning
    changed = True
    while changed and time.time() < deadline - 1:
        changed = False
        cover_count = M[best_codewords, :].sum(axis=0)
        for i in range(len(best_codewords)):
            c = best_codewords[i]
            if np.min(cover_count[balls[c]]) > 1:
                best_codewords.pop(i)
                changed = True
                break

    # Convert to output format
    result = []
    for c in best_codewords:
        result.append(int_to_tuple(c, n, q))
        
    return result
