"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import random
import time
import itertools

def construct(q, n, radius):
    """
    Construct a covering code for {0,1,2}^n with covering radius 1.
    Uses a local search heuristic starting from the 486-word direct sum construction.
    """
    if q != 3 or n != 8 or radius != 1:
        # Fallback for other parameters
        if n <= 4:
            if n == 1:
                return [[i] for i in range(q)]
            elif n == 2:
                return [list(w) for w in itertools.product(range(q), repeat=2)]
            elif n == 4:
                tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
                return tetracode
            else:
                tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
                return [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]
        else:
            # General fallback: direct sum
            tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
            return [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]

    # Initialize with the 486-word direct sum construction
    tetracode = np.array([[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)])
    tails = np.array(list(itertools.product(range(3), repeat=4)))
    codewords = np.hstack([np.tile(tetracode, (len(tails), 1)), np.repeat(tails, len(tetracode), axis=0)])
    
    # Convert to list of tuples for easier set operations if needed, but we will use arrays for speed
    # Actually, let's stick to a list of lists for the final return, but use numpy for internal computation
    # Let's represent codewords as a list of numpy arrays? No, a single 2D array is better.
    # Let's maintain a set of indices for current codewords? 
    # Simpler: maintain a list of codewords (tuples) and a coverage count array.
    
    # Precompute all words as indices 0..3^8-1? 
    # Mapping: word w to index idx = sum(w[i] * 3^i)
    # This allows O(1) coverage count lookup.
    
    base = 3
    power = [1]
    for i in range(n):
        power.append(power[-1] * base)
    total_words = power[n]
    
    def word_to_idx(w):
        idx = 0
        for i in range(n):
            idx += w[i] * power[i]
        return idx
        
    def idx_to_word(idx):
        w = [0] * n
        for i in range(n):
            w[i] = idx % base
            idx //= base
        return w

    # Initial codewords
    initial_cw = [tuple(row) for row in codewords]
    
    # Coverage count array
    cov_counts = np.zeros(total_words, dtype=np.int32)
    
    # Helper to update coverage
    def update_coverage(cw, delta):
        # cw is a tuple
        idx = word_to_idx(cw)
        cov_counts[idx] += delta
        for i in range(n):
            for val in range(q):
                if val != cw[i]:
                    new_cw = list(cw)
                    new_cw[i] = val
                    new_idx = word_to_idx(tuple(new_cw))
                    cov_counts[new_idx] += delta

    # Initialize coverage
    for cw in initial_cw:
        update_coverage(cw, 1)
        
    # Current set of codewords (as tuples)
    current_cw_set = set(initial_cw)
    
    start_time = time.time()
    time_limit = 115.0 # Leave 5s buffer
    
    # Phase 1: Greedy Removal
    # Repeatedly try to remove codewords that are redundant
    # A codeword is redundant if all words it covers have cov_counts > 1
    
    def can_remove(cw):
        idx = word_to_idx(cw)
        if cov_counts[idx] <= 1:
            return False
        for i in range(n):
            for val in range(q):
                if val != cw[i]:
                    new_cw = list(cw)
                    new_cw[i] = val
                    new_idx = word_to_idx(tuple(new_cw))
                    if cov_counts[new_idx] <= 1:
                        return False
        return True

    # Loop until no more removals or time out
    while time.time() - start_time < time_limit:
        removed_any = False
        # Shuffle to avoid bias
        cw_list = list(current_cw_set)
        random.shuffle(cw_list)
        
        for cw in cw_list:
            if cw not in current_cw_set:
                continue
            if can_remove(cw):
                update_coverage(cw, -1)
                current_cw_set.remove(cw)
                removed_any = True
                if time.time() - start_time > time_limit:
                    break
        if not removed_any:
            break
            
    # Phase 2: Critical Word Heuristic / Swapping
    # Identify words with cov_count == 1
    # For each such word, find the unique codeword covering it.
    # Try to replace that codeword with a new random codeword that also covers the critical word.
    # Accept if the new codeword allows for more removals or simply if it reduces the "cost" (sum of cov_counts^2 or similar).
    # Simpler: Just try to find a replacement that doesn't increase the number of codewords and potentially allows removal.
    # Actually, just doing swaps is complex. Let's try "Add and Remove".
    
    # Let's try a simpler local search:
    # 1. Find all critical words (cov_count == 1).
    # 2. For each critical word w, let c_old be the unique cover.
    # 3. Generate a random c_new that covers w.
    # 4. Check if adding c_new and removing c_old is valid (i.e., c_new covers all words that c_old covered uniquely, 
    #    or more generally, the coverage remains >= 1 everywhere).
    #    Actually, if we remove c_old, the words only covered by c_old become uncovered unless c_new covers them.
    #    So we need: For all w' in covered(c_old) \ covered(c_new), cov_counts[w'] >= 2 (before removal).
    #    And for w (the critical word), c_new must cover it.
    
    # Optimization: 
    # We can precompute the list of words covered by each codeword? 
    # 3^8 = 6561 words. 486 codewords. 
    # Storing covered words for each cw is 486 * 25 = 12150 entries. Very small.
    
    # Let's rebuild the structure to use explicit covered word lists for faster checks.
    
    # Re-initialize state for this phase to ensure consistency
    # Actually, let's just continue from the current state.
    
    # Build mapping: word_idx -> list of codewords covering it
    # This is expensive to build from scratch, but we have cov_counts.
    # We can rebuild it.
    
    word_to_cw = [[] for _ in range(total_words)]
    for cw in current_cw_set:
        idx = word_to_idx(cw)
        word_to_cw[idx].append(cw)
        for i in range(n):
            for val in range(q):
                if val != cw[i]:
                    new_cw = list(cw)
                    new_cw[i] = val
                    new_idx = word_to_idx(tuple(new_cw))
                    word_to_cw[new_idx].append(cw)
                    
    # Critical words
    critical_words = [i for i in range(total_words) if cov_counts[i] == 1]
    
    # Iterate over critical words and try to fix them
    # We will do multiple passes
    
    max_passes = 50
    for p in range(max_passes):
        if time.time() - start_time > time_limit:
            break
            
        # Re-identify critical words
        critical_words = [i for i in range(total_words) if cov_counts[i] == 1]
        
        if not critical_words:
            break
            
        random.shuffle(critical_words)
        
        for w_idx in critical_words:
            if time.time() - start_time > time_limit:
                break
                
            if cov_counts[w_idx] != 1:
                continue
                
            # Get the unique codeword covering this word
            old_cw = word_to_cw[w_idx][0]
            
            # Generate a random new codeword that covers w_idx
            # w_idx corresponds to word w
            w = idx_to_word(w_idx)
            
            # A random word covering w is w itself, or w with one position changed.
            # To be a "good" candidate, it should ideally cover other words that are heavily covered, 
            # or allow removal of other codewords.
            # Let's just pick a random word in the ball of radius 1 around w.
            
            # Candidates: w, and w with one symbol changed.
            candidates = [tuple(w)]
            for i in range(n):
                for val in range(q):
                    if val != w[i]:
                        new_w = list(w)
                        new_w[i] = val
                        candidates.append(tuple(new_w))
                        
            random.shuffle(candidates)
            
            improved = False
            for new_cw in candidates:
                if new_cw == old_cw:
                    continue
                if new_cw in current_cw_set:
                    continue
                    
                # Check if replacing old_cw with new_cw is valid
                # Condition: For all words x in covered(old_cw) \ covered(new_cw), 
                # cov_counts[x] >= 2 (so that removing old_cw doesn't drop them to 0).
                # Also, new_cw must cover w_idx (which it does by construction).
                
                # Efficient check:
                # Words only covered by old_cw are those x in covered(old_cw) where cov_counts[x] == 1.
                # If new_cw does not cover such an x, the replacement is invalid.
                
                # Let's list words covered by old_cw
                old_covered = []
                old_covered.append(word_to_idx(old_cw))
                for i in range(n):
                    for val in range(q):
                        if val != old_cw[i]:
                            nc = list(old_cw)
                            nc[i] = val
                            old_covered.append(word_to_idx(tuple(nc)))
                            
                # Check if new_cw covers all critical parts of old_cw
                valid = True
                for x_idx in old_covered:
                    if cov_counts[x_idx] == 1:
                        # This word is only covered by old_cw.
                        # new_cw MUST cover it.
                        # Check if x_idx is in covered(new_cw)
                        # x_idx is in covered(new_cw) if dist(x, new_cw) <= 1
                        # x is idx_to_word(x_idx)
                        # Instead of decoding, let's check distance
                        # We can precompute covered set for new_cw? No, too slow per candidate.
                        # Let's just check distance.
                        x_word = idx_to_word(x_idx)
                        dist = sum(1 for i in range(n) if x_word[i] != new_cw[i])
                        if dist > 1:
                            valid = False
                            break
                            
                if not valid:
                    continue
                    
                # If valid, perform the swap
                # Remove old_cw
                update_coverage(old_cw, -1)
                for x_idx in old_covered:
                    word_to_cw[x_idx].remove(old_cw)
                    if len(word_to_cw[x_idx]) == 0:
                         # Should not happen if valid, but safety check
                         pass 
                current_cw_set.remove(old_cw)
                
                # Add new_cw
                update_coverage(new_cw, 1)
                new_covered = []
                new_covered.append(word_to_idx(new_cw))
                for i in range(n):
                    for val in range(q):
                        if val != new_cw[i]:
                            nc = list(new_cw)
                            nc[i] = val
                            new_covered.append(word_to_idx(tuple(nc)))
                for x_idx in new_covered:
                    word_to_cw[x_idx].append(new_cw)
                current_cw_set.add(new_cw)
                
                improved = True
                break # Try next critical word
                
            if not improved:
                # Try random perturbation? 
                # For now, just move on.
                pass
                
    # Final Phase: Greedy Removal again after swaps
    while time.time() - start_time < time_limit:
        removed_any = False
        cw_list = list(current_cw_set)
        random.shuffle(cw_list)
        
        for cw in cw_list:
            if cw not in current_cw_set:
                continue
            if can_remove(cw):
                update_coverage(cw, -1)
                current_cw_set.remove(cw)
                removed_any = True
                if time.time() - start_time > time_limit:
                    break
        if not removed_any:
            break
            
    # Verify and return
    final_cw = list(current_cw_set)
    
    # Double check coverage
    # Rebuild coverage from scratch to be sure
    final_cov = np.zeros(total_words, dtype=np.int32)
    for cw in final_cw:
        update_coverage(cw, 1) # Note: this modifies the global cov_counts, 
                               # but we are done with the search, so it's fine if we just use this for verification
                               # Actually, update_coverage modifies the `cov_counts` variable in the closure.
                               # Let's do a separate check.
    
    # Proper verification
    check_cov = np.zeros(total_words, dtype=np.int32)
    for cw in final_cw:
        idx = word_to_idx(cw)
        check_cov[idx] += 1
        for i in range(n):
            for val in range(q):
                if val != cw[i]:
                    new_cw = list(cw)
                    new_cw[i] = val
                    new_idx = word_to_idx(tuple(new_cw))
                    check_cov[new_idx] += 1
                    
    if np.any(check_cov == 0):
        # Fallback to initial
        return [list(h + list(tail)) for h in tetracode for tail in itertools.product(range(3), repeat=4)]
        
    return [list(cw) for cw in final_cw]

# EVOLVE-BLOCK-END
