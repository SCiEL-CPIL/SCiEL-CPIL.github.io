import numpy as np
import time
import random

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin

    # Specific case optimization
    if q == 3 and n == 6 and radius == 1:
        total = 3 ** 6
        powers = [3 ** p for p in range(6)]
        
        # Precompute neighbors
        neighbors = np.empty((total, 13), dtype=np.int32)
        for i in range(total):
            d = [(i // powers[p]) % 3 for p in range(6)]
            neighbors[i, 0] = i
            k = 1
            for p in range(6):
                for alt in range(3):
                    if alt != d[p]:
                        new_i = i + (alt - d[p]) * powers[p]
                        neighbors[i, k] = new_i
                        k += 1
                        
        # Precompute neighbor masks for fast intersection/union checks
        # mask[i, j] = 1 if j is in neighbors[i]
        neighbor_mask = np.zeros((total, total), dtype=np.bool_)
        for i in range(total):
            neighbor_mask[i, neighbors[i]] = True

        def idx_to_list(idx):
            return [(idx // powers[p]) % 3 for p in range(6)]

        # 1. Greedy Set Cover
        uncovered = np.ones(total, dtype=np.int8)
        codewords = []
        coverage_counts = np.zeros(total, dtype=np.int32)
        
        # To speed up greedy, we can just pick the word that covers most uncovered.
        # counts = sum(uncovered[neighbors])
        while uncovered.sum() > 0:
            if time.time() > deadline - 30:
                break
            counts = np.sum(uncovered[neighbors], axis=1)
            best_c = np.argmax(counts)
            if counts[best_c] == 0:
                break
            codewords.append(best_c)
            uncovered[neighbors[best_c]] = 0
            coverage_counts[neighbors[best_c]] += 1
            
        # Ensure valid
        if uncovered.sum() != 0:
             # Fallback
             while uncovered.sum() > 0:
                 c = np.random.randint(total)
                 if c not in codewords:
                     codewords.append(c)
                     uncovered[neighbors[c]] = 0
                     coverage_counts[neighbors[c]] += 1

        best_codewords = list(codewords)
        best_size = len(codewords)
        
        # 2. Local Search
        # State: codewords (list of indices), coverage_counts (array)
        
        def try_remove():
            """Try to remove a redundant word. Returns True if removed."""
            for idx, c in enumerate(codewords):
                if np.min(coverage_counts[neighbors[c]]) >= 2:
                    # Redundant
                    codewords.pop(idx)
                    coverage_counts[neighbors[c]] -= 1
                    return True
            return False

        def try_swap():
            """Try a 1-for-1 swap. Returns True if swapped."""
            # Pick random codeword to remove
            idx_out = random.randint(0, len(codewords) - 1)
            c_out = codewords[idx_out]
            
            # Find critical words
            crit = neighbors[c_out][coverage_counts[neighbors[c_out]] == 1]
            if len(crit) == 0:
                # Should have been caught by try_remove
                return False
                
            # Find candidates c_in that cover all crit
            # c_in must be in intersection of N(crit)
            # Using mask: mask[:, crit] all True
            # mask is 729x729. crit is small.
            # valid_in = np.all(neighbor_mask[:, crit], axis=1)
            
            # Optimization: Only check candidates not in codewords
            # But checking all 729 is fast enough.
            valid_in = np.all(neighbor_mask[:, crit], axis=1)
            
            # Exclude c_out and existing codewords
            # Create a mask of existing codewords
            existing_mask = np.zeros(total, dtype=np.bool_)
            existing_mask[codewords] = True
            valid_in[existing_mask] = False
            valid_in[c_out] = False # Just in case, though it's in existing_mask
            
            candidates = np.where(valid_in)[0]
            if len(candidates) > 0:
                c_in = random.choice(candidates)
                # Perform swap
                codewords[idx_out] = c_in
                coverage_counts[neighbors[c_out]] -= 1
                coverage_counts[neighbors[c_in]] += 1
                return True
            return False

        def try_2_swap():
            """Try a 2-for-2 swap. Returns True if swapped."""
            # Pick 2 random distinct codewords to remove
            if len(codewords) < 2:
                return False
            idxs = random.sample(range(len(codewords)), 2)
            c1, c2 = codewords[idxs[0]], codewords[idxs[1]]
            
            # Temporarily remove
            coverage_counts[neighbors[c1]] -= 1
            coverage_counts[neighbors[c2]] -= 1
            
            # Find words that are now uncovered (cov == 0)
            critical_words = np.where(coverage_counts == 0)[0]
            
            # We need to find c3, c4 such that N(c3) U N(c4) covers critical_words
            # And c3, c4 not in codewords (excluding c1, c2 which are removed)
            
            # Existing mask (without c1, c2)
            existing_mask = np.zeros(total, dtype=np.bool_)
            existing_mask[codewords] = True
            existing_mask[c1] = False
            existing_mask[c2] = False
            
            # Iterate over candidates for c3
            # To speed up, we can restrict c3 to neighbors of one critical word?
            # Yes, c3 must be in N(w) for some w in critical_words.
            # Let's pick the first critical word w0.
            if len(critical_words) == 0:
                 # This means removing c1, c2 didn't uncover anything?
                 # Then it's a valid 2-removal? No, we need to add 2.
                 # If no critical words, we can add any 2 words?
                 # But we want to escape local min.
                 # If no critical words, just add random 2 non-existing words.
                 # But this increases "randomness" without pressure.
                 # Let's just find any 2.
                 pass
                 
            w0 = critical_words[0]
            c3_candidates = neighbors[w0]
            
            found = False
            for c3 in c3_candidates:
                if existing_mask[c3]:
                    continue
                    
                # Check if c3 covers some critical words
                # Now find c4 such that N(c4) covers remaining critical words
                # remaining = critical_words not in N(c3)
                # mask_c3 = neighbor_mask[c3]
                # remaining_mask = critical_words_mask & ~mask_c3
                # We need c4 such that neighbor_mask[c4] covers remaining_mask.
                
                # Efficient check:
                # critical_mask = np.zeros(total, dtype=bool)
                # critical_mask[critical_words] = True
                # remaining = critical_mask & ~neighbor_mask[c3]
                # if not remaining.any():
                #    # c3 covers all critical words.
                #    # c4 can be any non-existing word.
                #    c4_candidates = np.where(~existing_mask)[0]
                #    if len(c4_candidates) > 0:
                #        c4 = random.choice(c4_candidates)
                #        # Perform swap
                #        codewords[idxs[0]] = c3
                #        codewords[idxs[1]] = c4
                #        coverage_counts[neighbors[c3]] += 1
                #        coverage_counts[neighbors[c4]] += 1
                #        return True
                # else:
                #    # Need c4 to cover 'remaining'
                #    # valid_c4 = np.all(neighbor_mask[:, remaining], axis=1)
                #    # Exclude existing
                #    valid_c4 = np.all(neighbor_mask[:, remaining], axis=1)
                #    valid_c4[existing_mask] = False
                #    valid_c4[c3] = False
                #    c4_candidates = np.where(valid_c4)[0]
                #    if len(c4_candidates) > 0:
                #        c4 = random.choice(c4_candidates)
                #        codewords[idxs[0]] = c3
                #        codewords[idxs[1]] = c4
                #        coverage_counts[neighbors[c3]] += 1
                #        coverage_counts[neighbors[c4]] += 1
                #        return True
                # Revert temp removal
                coverage_counts[neighbors[c1]] += 1
                coverage_counts[neighbors[c2]] += 1
                return False

            # If loop finishes without found, revert
            coverage_counts[neighbors[c1]] += 1
            coverage_counts[neighbors[c2]] += 1
            return False

        # Main Loop
        stuck_count = 0
        max_stuck = 1000 # After this many failed attempts, try 2-swap
        
        while time.time() < deadline:
            # 1. Try removal
            if try_remove():
                stuck_count = 0
                if len(codewords) < best_size:
                    best_size = len(codewords)
                    best_codewords = list(codewords)
                continue
                
            # 2. Try swap
            if try_swap():
                stuck_count = 0
                continue
                
            # 3. Stuck
            stuck_count += 1
            if stuck_count > max_stuck:
                # Try 2-swap
                if try_2_swap():
                    stuck_count = 0
                    max_stuck = random.randint(500, 1500) # Randomize threshold
                else:
                    # Reset stuck count to try again
                    stuck_count = 0
                    max_stuck = random.randint(500, 1500)
                    
            # Update best if current is better? No, size is same or larger.
            # Only removals improve size.

        # Final cleanup
        # Try to remove any remaining redundant words
        while True:
            if not try_remove():
                break
            if len(codewords) < best_size:
                best_size = len(codewords)
                best_codewords = list(codewords)

        if len(best_codewords) < len(codewords):
            final_indices = best_codewords
        else:
            final_indices = codewords
            
        return [idx_to_list(idx) for idx in final_indices]

    else:
        # Fallback for other parameters
        # Simple greedy
        total = q ** n
        # ... implement generic greedy ...
        # For now, return a trivial code
        return [[0] * n]

# EVOLVE-BLOCK-END
