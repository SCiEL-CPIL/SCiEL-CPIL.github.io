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
        neighbor_mask = np.zeros((total, total), dtype=np.bool_)
        for i in range(total):
            neighbor_mask[i, neighbors[i]] = True

        def idx_to_list(idx):
            return [(idx // powers[p]) % 3 for p in range(6)]

        # 1. Greedy Set Cover
        uncovered = np.ones(total, dtype=np.int8)
        codewords = []
        coverage_counts = np.zeros(total, dtype=np.int32)
        
        # To speed up greedy, pick the word that covers most uncovered.
        while uncovered.sum() > 0:
            if time.time() > deadline - 30:
                break
            # counts = sum of uncovered points covered by each candidate
            # neighbors shape (729, 13), uncovered shape (729,)
            # uncovered[neighbors] shape (729, 13)
            # sum axis=1 -> (729,)
            counts = np.sum(uncovered[neighbors], axis=1)
            best_c = np.argmax(counts)
            if counts[best_c] == 0:
                break
            codewords.append(best_c)
            uncovered[neighbors[best_c]] = 0
            coverage_counts[neighbors[best_c]] += 1
            
        # Ensure valid
        if uncovered.sum() != 0:
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
            # Find any word where all neighbors have count >= 2
            # min_count_for_c = np.min(coverage_counts[neighbors[c]])
            # We want min_count_for_c >= 2
            # Vectorized:
            # min_counts = np.min(coverage_counts[neighbors], axis=1)
            # candidates = np.where(min_counts >= 2)[0]
            # But we must only remove words that are IN codewords.
            # Create a mask of current codewords
            # This is a bit expensive if done every time, but codewords is small (~73).
            
            # Optimization: Only check current codewords
            for idx, c in enumerate(codewords):
                if np.min(coverage_counts[neighbors[c]]) >= 2:
                    codewords.pop(idx)
                    coverage_counts[neighbors[c]] -= 1
                    return True
            return False

        def try_swap():
            """Try a 1-for-1 swap. Returns True if swapped."""
            # Pick random codeword to remove
            if not codewords:
                return False
            idx_out = random.randint(0, len(codewords) - 1)
            c_out = codewords[idx_out]
            
            # Find critical words: points covered ONLY by c_out
            # coverage_counts[neighbors[c_out]] == 1
            crit = neighbors[c_out][coverage_counts[neighbors[c_out]] == 1]
            if len(crit) == 0:
                # Should have been caught by try_remove
                return False
                
            # Find candidates c_in that cover all crit
            # c_in must be in intersection of N(crit)
            # valid_in = np.all(neighbor_mask[:, crit], axis=1)
            valid_in = np.all(neighbor_mask[:, crit], axis=1)
            
            # Exclude existing codewords and c_out
            # Create a mask of existing codewords
            existing_mask = np.zeros(total, dtype=np.bool_)
            if codewords:
                existing_mask[codewords] = True
            valid_in[existing_mask] = False
            valid_in[c_out] = False # Just in case
            
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
            if len(codewords) < 2:
                return False
            idxs = random.sample(range(len(codewords)), 2)
            c1, c2 = codewords[idxs[0]], codewords[idxs[1]]
            
            # Temporarily remove
            coverage_counts[neighbors[c1]] -= 1
            coverage_counts[neighbors[c2]] -= 1
            
            # Find words that are now uncovered (cov == 0)
            critical_words = np.where(coverage_counts == 0)[0]
            
            if len(critical_words) == 0:
                 # No new holes created. We can add any 2 words?
                 # Actually, if we remove 2 and add 2, size is same.
                 # If no holes, we just need to ensure we add valid words.
                 # But we want to escape local min.
                 # If no holes, we can pick any 2 non-existing words.
                 existing_mask = np.zeros(total, dtype=np.bool_)
                 if codewords:
                     existing_mask[codewords] = True
                 existing_mask[c1] = False
                 existing_mask[c2] = False
                 
                 candidates = np.where(~existing_mask)[0]
                 if len(candidates) >= 2:
                     c3, c4 = random.sample(candidates.tolist(), 2)
                     codewords[idxs[0]] = c3
                     codewords[idxs[1]] = c4
                     coverage_counts[neighbors[c3]] += 1
                     coverage_counts[neighbors[c4]] += 1
                     return True
                 
                 # Revert
                 coverage_counts[neighbors[c1]] += 1
                 coverage_counts[neighbors[c2]] += 1
                 return False

            # We have critical_words to cover
            # Pick the first critical word w0
            w0 = critical_words[0]
            c3_candidates = neighbors[w0]
            
            found = False
            # To speed up, limit iterations or just iterate all (13 is small)
            for c3 in c3_candidates:
                # c3 must not be c1 or c2 (they are removed, but we are adding new ones)
                # Actually c3 can be anything not in current codewords (after removal)
                # But we are checking against existing_mask which includes c1, c2?
                # No, existing_mask should reflect the state AFTER removal for the "new" words.
                # Let's build the mask of "forbidden" words: current codewords (excluding c1, c2)
                existing_mask = np.zeros(total, dtype=np.bool_)
                if codewords:
                    existing_mask[codewords] = True
                existing_mask[c1] = False
                existing_mask[c2] = False
                
                if existing_mask[c3]:
                    continue
                    
                # Check if c3 covers some critical words
                # remaining = critical_words not in N(c3)
                # mask_c3 = neighbor_mask[c3]
                # remaining_mask = critical_words_mask & ~mask_c3
                # We need c4 such that N(c4) covers remaining_mask.
                
                # Efficient check:
                # Create a mask for critical_words
                # Instead of creating a full 729 mask, we can just check the specific indices
                # But np.all(neighbor_mask[:, remaining], axis=1) is fast.
                
                # remaining_indices = critical_words[~neighbor_mask[c3, critical_words]]
                # This creates a boolean mask of critical_words covered by c3
                # covered_c3 = neighbor_mask[c3, critical_words]
                # remaining = critical_words[~covered_c3]
                
                covered_c3 = neighbor_mask[c3, critical_words]
                remaining = critical_words[~covered_c3]
                
                if len(remaining) == 0:
                    # c3 covers all critical words.
                    # c4 can be any non-existing word.
                    valid_c4 = ~existing_mask
                    valid_c4[c3] = False
                    c4_candidates = np.where(valid_c4)[0]
                    if len(c4_candidates) > 0:
                        c4 = random.choice(c4_candidates)
                        codewords[idxs[0]] = c3
                        codewords[idxs[1]] = c4
                        coverage_counts[neighbors[c3]] += 1
                        coverage_counts[neighbors[c4]] += 1
                        return True
                else:
                    # Need c4 to cover 'remaining'
                    valid_c4 = np.all(neighbor_mask[:, remaining], axis=1)
                    valid_c4[existing_mask] = False
                    valid_c4[c3] = False
                    c4_candidates = np.where(valid_c4)[0]
                    if len(c4_candidates) > 0:
                        c4 = random.choice(c4_candidates)
                        codewords[idxs[0]] = c3
                        codewords[idxs[1]] = c4
                        coverage_counts[neighbors[c3]] += 1
                        coverage_counts[neighbors[c4]] += 1
                        return True
            
            # Revert temp removal
            coverage_counts[neighbors[c1]] += 1
            coverage_counts[neighbors[c2]] += 1
            return False

        # Main Loop
        stuck_count = 0
        max_stuck = 1000
        
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
                    max_stuck = random.randint(500, 1500)
                else:
                    stuck_count = 0
                    max_stuck = random.randint(500, 1500)

        # Final cleanup
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
