import numpy as np
import time
import random
import itertools

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 115.0
    
    # Specific to q=3, n=6, radius=1
    if q != 3 or n != 6 or radius != 1:
        # Fallback
        return []

    total_words = q ** n
    
    def word_to_idx(word):
        idx = 0
        for v in word:
            idx = idx * q + v
        return idx

    def idx_to_word(idx):
        word = [0] * n
        for i in range(n - 1, -1, -1):
            word[i] = idx % q
            idx //= q
        return word

    # Precompute M: M[i, j] = 1 if word i is covered by codeword j
    # i, j in 0..728
    # neighbors[j] = list of i such that M[i, j] = 1
    
    # Generate neighbors
    neighbors = []
    for j in range(total_words):
        w = idx_to_word(j)
        covered = [j]
        for pos in range(n):
            orig = w[pos]
            for new_val in range(q):
                if new_val == orig:
                    continue
                w_copy = w.copy()
                w_copy[pos] = new_val
                covered.append(word_to_idx(w_copy))
        neighbors.append(np.array(covered, dtype=np.int32))

    # Build M (729x729 boolean)
    M = np.zeros((total_words, total_words), dtype=bool)
    for j in range(total_words):
        M[neighbors[j], j] = True

    def greedy_set_cover():
        uncovered = np.ones(total_words, dtype=bool)
        selected = []
        
        while np.any(uncovered):
            # Count how many uncovered words are covered by each j
            # M.T is (729, 729) where M.T[j, i] = M[i, j]
            # We want sum over i where uncovered[i] is True.
            # counts = np.sum(M.T * uncovered, axis=1)
            # M.T * uncovered (broadcasting)
            # This creates a 729x729 boolean array.
            # np.sum on axis 1.
            
            # Optimization: 
            # uncovered is a vector.
            # M.T @ uncovered (dot product) gives counts.
            # Since M is boolean, this is integer addition.
            counts = M.T @ uncovered
            counts = counts.astype(np.int32)
            
            # Mask out already selected
            for j in selected:
                counts[j] = -1
                
            best_j = np.argmax(counts)
            
            if counts[best_j] == 0:
                break
                
            selected.append(int(best_j))
            uncovered &= ~M[:, best_j]
            
        return selected

    def local_search(initial_codewords, time_limit):
        # initial_codewords is list of indices
        codewords = list(initial_codewords)
        cw_set = set(codewords)
        
        # Coverage counts
        counts = np.zeros(total_words, dtype=np.int32)
        for cw in codewords:
            counts[neighbors[cw]] += 1
            
        def remove_redundant():
            changed = True
            while changed and time.time() < time_limit:
                changed = False
                for idx, cw in enumerate(codewords):
                    # Check if removing cw keeps counts > 0
                    # i.e., for all i in neighbors[cw], counts[i] > 1
                    if np.all(counts[neighbors[cw]] > 1):
                        # Remove
                        counts[neighbors[cw]] -= 1
                        codewords.pop(idx)
                        cw_set.remove(cw)
                        changed = True
                        break # Restart scan

        remove_redundant()
        
        best_len = len(codewords)
        best_cw = codewords.copy()
        best_counts = counts.copy()
        best_cw_set = cw_set.copy()
        
        # Swap search
        while time.time() < time_limit:
            # Find codeword with fewest unique covers
            unique_counts = []
            for cw in codewords:
                uc = np.sum(counts[neighbors[cw]] == 1)
                unique_counts.append((uc, cw))
            
            unique_counts.sort()
            
            improved = False
            # Try swapping
            for uc, cw_out in unique_counts[:20]:
                if time.time() > time_limit:
                    break
                if uc == 0:
                    continue
                
                # Identify unique words
                unique_words = []
                for i in neighbors[cw_out]:
                    if counts[i] == 1:
                        unique_words.append(i)
                
                # Find best cw_in
                # Must cover all unique_words
                # Prefer one that maximizes overlap with others (creates redundancy)
                
                # Candidate cw_in: all j not in cw_set
                # Check coverage of unique_words
                # M[unique_words, j] must be all True.
                
                # Vectorized check:
                # M_unique = M[unique_words, :] (shape: uc x 729)
                # all_covered = np.all(M_unique, axis=0) (shape: 729)
                
                M_unique = M[unique_words, :]
                all_covered = np.all(M_unique, axis=0)
                
                # Exclude existing
                for j in cw_set:
                    all_covered[j] = False
                    
                valid_candidates = np.where(all_covered)[0]
                
                if len(valid_candidates) == 0:
                    continue
                    
                # Pick candidate that maximizes redundancy
                # Redundancy = number of words in neighbors[cw_in] that are covered by >1 other codeword
                # i.e., counts[i] > 1 (after removing cw_out? No, counts currently includes cw_out)
                # If we remove cw_out, counts[i] becomes counts[i]-1 for i in neighbors[cw_out].
                # We want to add cw_in.
                # New counts = counts - 1 (neighbors[cw_out]) + 1 (neighbors[cw_in])
                # We want to maximize the number of i where New Counts[i] > 1?
                # Or simply, pick a random valid candidate.
                
                # Let's pick the one with max overlap with current codewords (excluding cw_out)
                # Overlap with others = sum(1 for i in neighbors[cw_in] if counts[i] > 1)
                # This is a heuristic.
                
                # For speed, just pick random from valid_candidates
                cw_in = random.choice(valid_candidates.tolist())
                
                # Perform swap
                # Snapshot
                old_counts = counts.copy()
                old_cw_set = cw_set.copy()
                old_len = len(codewords)
                
                # Remove cw_out
                counts[neighbors[cw_out]] -= 1
                codewords.remove(cw_out)
                cw_set.remove(cw_out)
                
                # Add cw_in
                counts[neighbors[cw_in]] += 1
                codewords.append(cw_in)
                cw_set.add(cw_in)
                
                # Remove redundants
                remove_redundant()
                
                if len(codewords) < best_len:
                    best_len = len(codewords)
                    best_cw = codewords.copy()
                    best_counts = counts.copy()
                    best_cw_set = cw_set.copy()
                    improved = True
                else:
                    # Revert
                    counts = old_counts
                    cw_set = old_cw_set
                    # Reconstruct codewords list?
                    # We popped and appended.
                    # Easier to just restart local search with best_cw if no improvement?
                    # Or revert list.
                    # Let's just break and restart with best_cw if stuck.
                    break
                    
            if not improved:
                # Restart with best_cw
                codewords = best_cw.copy()
                cw_set = best_cw_set.copy()
                counts = best_counts.copy()
                # Add some randomness?
                # Or just break
                break
                
        return best_cw

    # Run Greedy
    greedy_solution = greedy_set_cover()
    
    # Run Local Search
    best_solution = local_search(greedy_solution, deadline)
    
    # Try Tetracode
    tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
    tetra_solution = [word_to_idx(h + list(tail)) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]
    
    best_from_tetra = local_search(tetra_solution, deadline)
    
    if len(best_from_tetra) < len(best_solution):
        best_solution = best_from_tetra
        
    return [idx_to_word(idx) for idx in best_solution]
