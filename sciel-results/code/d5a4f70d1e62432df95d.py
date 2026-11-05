import itertools
import time
import sys
import numpy as np

def construct(q, n, radius):
    # For this specific task: q=3, n=7, radius=1
    # But the function signature is general. We will specialize for the known task 
    # but keep the structure. If parameters differ, we might fall back to a simple 
    # construction or raise an error, but the prompt implies this is for the specific 
    # ternary covering code problem.
    
    if q != 3 or n != 7 or radius != 1:
        # Fallback to a simple valid construction if parameters change, 
        # though the task is specific.
        # Direct sum of tetracode is valid for q=3.
        if q == 3 and n >= 4:
            tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
            return [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]
        else:
            # Generic fallback: all words (valid but large)
            return [list(word) for word in itertools.product(range(q), repeat=n)]

    # Specific optimization for 3, 7, 1
    total_words = q ** n
    all_words = np.array(list(itertools.product(range(q), repeat=n)), dtype=np.uint8)
    
    # Precompute masks as boolean arrays: mask[i] is True for words within radius 1 of all_words[i]
    # We'll use a 2D boolean array of shape (total_words, total_words)
    # mask[i, j] = True if word j is within radius of word i
    
    # For radius 1 in q=3, n=7: each word covers 1 + 7*2 = 15 words
    # We can compute this efficiently by generating all neighbors
    
    # Build adjacency: for each word, list of words within distance 1
    # Use a more efficient approach: precompute the "neighbor list" for each word
    
    # Actually, let's use a different representation for speed:
    # For each word, we know its 15 neighbors (itself + 14 at distance 1)
    # We can store these as indices
    
    # Precompute all neighbor indices
    # For word i with digits d_0...d_6, neighbors are:
    # itself, and for each position p, change d_p to (d_p+1)%3 and (d_p+2)%3
    
    # Let's precompute a list of 15 indices for each word
    neighbor_lists = np.zeros((total_words, 15), dtype=np.int32)
    
    for i in range(total_words):
        word = all_words[i]
        neighbor_lists[i, 0] = i
        count = 1
        for pos in range(n):
            orig = word[pos]
            for new_val in range(q):
                if new_val == orig:
                    continue
                # Compute index of word with position pos changed to new_val
                # Use mixed-radix conversion
                idx = 0
                for k in range(n):
                    if k == pos:
                        idx = idx * q + new_val
                    else:
                        idx = idx * q + word[k]
                neighbor_lists[i, count] = idx
                count += 1
    
    # Now for local search, we need to check coverage.
    # Represent the set of chosen codewords as a boolean array of length total_words
    # covered[j] = True if word j is within radius 1 of some chosen codeword
    
    # Greedy initialization using vectorized operations
    covered = np.zeros(total_words, dtype=bool)
    C = []  # list of chosen codeword indices
    C_set = set()
    
    # Precompute the neighbor matrix M (total_words x total_words) boolean
    M = np.zeros((total_words, total_words), dtype=bool)
    for i in range(total_words):
        M[i, neighbor_lists[i]] = True
        
    # Convert M to int32 for matrix multiplication if needed, but boolean @ boolean works in numpy?
    # M @ vector (bool) -> int. Yes.
    
    # Greedy
    while not covered.all():
        uncovered = ~covered
        gains = M @ uncovered.astype(np.int32)
        # Among unchosen words, pick max gain
        # We need to exclude already chosen words
        # Create a mask of unchosen words
        # Using C_set is slow for large C, but C is small (< 300).
        # Alternatively, maintain a boolean array `chosen_mask`.
        if C:
            chosen_mask = np.zeros(total_words, dtype=bool)
            for c in C:
                chosen_mask[c] = True
            gains[chosen_mask] = -1
        best_idx = np.argmax(gains)
        if gains[best_idx] <= 0:
            break
        C.append(best_idx)
        C_set.add(best_idx)
        covered |= M[best_idx]
        
    # Local search with vectorized operations
    start_time = time.time()
    time_limit = 110.0  # Leave 40s margin for safety and evaluation overhead
    
    import random
    random.seed(42)
    
    improved = True
    while improved and (time.time() - start_time) < time_limit:
        improved = False
        
        # 1. Redundancy removal (vectorized)
        # Compute cover counts: cover_counts[w] = number of codewords in C covering w
        if C:
            # M[C] is (K, N). Sum along axis 0 gives (N,)
            cover_counts = M[C].sum(axis=0)
            
            # A codeword c is redundant if there is no word w such that M[c, w] is True AND cover_counts[w] == 1
            # unique_mask[c, w] = M[c, w] AND (cover_counts[w] == 1)
            # is_redundant[c] = NOT (ANY w such that unique_mask[c, w])
            
            # We can do this for all c in C at once?
            # M[C] is (K, N). cover_counts is (N,).
            # unique_mask = M[C] & (cover_counts[None, :] == 1)
            # is_redundant = ~unique_mask.any(axis=1)
            
            unique_mask = M[C] & (cover_counts[None, :] == 1)
            is_redundant = ~unique_mask.any(axis=1)
            
            if is_redundant.any():
                # Remove redundant codewords
                indices_to_remove = np.where(is_redundant)[0]
                for idx in indices_to_remove:
                    c = C[idx]
                    C.remove(c)
                    C_set.remove(c)
                improved = True
                continue
        
        # 2. 2-for-1 replacement
        # We use the fast rest_cover calculation:
        # rest_cover = (cover_counts - M[c1] - M[c2]) > 0
        
        if len(C) >= 2:
            # Recompute cover_counts if C changed
            if C:
                cover_counts = M[C].sum(axis=0)
            else:
                cover_counts = np.zeros(total_words, dtype=np.int32)
                
            C_list = list(C)
            K = len(C_list)
            
            # Sample pairs
            # To be fast, we limit the number of pairs checked per iteration
            # With K~200, K^2/2 = 20000 pairs.
            # Each pair check is O(N) numpy ops. 20000 * 2187 ~ 43M ops.
            # Numpy handles this in ~0.1-0.5s. So we can check ALL pairs!
            
            found_replacement = False
            for i in range(K):
                if (time.time() - start_time) > time_limit - 5:
                    break
                c1 = C_list[i]
                M_c1 = M[c1]
                for j in range(i + 1, K):
                    c2 = C_list[j]
                    M_c2 = M[c2]
                    
                    # Compute exclusive coverage of c1 and c2
                    # exclusive[w] = True if w is covered by c1 or c2, but NOT by any other c in C
                    # rest_cover = (cover_counts - M_c1 - M_c2) > 0
                    # Note: M_c1 and M_c2 are bool. cover_counts is int.
                    # We need to subtract 1 where M_c1 or M_c2 is True.
                    # Since M is 0/1, M_c1.astype(int) is 0/1.
                    
                    rest_cover = (cover_counts - M_c1.astype(np.int32) - M_c2.astype(np.int32)) > 0
                    exclusive = (M_c1 | M_c2) & ~rest_cover
                    
                    if not exclusive.any():
                        # Both are redundant relative to others (should have been caught by redundancy check,
                        # but maybe due to integer issues or if we skipped it? Unlikely.
                        # Actually, if exclusive is empty, it means every word covered by c1 or c2 is also covered by others.
                        # So we can remove both.
                        C.remove(c1)
                        C.remove(c2)
                        C_set.remove(c1)
                        C_set.remove(c2)
                        improved = True
                        found_replacement = True
                        break
                        
                    # Check if any c_new not in C covers exclusive
                    # We need M[c_new] to have True wherever exclusive has True.
                    # i.e., (M & exclusive[:, None]).all(axis=0) is not quite right.
                    # We need: for all w, if exclusive[w] then M[c_new, w].
                    # This is equivalent to: M[c_new] & exclusive == exclusive
                    # Or: (M[c_new] & ~exclusive).any() is False? No.
                    # (M[c_new] & exclusive).sum() == exclusive.sum()
                    
                    exclusive_size = exclusive.sum()
                    if exclusive_size == 0:
                        continue
                        
                    # Vectorized check for all c_new
                    # cover_counts_new = M @ exclusive.astype(np.int32)
                    # We want cover_counts_new == exclusive_size
                    # And c_new not in C_set
                    
                    # This is O(N^2) which is 2187^2 ~ 4.8M ops.
                    # Doing this for every pair (20000 pairs) is 96B ops. TOO SLOW.
                    
                    # Optimization:
                    # Only do the full search if exclusive_size is small?
                    # Or limit the number of pairs we check this way.
                    
                    # Let's only do the full search for the first few promising pairs.
                    # Or, we can break after finding one replacement.
                    
                    # To make this feasible, we will only attempt the replacement search
                    # for a limited number of pairs (e.g., 100) per local search iteration.
                    # If we find a replacement, we break.
                    
                    # Actually, the previous code sampled 5000 pairs.
                    # Let's sample 500 pairs to be safe on time.
                    
                    # We are in a nested loop. Let's restructure.
                    # Instead of nested loops, let's sample pairs.
                    
                    pass # Placeholder, logic below
                    
                if found_replacement:
                    break
            
            if not found_replacement:
                # Sample pairs for replacement search
                # To avoid O(K^2) overhead in Python, we pre-sample indices
                # But K is small (~200).
                
                # Let's try a fixed number of random pairs
                num_trials = 200
                for _ in range(num_trials):
                    if (time.time() - start_time) > time_limit - 5:
                        break
                    i, j = random.sample(range(K), 2)
                    c1 = C_list[i]
                    c2 = C_list[j]
                    
                    M_c1 = M[c1]
                    M_c2 = M[c2]
                    
                    rest_cover = (cover_counts - M_c1.astype(np.int32) - M_c2.astype(np.int32)) > 0
                    exclusive = (M_c1 | M_c2) & ~rest_cover
                    
                    if not exclusive.any():
                        C.remove(c1)
                        C.remove(c2)
                        C_set.remove(c1)
                        C_set.remove(c2)
                        improved = True
                        found_replacement = True
                        break
                        
                    exclusive_size = exclusive.sum()
                    if exclusive_size == 0:
                        continue
                        
                    # Check candidates
                    # To speed up, we can check if any word in exclusive is NOT covered by ANY candidate?
                    # No, we need a single candidate to cover ALL of exclusive.
                    
                    # Vectorized check
                    cover_counts_new = M @ exclusive.astype(np.int32)
                    candidate_mask = (cover_counts_new == exclusive_size)
                    
                    # Exclude current C
                    for c in C_set:
                        candidate_mask[c] = False
                        
                    candidates = np.where(candidate_mask)[0]
                    
                    if len(candidates) > 0:
                        c_new = candidates[0]
                        C.remove(c1)
                        C.remove(c2)
                        C_set.remove(c1)
                        C_set.remove(c2)
                        C.append(c_new)
                        C_set.add(c_new)
                        improved = True
                        found_replacement = True
                        break
                        
        # 3. Kick
        if not improved and (time.time() - start_time) < time_limit - 10:
            if len(C) > 0:
                c_kick = random.choice(C)
                C.remove(c_kick)
                C_set.remove(c_kick)
                
                # Recompute covered
                covered = np.zeros(total_words, dtype=bool)
                for c in C:
                    covered |= M[c]
                
                # Re-greedy the uncovered words
                while not covered.all():
                    if (time.time() - start_time) > time_limit - 5:
                        break
                    uncovered = ~covered
                    gains = M @ uncovered.astype(np.int32)
                    if C:
                        chosen_mask = np.zeros(total_words, dtype=bool)
                        for c in C:
                            chosen_mask[c] = True
                        gains[chosen_mask] = -1
                    best_idx = np.argmax(gains)
                    if gains[best_idx] <= 0:
                        break
                    C.append(best_idx)
                    C_set.add(best_idx)
                    covered |= M[best_idx]
                    
                improved = True
                continue
                
        if not improved:
            break
    
    # Convert indices back to words
    return [all_words[i].tolist() for i in C]
