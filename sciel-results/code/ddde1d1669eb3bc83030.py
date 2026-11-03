import numpy as np
import itertools
import random
import time

def construct(q, n, radius):
    # Specific optimization for q=3, n=6, radius=1
    if q == 3 and n == 6 and radius == 1:
        return _solve_3_6_1()
    
    # Fallback for other parameters (keep original logic or simple greedy)
    all_words = list(itertools.product(range(q), repeat=n))
    uncovered = set(all_words)
    code = []
    while uncovered:
        best_word = None
        best_count = -1
        for w in all_words:
            count = sum(1 for u in uncovered if sum(a != b for a, b in zip(w, u)) <= radius)
            if count > best_count:
                best_count = count
                best_word = w
        code.append(list(best_word))
        uncovered = {u for u in uncovered if sum(a != b for a, b in zip(best_word, u)) > radius}
    return code

def _solve_3_6_1():
    q, n, radius = 3, 6, 1
    total_words = q ** n
    indices = list(range(total_words))
    
    def word_to_idx(word):
        idx = 0
        for i in range(n):
            idx = idx * q + word[i]
        return idx

    def idx_to_word(idx):
        word = [0] * n
        for i in range(n - 1, -1, -1):
            word[i] = idx % q
            idx //= q
        return word

    # Precompute neighbors
    neighborhoods = []
    for i in indices:
        w = idx_to_word(i)
        neigh = [i]
        for pos in range(n):
            for val in range(q):
                if val == w[pos]:
                    continue
                new_w = w.copy()
                new_w[pos] = val
                neigh.append(word_to_idx(new_w))
        neighborhoods.append(neigh)
    
    # Create symmetric adjacency matrix M
    # M[i, j] = 1 if i is in neighbors[j] (i.e., j covers i)
    M = np.zeros((total_words, total_words), dtype=np.int8)
    for i in range(total_words):
        for j in neighborhoods[i]:
            M[i, j] = 1
            
    # neighbors_set for fast lookup
    neighbors_set = [set(neigh) for neigh in neighborhoods]
    
    start_time = time.time()
    time_limit = 115 # Leave buffer
    
    random.seed(42)
    
    best_code_size = float('inf')
    best_code_vec = None
    
    # GRASP Construction
    def grasp_construct():
        code_vec = np.zeros(total_words, dtype=np.int8)
        coverage = np.zeros(total_words, dtype=np.int32)
        
        while np.min(coverage) == 0:
            uncovered_mask = (coverage == 0)
            if not np.any(uncovered_mask):
                break
                
            # Find best candidates
            # score[j] = number of uncovered words covered by j
            # score = M.T @ uncovered_mask (since M[i,j]=1 if j covers i)
            # Wait, M[i,j] = 1 if i in neighbors[j].
            # coverage[i] = sum_j M[i,j] code_vec[j].
            # So M @ code_vec = coverage.
            # We want sum_i uncovered[i] * M[i,j] for each j.
            # This is (uncovered_mask @ M) ? No.
            # uncovered_mask is (N,).
            # We want for each j: sum_i M[i,j] * uncovered_mask[i].
            # This is M.T @ uncovered_mask.
            
            scores = M.T @ uncovered_mask
            # Get top 5 candidates
            # argsort scores descending
            top_k = 5
            sorted_indices = np.argsort(scores)[::-1]
            candidates = sorted_indices[:top_k]
            
            # Randomly pick one from candidates
            # Ensure score > 0
            valid_candidates = [c for c in candidates if scores[c] > 0]
            if not valid_candidates:
                break
            c_in = random.choice(valid_candidates)
            
            code_vec[c_in] = 1
            coverage = M[:, c_in] # Add column c_in
            
        return code_vec, coverage

    def local_search(code_vec, coverage):
        current_size = np.sum(code_vec)
        no_improve = 0
        max_no_improve = 500
        
        while time.time() - start_time < time_limit and no_improve < max_no_improve:
            # Check for removable words
            # A word c is removable if min(coverage[neighbors[c]]) > 1
            # We can check this for all c in code
            
            removable = False
            code_indices = np.where(code_vec == 1)[0]
            
            # Shuffle to randomize
            random.shuffle(code_indices)
            
            for c in code_indices:
                # Check if c is removable
                # neighbors[c] is a list of indices
                # We need min(coverage[neighbors[c]]) > 1
                if np.min(coverage[neighbors[c]]) > 1:
                    code_vec[c] = 0
                    coverage -= M[:, c]
                    current_size -= 1
                    removable = True
                    break
            
            if removable:
                no_improve = 0
                continue
            
            # No removable words. Try swaps.
            # Pick a random c_out
            c_out = random.choice(code_indices)
            
            # Find unique words covered by c_out
            unique_words = [w for w in neighbors[c_out] if coverage[w] == 1]
            
            # If no unique words, c_out is redundant? 
            # If min(coverage[neighbors[c_out]]) == 1, there is at least one unique word.
            # So unique_words is not empty.
            
            # Find c_in not in code that covers all unique_words
            # To speed up, we can just iterate through all non-code words
            # But we can filter: c_in must be in intersection of neighborhoods of unique_words?
            # No, c_in must cover each unique word.
            # So c_in must be in neighbors[w] for all w in unique_words.
            
            # Let's just iterate through all words not in code
            # This is 729 iterations.
            # For each, check if it covers all unique_words.
            
            swap_made = False
            # Randomize order of potential c_in?
            # Or just iterate.
            
            for c_in in range(total_words):
                if code_vec[c_in] == 1:
                    continue
                    
                # Check if c_in covers all unique_words
                # Use neighbors_set for O(1) lookup
                covers_all = True
                for w in unique_words:
                    if w not in neighbors_set[c_in]:
                        covers_all = False
                        break
                        
                if not covers_all:
                    continue
                
                # Perform swap
                code_vec[c_out] = 0
                code_vec[c_in] = 1
                coverage -= M[:, c_out]
                coverage += M[:, c_in]
                
                # Check if this enables a removal
                removed = False
                # Check all code words for removability
                new_code_indices = np.where(code_vec == 1)[0]
                random.shuffle(new_code_indices)
                
                for c_rem in new_code_indices:
                    if np.min(coverage[neighbors[c_rem]]) > 1:
                        code_vec[c_rem] = 0
                        coverage -= M[:, c_rem]
                        current_size -= 1
                        removed = True
                        break
                        
                if removed:
                    swap_made = True
                    no_improve = 0
                    break
                else:
                    # Revert swap
                    code_vec[c_in] = 0
                    code_vec[c_out] = 1
                    coverage -= M[:, c_in]
                    coverage += M[:, c_out]
            
            if not swap_made:
                no_improve += 1
                # If stuck, maybe try a random perturbation?
                # Or just restart local search from a different GRASP solution?
                # For now, just stop this LS phase.
        
        return code_vec, coverage, current_size

    # Run GRASP + LS multiple times
    best_size = float('inf')
    best_final_vec = None
    
    while time.time() - start_time < time_limit:
        # Construct
        code_vec, coverage = grasp_construct()
        
        # Local Search
        code_vec, coverage, current_size = local_search(code_vec, coverage)
        
        if current_size < best_size:
            best_size = current_size
            best_final_vec = code_vec.copy()
            
        # If we hit the lower bound (71), we can stop early?
        # 71 is the lower bound.
        if best_size <= 71:
            break

    if best_final_vec is not None:
        code_indices = np.where(best_final_vec == 1)[0]
        return [idx_to_word(i) for i in code_indices]
    else:
        # Fallback
        return []
