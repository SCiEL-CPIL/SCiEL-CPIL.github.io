import time
import random
import numpy as np

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before the 150s kill
    
    total_space = q ** n
    
    # Precompute word to index and index to word
    def word_to_idx(w):
        idx = 0
        for v in w:
            idx = idx * q + v
        return idx

    def idx_to_word(idx):
        w = []
        for i in range(n):
            w.append(idx % q)
            idx //= q
        w.reverse()
        return w

    # Precompute all words
    all_words = [idx_to_word(i) for i in range(total_space)]
    
    # Build cover matrix
    cover_matrix = np.zeros((total_space, total_space), dtype=bool)
    np.fill_diagonal(cover_matrix, True)
    
    if radius >= 1:
        for i in range(total_space):
            w = all_words[i]
            for pos in range(n):
                orig_val = w[pos]
                for val in range(q):
                    if val == orig_val:
                        continue
                    power = q ** (n - 1 - pos)
                    j = i + (val - orig_val) * power
                    cover_matrix[i, j] = True

    # Precompute int8 version of cover matrix for fast dot products
    cm_int8 = cover_matrix.astype(np.int8)
    
    # Precompute neighbor lists for each word (words within radius)
    # neighbors[i] is a list of indices j such that cover_matrix[i, j] is True
    # This is used for fast gain calculation and swap checks
    neighbors = [np.where(cover_matrix[i])[0] for i in range(total_space)]
    
    # Also precompute the count of neighbors for each word
    neighbor_counts = np.array([len(nb) for nb in neighbors], dtype=np.int32)
    
    def greedy_cover_randomized():
        uncovered = np.ones(total_space, dtype=np.int8)
        code = []
        
        while True:
            if time.time() > deadline - 5.0:
                break
                
            # Calculate gain for each codeword
            gains = cm_int8.dot(uncovered)
            
            max_gain = np.max(gains)
            if max_gain == 0:
                break
                
            # Find all indices with max gain
            candidates = np.where(gains == max_gain)[0]
            
            # Better tie-breaking: prefer codewords that cover more words with low current coverage
            # This helps balance the load and may lead to better final codes
            if len(candidates) > 1:
                # Score each candidate by how many of its neighbors have cover_count == 1
                # We don't have cover_count here, so use a simpler heuristic:
                # Prefer candidates whose neighbors are less "contested"
                # Approximate: prefer candidates with fewer neighbors that are also high-gain
                high_gain_mask = gains >= max_gain - 1  # near-max gain
                contest_scores = cm_int8[candidates].dot(high_gain_mask.astype(np.int8))
                # Prefer lower contest score (less overlap with other high-gain candidates)
                min_contest = np.min(contest_scores)
                best_candidates = candidates[contest_scores == min_contest]
                best_idx = best_candidates[random.randint(0, len(best_candidates)-1)]
            else:
                best_idx = candidates[0]
            
            code.append(int(best_idx))
            uncovered[cover_matrix[best_idx]] = 0
            
        return code

    def local_search_reduce(code, max_time=None):
        if not code:
            return code
        
        if max_time is None:
            max_time = deadline - time.time()
        
        current_code = code[:]
        
        # cover_count[w] = number of codewords covering word w
        cover_count = np.zeros(total_space, dtype=np.int32)
        
        # Update cover_count
        for idx in current_code:
            cover_count += cover_matrix[idx]
        
        improved = True
        while improved:
            improved = False
            random.shuffle(current_code)
            n_cw = len(current_code)
            
            for i in range(n_cw):
                if time.time() > deadline - 2.0:
                    return current_code
                
                idx = current_code[i]
                mask = cover_matrix[idx]
                
                # Check if removable: all covered words have count > 1
                min_count = np.min(cover_count[mask])
                
                if min_count > 1:
                    # Remove
                    cover_count -= mask
                    current_code.pop(i)
                    improved = True
                    break
                else:
                    # Try swap
                    unique_mask = mask & (cover_count == 1)
                    
                    if np.any(unique_mask):
                        unique_indices = np.where(unique_mask)[0]
                        
                        if len(unique_indices) > 0:
                            # Create exclusion mask
                            exclude_mask = np.zeros(total_space, dtype=bool)
                            for j in current_code:
                                exclude_mask[j] = True
                        
                            # Find candidates that cover all unique words
                            # Use the precomputed neighbors for efficiency
                            # A candidate j must be in neighbors[u] for all u in unique_indices
                            # Start with neighbors of the first unique word, intersect with others
                            candidate_set = set(neighbors[unique_indices[0]].tolist())
                            for u in unique_indices[1:]:
                                candidate_set &= set(neighbors[u].tolist())
                                if not candidate_set:
                                    break
                            
                            if candidate_set:
                                # Remove current code words
                                candidate_set -= set(current_code)
                                
                                if candidate_set:
                                    # Score candidates by how many words they cover that are currently under-covered
                                    # Prefer candidates that cover more words with count == 1 or 2
                                    candidates_list = list(candidate_set)
                                    # Sort by coverage of low-count words
                                    def score_candidate(j):
                                        j_mask = cover_matrix[j]
                                        # Count words with count 1 or 2 that this candidate covers
                                        return np.sum(cover_count[j_mask] <= 2)
                                    
                                    # For large candidate sets, just pick randomly from top scorers
                                    if len(candidates_list) > 50:
                                        # Sample a subset for scoring
                                        sample = random.sample(candidates_list, min(50, len(candidates_list)))
                                    else:
                                        sample = candidates_list
                                    
                                    scores = [score_candidate(j) for j in sample]
                                    max_score = max(scores)
                                    best_candidates = [j for j, s in zip(sample, scores) if s == max_score]
                                    new_idx = random.choice(best_candidates)
                                    
                                    # Update cover_count
                                    cover_count -= mask
                                    cover_count += cover_matrix[new_idx]
                                    
                                    current_code[i] = int(new_idx)
                                    improved = True
                                    break
                            
        return current_code

    # Initial greedy construction
    best_code = greedy_cover_randomized()
    best_code = local_search_reduce(best_code)
    
    # Run restarts if time permits
    # We use a while loop with a time check to maximize the number of restarts
    # Track best seen to allow early termination of local search if we hit a good solution
    best_len = len(best_code)
    
    while time.time() < deadline - 10.0:
        cand_code = greedy_cover_randomized()
        cand_code = local_search_reduce(cand_code)
        if len(cand_code) < best_len:
            best_code = cand_code
            best_len = len(cand_code)
    
    # Final polish: run local search on best_code with remaining time
    # This might find further reductions if the previous local search stopped early due to time
    best_code = local_search_reduce(best_code)
    
    # Convert indices back to words
    return [all_words[i] for i in best_code]
