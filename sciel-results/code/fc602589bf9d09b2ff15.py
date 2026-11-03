import numpy as np
import random
import time
import itertools

def construct(q, n, radius):
    # Precompute all words
    all_words = np.array(list(itertools.product(range(q), repeat=n)), dtype=np.int8)
    N = len(all_words)
    
    # Precompute coverage mask
    # cover_mask[i, j] = 1 if dist(all_words[i], all_words[j]) <= radius
    # Using broadcasting for Hamming distance
    # dist_mat[i, j] = sum_k (all_words[i, k] != all_words[j, k])
    
    # To save memory and time, we can compute the mask directly
    # M[i, j] = 1 if dist <= radius
    
    # Efficient way:
    # diff = all_words[:, :, np.newaxis] - all_words[:, np.newaxis, :] # (N, N, n) - too big?
    # 2187^2 * 7 * 1 byte = 33MB. OK.
    # But let's stick to the loop over n which is safer for memory in some environments, 
    # though 33MB is trivial.
    
    # Let's use the loop method to be safe with memory spikes if N were larger, 
    # but for N=2187, the vectorized diff is faster.
    
    # Let's try the vectorized diff approach for speed.
    # all_words is (N, n).
    # a = all_words[:, :, np.newaxis] # (N, n, 1)
    # b = all_words[:, np.newaxis, :] # (N, 1, n)
    # This creates (N, n, N) which is N^2*n.
    # dist = np.sum(a != b, axis=1) # (N, N)
    
    # This is the fastest way.
    a = all_words[:, :, np.newaxis]
    b = all_words[:, np.newaxis, :]
    dist_mat = np.sum(a != b, axis=1)
    cover_mask = (dist_mat <= radius).astype(np.int8)
    
    best_size = N + 1
    best_solution = None
    start_time = time.time()
    time_limit = 115.0
    
    # Helper to convert solution indices to list of lists
    def indices_to_solution(indices):
        return [all_words[i].tolist() for i in indices]

    while time.time() - start_time < time_limit:
        # 1. Greedy Phase
        uncovered = np.ones(N, dtype=bool)
        solution = []
        
        while np.any(uncovered):
            # cov_counts[i] = number of uncovered words covered by i
            cov_counts = cover_mask @ uncovered
            
            # Find max
            max_cov = np.max(cov_counts)
            if max_cov == 0:
                # Should not happen if uncovered is not empty
                break
                
            candidates = np.where(cov_counts == max_cov)[0]
            chosen = random.choice(candidates)
            solution.append(chosen)
            
            # Update uncovered
            uncovered[cover_mask[chosen] == 1] = False
            
        current_size = len(solution)
        
        # 2. Local Search Phase
        # Maintain cover counts
        cover_count = np.zeros(N, dtype=np.int8)
        for idx in solution:
            cover_count += cover_mask[idx]
            
        improved = True
        while improved:
            improved = False
            
            # A. Try removals
            # Check if any word in solution is redundant
            # Redundant if min(cover_count[neighbors]) > 1
            
            # Vectorized check for all words in solution?
            # It's easier to loop through solution indices
            # Optimization: Only check words that are "likely" redundant?
            # For now, loop.
            
            for i in range(len(solution)):
                idx = solution[i]
                neighbors = np.where(cover_mask[idx] == 1)[0]
                if len(neighbors) == 0:
                    continue
                min_cover = np.min(cover_count[neighbors])
                if min_cover > 1:
                    # Remove
                    cover_count -= cover_mask[idx]
                    solution.pop(i)
                    improved = True
                    break # Restart removal loop
            
            if improved:
                continue
                
            # B. Try swaps
            # If no removals, try to swap to escape local optimum
            # We try a limited number of random swaps per local search pass
            
            # Prepare in_solution mask
            in_solution = np.zeros(N, dtype=bool)
            in_solution[solution] = True
            
            swap_attempts = 0
            max_swap_attempts = 50 # Limit to prevent stalling
            
            while swap_attempts < max_swap_attempts:
                swap_attempts += 1
                if not solution:
                    break
                
                # Pick a random index from solution
                i = random.randint(0, len(solution) - 1)
                idx = solution[i]
                
                # Find words that are ONLY covered by idx
                # These are j such that cover_count[j] == 1 and cover_mask[idx, j] == 1
                # mask_only = (cover_count == 1) & (cover_mask[idx] == 1)
                # required_words = np.where(mask_only)[0]
                
                # Calculate required words
                # To avoid creating large temp arrays, we can do:
                # single_covered = (cover_count == 1)
                # covered_by_idx = (cover_mask[idx] == 1)
                # required = single_covered & covered_by_idx
                
                # But we can optimize:
                # If idx covers no unique words, any swap is valid (coverage wise)
                # Let's find the indices of words covered by idx
                idx_neighbors = np.where(cover_mask[idx] == 1)[0]
                
                # Among these, which have cover_count == 1?
                # We need to check cover_count[idx_neighbors] == 1
                unique_mask = cover_count[idx_neighbors] == 1
                required_words = idx_neighbors[unique_mask]
                
                if len(required_words) == 0:
                    # Any word not in solution can replace idx
                    # Pick a random one to diversify
                    candidates = np.where(~in_solution)[0]
                    if len(candidates) > 0:
                        new_idx = random.choice(candidates)
                        # Perform swap
                        cover_count -= cover_mask[idx]
                        cover_count += cover_mask[new_idx]
                        solution[i] = new_idx
                        in_solution[idx] = False
                        in_solution[new_idx] = True
                        improved = True
                        break # Restart local search loop
                else:
                    # We need a new_idx that covers ALL required_words
                    # And new_idx is not in solution.
                    
                    # Check candidates
                    # valid[c] = 1 if cover_mask[c, required_words] are all 1
                    # sums = np.sum(cover_mask[:, required_words], axis=1)
                    # valid = (sums == len(required_words)) & (~in_solution)
                    
                    # Optimization: If required_words is large, skip
                    if len(required_words) > 15:
                        continue
                    
                    sums = np.sum(cover_mask[:, required_words], axis=1)
                    valid = (sums == len(required_words)) & (~in_solution)
                    candidates = np.where(valid)[0]
                    
                    if len(candidates) > 0:
                        new_idx = random.choice(candidates)
                        cover_count -= cover_mask[idx]
                        cover_count += cover_mask[new_idx]
                        solution[i] = new_idx
                        in_solution[idx] = False
                        in_solution[new_idx] = True
                        improved = True
                        break # Restart local search loop
        
        # End Local Search
        
        if len(solution) < best_size:
            best_size = len(solution)
            best_solution = indices_to_solution(solution)
            
    if best_solution is None:
        # Fallback: Return all words (invalid score but valid structure)
        # Or a simple greedy
        return indices_to_solution(range(N))
        
    return best_solution
