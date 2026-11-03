import numpy as np
import time

def construct(q, n, radius):
    total = q ** n
    cov_size = 1 + n * (q - 1)
    
    # Precompute neighbors
    powers = [q ** (n - 1 - pos) for pos in range(n)]
    neighbors = np.zeros((total, cov_size), dtype=np.int32)
    
    for i in range(total):
        digits = [0] * n
        tmp = i
        for pos in range(n - 1, -1, -1):
            digits[pos] = tmp % q
            tmp //= q
            
        k = 0
        neighbors[i, k] = i
        k += 1
        for pos in range(n):
            orig_val = digits[pos]
            for new_val in range(q):
                if new_val == orig_val:
                    continue
                new_idx = i + (new_val - orig_val) * powers[pos]
                neighbors[i, k] = new_idx
                k += 1

    # Initialize state: Start with all words
    in_set = np.ones(total, dtype=bool)
    cover_counts = np.full(total, cov_size, dtype=np.int32)
    uncovered_count = 0
    set_size = total
    
    start_time = time.time()
    time_limit = 110.0
    
    T = 10.0
    T_min = 0.01
    cooling = 0.9995
    M = 100.0 # Penalty for uncovered
    
    best_set_size = set_size
    best_in_set = in_set.copy()
    
    # Helper to update cover counts and uncovered count
    # Returns delta_uncovered
    def apply_change(indices, sign):
        # sign: 1 for add, -1 for remove
        delta_uncov = 0
        for idx in indices:
            cover_counts[idx] += sign
            if sign == -1 and cover_counts[idx] == 0:
                delta_uncov += 1
            elif sign == 1 and cover_counts[idx] == 1:
                delta_uncov -= 1
        return delta_uncov

    def revert_change(indices, sign):
        # sign: 1 for add, -1 for remove
        # Revert means applying -sign
        delta_uncov = 0
        for idx in indices:
            cover_counts[idx] -= sign
            if sign == -1 and cover_counts[idx] == 1: # was 0, now 1
                delta_uncov -= 1
            elif sign == 1 and cover_counts[idx] == 0: # was 1, now 0
                delta_uncov += 1
        return delta_uncov

    while time.time() - start_time < time_limit:
        move_type = np.random.rand()
        
        if move_type < 0.8: # Remove or Add
            i = np.random.randint(total)
            if in_set[i]:
                # Remove i
                old_uncov = uncovered_count
                neigh_i = neighbors[i]
                delta_uncov = apply_change(neigh_i, -1)
                new_uncov = uncovered_count + delta_uncov
                # Wait, apply_change updates cover_counts. 
                # I need to calculate delta_cost based on new state.
                # Let's just use the updated cover_counts.
                
                delta_size = -1
                delta_uncov_val = new_uncov - old_uncov
                delta_cost = delta_size + M * delta_uncov_val
                
                accept = False
                if delta_cost < 0:
                    accept = True
                else:
                    if T > T_min:
                        prob = np.exp(-delta_cost / T)
                        if np.random.rand() < prob:
                            accept = True
                
                if accept:
                    in_set[i] = False
                    set_size += delta_size
                    uncovered_count = new_uncov
                    if set_size < best_set_size and uncovered_count == 0:
                        best_set_size = set_size
                        best_in_set = in_set.copy()
                else:
                    # Revert
                    revert_change(neigh_i, -1)
                    uncovered_count = old_uncov
            else:
                # Add i
                old_uncov = uncovered_count
                neigh_i = neighbors[i]
                delta_uncov = apply_change(neigh_i, 1)
                new_uncov = uncovered_count + delta_uncov
                
                delta_size = 1
                delta_uncov_val = new_uncov - old_uncov
                delta_cost = delta_size + M * delta_uncov_val
                
                accept = False
                if delta_cost < 0:
                    accept = True
                else:
                    if T > T_min:
                        prob = np.exp(-delta_cost / T)
                        if np.random.rand() < prob:
                            accept = True
                
                if accept:
                    in_set[i] = True
                    set_size += delta_size
                    uncovered_count = new_uncov
                    if set_size < best_set_size and uncovered_count == 0:
                        best_set_size = set_size
                        best_in_set = in_set.copy()
                else:
                    # Revert
                    revert_change(neigh_i, 1)
                    uncovered_count = old_uncov
                    
        else: # Swap
            # Pick i in set, j not in set
            i = np.random.randint(total)
            if in_set[i]:
                # Find j not in set
                while True:
                    j = np.random.randint(total)
                    if not in_set[j]:
                        break
            
            old_uncov = uncovered_count
            neigh_i = neighbors[i]
            neigh_j = neighbors[j]
            
            # Remove i
            apply_change(neigh_i, -1)
            # Add j
            apply_change(neigh_j, 1)
            
            new_uncov = uncovered_count
            
            delta_size = 0
            delta_uncov_val = new_uncov - old_uncov
            delta_cost = delta_size + M * delta_uncov_val
            
            accept = False
            if delta_cost < 0:
                accept = True
            else:
                if T > T_min:
                    prob = np.exp(-delta_cost / T)
                    if np.random.rand() < prob:
                        accept = True
            
            if accept:
                in_set[i] = False
                in_set[j] = True
                uncovered_count = new_uncov
                if set_size < best_set_size and uncovered_count == 0:
                    best_set_size = set_size
                    best_in_set = in_set.copy()
            else:
                # Revert
                revert_change(neigh_i, -1)
                revert_change(neigh_j, 1)
                uncovered_count = old_uncov
                
        T *= cooling
        
    # Repair if necessary
    if uncovered_count > 0 or best_in_set is None:
        # Use best_in_set if it exists, else current
        if best_in_set is not None:
            final_set = best_in_set
        else:
            final_set = in_set
        # Simple greedy repair
        test_cover = np.zeros(total, dtype=np.int32)
        for idx in np.where(final_set)[0]:
            test_cover[neighbors[idx]] += 1
            
        uncovered_indices = np.where(test_cover == 0)[0]
        for u in uncovered_indices:
            best_w = -1
            best_gain = -1
            for w in neighbors[u]:
                if final_set[w]:
                    continue
                gain = 0
                for v in neighbors[w]:
                    if test_cover[v] == 0:
                        gain += 1
                if gain > best_gain:
                    best_gain = gain
                    best_w = w
            if best_w != -1:
                final_set[best_w] = True
                for v in neighbors[best_w]:
                    test_cover[v] += 1
        best_in_set = final_set

    codewords = []
    for idx in np.where(best_in_set)[0]:
        digits = [0] * n
        tmp = idx
        for pos in range(n - 1, -1, -1):
            digits[pos] = tmp % q
            tmp //= q
        codewords.append(digits)
        
    return codewords
