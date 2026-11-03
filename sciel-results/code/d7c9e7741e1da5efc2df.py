import numpy as np
import time

def construct(q, n, radius):
    # 1. Precomputation
    total = q ** n
    cov_size = 1 + n * (q - 1)
    
    # Build neighbors matrix
    # neighbors[i] contains indices of words within distance radius from word i
    # For radius 1, this is i and all single-digit changes.
    neighbors = np.zeros((total, cov_size), dtype=np.int32)
    
    powers = [q ** (n - 1 - pos) for pos in range(n)]
    
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
                
    # 2. Local Search Initialization
    # Start with a random set of codewords.
    # A good initial size is slightly above the known optimum (486).
    # Let's start with 600.
    init_size = 600
    if init_size > total:
        init_size = total
        
    # We will maintain:
    # selected: list of selected indices
    # coverage_count: array of size total, count of how many selected codewords cover each word
    
    # Helper to update coverage counts
    def add_codeword(idx):
        coverage_count[neighbors[idx]] += 1
        
    def remove_codeword(idx):
        coverage_count[neighbors[idx]] -= 1
        
    # Initialize
    np.random.seed(42) # Fixed seed for reproducibility in this context, or random
    initial_indices = np.random.choice(total, init_size, replace=False)
    
    coverage_count = np.zeros(total, dtype=np.int32)
    for idx in initial_indices:
        add_codeword(idx)
        
    selected_set = set(initial_indices)
    
    start_time = time.time()
    time_limit = 110.0
    
    best_len = len(selected_set)
    best_selected = list(selected_set)
    
    # 3. Main Loop
    while time.time() - start_time < time_limit:
        # Check uncovered
        uncovered_mask = (coverage_count == 0)
        
        if not np.any(uncovered_mask):
            # All covered. Try to reduce.
            # Find removable codewords: those where all neighbors have count > 1
            if len(selected_set) == 0:
                break
                
            # Vectorized check for removability
            # Get neighbors of all selected
            selected_list = list(selected_set)
            sel_neighbors = neighbors[selected_list] # Shape (S, cov_size)
            # Get counts at these neighbors
            # coverage_count[sel_neighbors] gives counts for each neighbor of each selected
            counts_at_neighbors = coverage_count[sel_neighbors]
            min_counts = np.min(counts_at_neighbors, axis=1)
            
            removable_indices = np.where(min_counts > 1)[0]
            
            if len(removable_indices) > 0:
                # Remove one randomly from removable
                rem_idx_pos = removable_indices[np.random.randint(len(removable_indices))]
                rem_idx = selected_list[rem_idx_pos]
                
                remove_codeword(rem_idx)
                selected_set.remove(rem_idx)
                
                if len(selected_set) < best_len:
                    best_len = len(selected_set)
                    best_selected = list(selected_set)
                    
            else:
                # No removable codewords. Try a swap?
                # Or just break if we can't improve?
                # Let's try a random swap: remove one, add one.
                # This might uncover words, but we fix them next iteration.
                # Actually, if no removable, we are in a local minimum of size.
                # We can try to "shake" by removing a random one and adding the best one.
                if len(selected_set) > 10: # Avoid emptying
                    rem_idx = selected_list[np.random.randint(len(selected_list))]
                    remove_codeword(rem_idx)
                    selected_set.remove(rem_idx)
                    # The next iteration will add a codeword to fix uncovered words
        else:
            # Uncovered words exist. Find best codeword to add.
            # candidates: all indices not in selected_set
            # We can just check all indices and subtract those in selected_set later,
            # or use a mask.
            
            # For each candidate c, count how many uncovered words it covers.
            # uncovered_mask[neighbors[c]] sum
            # We can do this for all candidates at once?
            # neighbors is (total, cov_size).
            # uncovered_mask is (total,).
            # We want sum of uncovered_mask[neighbors[i]] for all i.
            # This is equivalent to:
            # counts = np.zeros(total, dtype=np.int32)
            # For each i, counts[i] = np.sum(uncovered_mask[neighbors[i]])
            # This is a gather-sum.
            # Using numpy:
            # uncovered_mask is bool.
            # We can do:
            # idx_array = neighbors # (total, cov_size)
            # vals = uncovered_mask[idx_array] # (total, cov_size)
            # counts = vals.sum(axis=1)
            
            idx_array = neighbors
            vals = uncovered_mask[idx_array]
            counts = vals.sum(axis=1)
            
            # We should not pick a codeword that is already selected?
            # If we pick a selected one, it does nothing (count doesn't change for coverage, 
            # but we don't want duplicates).
            # So set counts for selected indices to -1 or 0?
            # Actually, if it's selected, adding it again is invalid.
            # So we mask out selected ones.
            
            # Create a mask of selected
            # This is O(total) to create mask, which is fine.
            selected_mask = np.zeros(total, dtype=bool)
            for idx in selected_set:
                selected_mask[idx] = True
                
            # Set counts of selected to -1 so they are never picked
            counts[selected_mask] = -1
            
            best_idx = np.argmax(counts)
            
            if counts[best_idx] > 0:
                add_codeword(best_idx)
                selected_set.add(best_idx)
            else:
                # No candidate covers any uncovered word?
                # This implies the uncovered words are not coverable by any codeword?
                # Impossible, since every word covers itself.
                # Unless best_idx was masked?
                # If all counts are <= 0, it means no non-selected codeword covers any uncovered word.
                # But every uncovered word is covered by itself.
                # So if an uncovered word is not in selected_set, its count should be >= 1.
                # So this shouldn't happen unless selected_set contains all words.
                if len(selected_set) == total:
                    break
                # If we are stuck, break
                break

    # 4. Convert best_selected to words
    codewords = []
    for idx in best_selected:
        digits = [0] * n
        tmp = idx
        for pos in range(n - 1, -1, -1):
            digits[pos] = tmp % q
            tmp //= q
        codewords.append(digits)
        
    return codewords
