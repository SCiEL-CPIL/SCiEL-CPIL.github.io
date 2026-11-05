import numpy as np
import time

def construct(q, n, radius):
    start_time = time.time()
    deadline = 115.0 # 5s safety margin
    
    total_words = q ** n
    
    # Precompute index <-> word conversions
    # For speed, we can precompute the list of words
    words = []
    for i in range(total_words):
        w = [0] * n
        x = i
        for k in range(n - 1, -1, -1):
            w[k] = x % q
            x //= q
        words.append(w)
        
    # Precompute sphere indices
    # sphere_indices[i] is a list of indices of words within distance 1 from words[i]
    sphere_indices = []
    for i in range(total_words):
        w = words[i]
        neighbors = [i]
        for pos in range(n):
            orig_val = w[pos]
            for val in range(q):
                if val == orig_val:
                    continue
                w[pos] = val
                # Calculate index
                idx = 0
                for k in range(n):
                    idx = idx * q + w[k]
                neighbors.append(idx)
            w[pos] = orig_val
        sphere_indices.append(np.array(neighbors, dtype=np.int32))
        
    sphere_indices = np.array(sphere_indices, dtype=np.int32)
    
    covered = np.zeros(total_words, dtype=bool)
    code_indices = []
    
    # Greedy Phase
    while not np.all(covered):
        if time.time() > deadline:
            break
        
        # Count uncovered neighbors for each candidate
        # covered[sphere_indices] is shape (total_words, sphere_size)
        # We want sum of ~covered
        uncovered_counts = np.sum(~covered[sphere_indices], axis=1)
        
        best_idx = np.argmax(uncovered_counts)
        
        if uncovered_counts[best_idx] == 0:
            break # Should not happen
            
        code_indices.append(best_idx)
        covered[sphere_indices[best_idx]] = True
        
    # Local Search Phase: Try to remove codewords
    current_code = code_indices[:]
    improved = True
    while improved:
        improved = False
        if time.time() > deadline:
            break
        for i in range(len(current_code)):
            if time.time() > deadline:
                break
            # Check if removing current_code[i] still covers all
            # We can do this by checking if there is any word covered ONLY by current_code[i]
            # But simpler: recompute coverage.
            # Optimization: Only check words in the sphere of the removed codeword?
            # No, other codewords might cover them.
            # Recomputing full coverage is ~3000 ops.
            
            # Efficient check:
            # For each word in the sphere of the removed codeword, check if it is covered by any OTHER codeword in the current set.
            # This is O(sphere_size * (len(current_code) * sphere_size)) which is 15 * 190 * 15 ~ 42750.
            # Recomputing full coverage is 190 * 15 = 2850.
            # So full recompute is faster.
            
            new_code = current_code[:i] + current_code[i+1:]
            
            if not new_code:
                continue
                
            temp_covered = np.zeros(total_words, dtype=bool)
            for idx in new_code:
                temp_covered[sphere_indices[idx]] = True
                
            if not np.all(temp_covered):
                continue # Cannot remove
                
            current_code = new_code
            improved = True
            break # Restart loop
            
    # Convert indices back to words
    result = [words[idx] for idx in current_code]
    
    return result
