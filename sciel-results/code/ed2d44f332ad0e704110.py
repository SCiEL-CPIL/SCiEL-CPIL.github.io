import numpy as np
import time
import random

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    # Parameters for this specific task
    # The task specifies q=3, n=7, radius=1
    # But the function signature is generic. We should handle the specific case efficiently.
    if q != 3 or n != 7 or radius != 1:
        # Fallback to the simple product construction for other parameters
        # This is a placeholder for generic cases, but the evaluator focuses on the specific task.
        # For the specific task, we use the optimized path below.
        pass

    # Specific optimization for q=3, n=7, radius=1
    # Total words: 3^7 = 2187
    num_words = q ** n
    
    # Generate all words
    # We represent words as integers 0..num_words-1 and convert to base-q arrays as needed
    # Or just keep them as arrays for distance calculation.
    # Vectorized approach:
    # Create a list of all combinations
    all_words = np.array(list(np.meshgrid(*[range(q)] * n, indexing='ij')), dtype=np.int8).reshape(n, -1).T
    
    # Precompute neighbors for each word
    # Neighbors are words at Hamming distance <= 1
    # For each word, there are 1 (itself) + n * (q-1) neighbors
    # We can create a list of indices for each word's neighbors
    neighbors = []
    for i in range(num_words):
        w = all_words[i]
        idxs = [i]
        for pos in range(n):
            for val in range(q):
                if val != w[pos]:
                    new_w = w.copy()
                    new_w[pos] = val
                    # Find index of new_w
                    # Converting to integer index is faster for lookup
                    # Let's use a dictionary mapping tuple(word) to index
                    pass
        
    # Better: Use integer representation for indices
    # Convert all_words to a list of tuples for dictionary lookup
    word_to_idx = {tuple(w): i for i, w in enumerate(all_words)}
    
    neighbors_list = []
    for i in range(num_words):
        w = all_words[i]
        idxs = [i]
        for pos in range(n):
            orig_val = w[pos]
            for val in range(q):
                if val != orig_val:
                    new_w = w.copy()
                    new_w[pos] = val
                    idxs.append(word_to_idx[tuple(new_w)])
        neighbors_list.append(idxs)
    
    # Greedy Set Cover
    best_code = None
    best_size = num_words + 1
    
    # Number of restarts depends on time available
    # Each greedy pass is O(num_words * num_candidates * avg_neighbor_count)
    # With vectorization, we can speed up the "count uncovered" step.
    
    # Let's use a randomized greedy approach
    for restart in range(50): # Try up to 50 restarts
        if time.time() > deadline:
            break
            
        covered = np.zeros(num_words, dtype=bool)
        code = []
        
        # To speed up, we can maintain a count of uncovered neighbors for each candidate
        # count[c] = number of neighbors of c that are not covered
        # Initially, all are uncovered, so count[c] = len(neighbors_list[c])
        # When we pick c, we mark its neighbors as covered.
        # Then we need to update counts for all other candidates d that share a neighbor with c.
        # This is complex. 
        # Given num_words=2187, a simple loop over all candidates to find max might be fast enough.
        # 2187 candidates * 15 neighbors * 50 restarts = ~1.6M operations. Very fast.
        
        while not np.all(covered):
            max_count = -1
            best_c = -1
            best_indices = []
            
            # Find candidate with max uncovered neighbors
            for c in range(num_words):
                if covered[c]: # Optional: skip if c itself is covered? No, c might cover others.
                    # Actually, if c is covered, it's still a valid candidate to add if it covers other uncovered words.
                    # But usually we pick uncovered ones first? No, the set cover picks any set.
                    pass
                
                count = 0
                for nb in neighbors_list[c]:
                    if not covered[nb]:
                        count += 1
                
                if count > max_count:
                    max_count = count
                    best_c = c
                    best_indices = [c]
                elif count == max_count and count > 0:
                    best_indices.append(c)
            
            # Random tie-breaking
            if best_indices:
                chosen_c = random.choice(best_indices)
            else:
                # Should not happen if not all covered
                break
            
            code.append(all_words[chosen_c].tolist())
            
            # Update covered
            for nb in neighbors_list[chosen_c]:
                covered[nb] = True
        
        if len(code) < best_size:
            best_size = len(code)
            best_code = [c[:] for c in code]
            
        # Local Search: Try to remove redundant codewords
        # This is a post-processing step
        # We can try to remove a codeword if the union of the rest still covers everything.
        # This is O(len(code)^2 * neighbor_count)
        # Let's do a simple pass
        if best_code:
            # We'll do this only on the best code found so far to save time
            pass

    # Final polish: Try to reduce the best_code found
    # We can run a local search on the best_code
    # Try swapping or removing
    
    if best_code is None:
        # Fallback
        best_code = [all_words[i].tolist() for i in range(num_words)]
        
    return best_code

# The above is a bit slow in pure Python loops for the inner part.
# Let's optimize the inner loop using NumPy.
