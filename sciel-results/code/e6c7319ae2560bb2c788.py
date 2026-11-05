import itertools
import random
import time
import numpy as np

def construct(q, n, radius):
    start_time = time.time()
    time_limit = 110
    
    # Generate all words
    all_words = list(itertools.product(range(q), repeat=n))
    N = len(all_words)
    
    # Map word to index
    word_to_idx = {w: i for i, w in enumerate(all_words)}
    
    # Precompute neighbor indices
    # neighbors[i] is a list of indices
    neighbors = []
    for i, w in enumerate(all_words):
        nb = [i]
        for j in range(n):
            orig = w[j]
            for val in range(q):
                if val == orig:
                    continue
                new_w = list(w)
                new_w[j] = val
                nb.append(word_to_idx[tuple(new_w)])
        neighbors.append(np.array(nb, dtype=np.int32))
        
    # Convert neighbors to a list of arrays for fast indexing
    # Actually, we can create a single matrix if lengths are equal? 
    # Length is always 1 + n*(q-1) = 1 + 8*2 = 17.
    # So we can have a (N, 17) array.
    neigh_matrix = np.array(neighbors, dtype=np.int32)
    
    best_code = None
    best_size = float('inf')
    
    # Greedy using NumPy
    def greedy_numpy():
        uncovered = np.ones(N, dtype=bool)
        code_indices = []
        
        while np.any(uncovered):
            # For each candidate i, count how many of its neighbors are in uncovered
            # uncovered[neigh_matrix] gives a (N, 17) boolean matrix
            # Sum along axis 1 gives counts
            # This allocates a large array (6561, 17) which is ~111k booleans. Fast.
            counts = np.sum(uncovered[neigh_matrix], axis=1)
            
            # Find the index with max count
            # Tie-breaking: random or first?
            # To allow randomization, we can add small random noise to counts?
            # Or just pick the first max.
            # Let's pick the first max for determinism in one run, but shuffle the search order in outer loop?
            # No, the "first max" depends on the index order.
            # To randomize, we can shuffle the indices we check? 
            # Actually, `np.argmax` returns the first occurrence.
            # To get a random tie-breaker, we can add random noise to counts where counts are equal to max.
            # Simpler: Just run multiple times with different random initializations of the "uncovered" set? No.
            # Let's just use `np.argmax`. The structure of the problem is symmetric, so the "first" max might always be the same.
            # We need to break symmetry.
            # We can randomly permute the rows of neigh_matrix and all_words?
            # That's expensive.
            
            # Alternative: Add small random noise to all counts to break ties randomly.
            noise = np.random.uniform(0, 1e-6, size=N)
            counts_noisy = counts + noise
            best_idx = np.argmax(counts_noisy)
            
            code_indices.append(best_idx)
            uncovered[neigh_matrix[best_idx]] = False
            
        return code_indices

    def prune_numpy(code_indices):
        code_set = set(code_indices)
        # Build coverage counts
        # cover_count[w] = number of codewords covering w
        cover_count = np.zeros(N, dtype=np.int32)
        for ci in code_indices:
            cover_count[neigh_matrix[ci]] += 1
            
        # A codeword ci can be removed if for all w in neighbors(ci), cover_count[w] > 1
        # i.e., min(cover_count[neighbors(ci)]) > 1
        
        keep = [True] * len(code_indices)
        
        # Iterative removal
        changed = True
        while changed:
            changed = False
            new_code_indices = []
            # Recalculate cover_count for current new_code_indices? 
            # This is O(K * 17) per pass.
            
            # Rebuild cover_count
            cover_count = np.zeros(N, dtype=np.int32)
            for ci in code_indices:
                cover_count[neigh_matrix[ci]] += 1
                
            for ci in code_indices:
                min_cov = np.min(cover_count[neigh_matrix[ci]])
                if min_cov > 1:
                    changed = True
                else:
                    new_code_indices.append(ci)
                    
            if changed:
                code_indices = new_code_indices
            else:
                break
                
        return code_indices

    # Run multiple times
    # To get different results, we can randomize the noise or the initial state?
    # The noise in greedy is the main randomization.
    
    for run in range(200):
        if time.time() - start_time > time_limit:
            break
            
        code_indices = greedy_numpy()
        code_indices = prune_numpy(code_indices)
        
        if len(code_indices) < best_size:
            best_size = len(code_indices)
            best_code = code_indices
            if best_size <= 486:
                # We can try to continue for a bit to see if we can go lower, 
                # but 486 is the best known.
                pass

    if best_code is None:
        # Fallback
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        return [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]
        
    # Convert indices back to words
    return [list(all_words[i]) for i in best_code]
