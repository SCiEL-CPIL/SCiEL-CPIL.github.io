import time
import numpy as np
import itertools

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    # Precompute neighbors for the space {0..q-1}^n
    # Total words = q^n
    total_words = q ** n
    
    # Map word (tuple) to index
    # We will work with indices 0..total_words-1
    # Generate all words as tuples, then map to index
    # For speed, we can generate neighbors directly in index space if we have a fast conversion.
    
    # Efficient way:
    # Generate all words as a list of tuples
    words = list(itertools.product(range(q), repeat=n))
    word_to_idx = {w: i for i, w in enumerate(words)}
    
    # Precompute neighbors for each index
    # neighbors[i] is a list of indices
    neighbors = [[] for _ in range(total_words)]
    for i, w in enumerate(words):
        neighbors[i].append(i) # Distance 0
        for pos in range(n):
            orig_val = w[pos]
            for new_val in range(q):
                if new_val != orig_val:
                    new_w = list(w)
                    new_w[pos] = new_val
                    new_w = tuple(new_w)
                    neighbors[i].append(word_to_idx[new_w])
    
    # Convert neighbors to numpy arrays for vectorized updates
    # Max neighbors = 1 + n*(q-1)
    max_neighbors = 1 + n * (q - 1)
    # Pad neighbors to same length for array creation? Or use list of arrays.
    # List of arrays is fine.
    neighbor_arrays = [np.array(nei, dtype=np.int32) for nei in neighbors]
    
    # Initial Code: Direct Sum Construction
    # Tetracode: 9 words of length 4
    tetracode = [[a, b, (a + b) % q, (a + 2 * b) % q] for a in range(q) for b in range(q)]
    
    # If n < 4, handle differently, but task is n=8.
    # If n > 4, append all possible tails.
    if n == 4:
        initial_code_indices = [word_to_idx[tuple(h)] for h in tetracode]
    elif n > 4:
        tails = list(itertools.product(range(q), repeat=n - 4))
        initial_code_indices = []
        for h in tetracode:
            for tail in tails:
                full_word = tuple(h) + tail
                initial_code_indices.append(word_to_idx[full_word])
    else:
        # Fallback: Greedy or all words?
        initial_code_indices = list(range(total_words))

    # Ensure unique
    initial_code_indices = list(set(initial_code_indices))
    
    # State
    code_indices = set(initial_code_indices)
    coverage_counts = np.zeros(total_words, dtype=np.int32)
    
    # Initialize coverage
    for idx in code_indices:
        coverage_counts[neighbor_arrays[idx]] += 1
    
    # Phase 1: Remove Redundant
    # A codeword c is redundant if min(coverage_counts[neighbors[c]]) >= 2
    # We iterate until no changes
    changed = True
    while changed:
        if time.time() > deadline:
            break
        changed = False
        # Copy list to allow modification during iteration
        for c in list(code_indices):
            # Check redundancy
            # Get counts for all neighbors of c
            # If all are >= 2, then removing c leaves all neighbors with >= 1
            counts_for_c = coverage_counts[neighbor_arrays[c]]
            if np.min(counts_for_c) >= 2:
                # Remove c
                code_indices.remove(c)
                coverage_counts[neighbor_arrays[c]] -= 1
                changed = True
    
    # Phase 2: Local Search (Swaps)
    # Try to swap c with c' (neighbor of c or random)
    # If valid, accept.
    # Validity: All words covered ONLY by c must be covered by c' or others.
    # Words covered only by c: w in neighbors[c] such that coverage_counts[w] == 1.
    # If any such w is NOT in neighbors[c'], swap is invalid.
    
    # To speed up, we can precompute "exclusive coverage" sets?
    # Or just check on the fly.
    
    # Random neighbor selection
    # We will try to swap c with a random neighbor of c.
    # Also try random words in space if stuck?
    
    # Sort code_indices for deterministic iteration? Or randomize?
    # Randomization helps escape local minima.
    
    num_swaps_attempted = 0
    max_swaps = 50000 # Limit iterations
    
    while time.time() < deadline and num_swaps < max_swaps:
        num_swaps += 1
        
        if not code_indices:
            break
            
        # Pick random codeword
        c = list(code_indices)[np.random.randint(len(code_indices))]
        
        # Pick random neighbor for swap target
        # Or pick random word in space?
        # Neighbors are more likely to maintain coverage.
        # Let's try neighbors first.
        cand_neighbors = neighbor_arrays[c]
        c_prime = cand_neighbors[np.random.randint(len(cand_neighbors))]
        
        if c_prime == c:
            continue
            
        if c_prime in code_indices:
            continue
            
        # Check validity of swap (Code - {c} + {c_prime})
        # 1. Identify words covered ONLY by c
        #    These are w in neighbors[c] where coverage_counts[w] == 1
        #    Note: coverage_counts includes c.
        #    If coverage_counts[w] == 1, then only c covers w.
        #    If we remove c, w becomes uncovered UNLESS c_prime covers w.
        
        # Get mask of words covered only by c
        # This requires checking neighbors of c
        counts_c = coverage_counts[neighbor_arrays[c]]
        only_by_c_mask = (counts_c == 1)
        
        if np.any(only_by_c_mask):
            # There are words covered only by c
            # We need to ensure c_prime covers all of them
            # Get the indices of these words
            # neighbors_c_arr = neighbor_arrays[c]
            # only_by_c_indices = neighbors_c_arr[only_by_c_mask]
            
            # Check if c_prime covers all these
            # c_prime covers w if w is in neighbor_arrays[c_prime]
            # We can create a boolean array for c_prime's neighbors
            # But this is O(17) per check.
            
            # Efficient check:
            # Create a set of neighbors of c_prime?
            # Or use numpy isin?
            # np.isin(only_by_c_indices, neighbor_arrays[c_prime])
            
            # Let's just loop, it's small (max 17 elements)
            valid_swap = True
            for w in neighbor_arrays[c][only_by_c_mask]:
                if w not in neighbor_arrays[c_prime]:
                    valid_swap = False
                    break
            
            if not valid_swap:
                continue
        
        # Swap is valid
        # Update state
        code_indices.remove(c)
        code_indices.add(c_prime)
        
        coverage_counts[neighbor_arrays[c]] -= 1
        coverage_counts[neighbor_arrays[c_prime]] += 1
        
        # After swap, try to remove redundant codewords immediately
        # This can cascade
        # We can do a quick pass
        # For simplicity, let the main loop handle it, or do a small local cleanup
        # Let's do a small cleanup to keep code size small
        for c2 in list(code_indices):
            if np.min(coverage_counts[neighbor_arrays[c2]]) >= 2:
                code_indices.remove(c2)
                coverage_counts[neighbor_arrays[c2]] -= 1

    # Convert indices back to words
    result = [words[i] for i in code_indices]
    # Ensure distinct (set guarantees this)
    return result
