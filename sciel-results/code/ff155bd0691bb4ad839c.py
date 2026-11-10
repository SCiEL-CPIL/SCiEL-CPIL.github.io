"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import itertools
import time
import numpy as np

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 110.0 # Safety margin for 120s limit
    
    # Generate all possible words
    all_words = list(itertools.product(range(q), repeat=n))
    N = len(all_words)
    
    # Map word to index for fast lookup
    word_to_idx = {w: i for i, w in enumerate(all_words)}
    
    # Precompute coverage masks
    # mask[i] is a boolean array of length N where mask[i][j] is True if word j is within radius of word i
    masks = np.zeros((N, N), dtype=bool)
    
    # Helper to get neighbors
    def get_neighbors(word):
        neighbors = [word]
        for pos in range(n):
            for val in range(q):
                if val != word[pos]:
                    new_word = list(word)
                    new_word[pos] = val
                    neighbors.append(tuple(new_word))
        return neighbors

    # Fill masks
    for i, w in enumerate(all_words):
        neighs = get_neighbors(w)
        for nw in neighs:
            j = word_to_idx[nw]
            masks[i, j] = True
            
    # Greedy Set Cover
    uncovered = np.ones(N, dtype=bool)
    code = []
    
    while True:
        if time.time() > deadline:
            break
            
        if not np.any(uncovered):
            break
            
        # Find candidate that covers most uncovered words
        # Sum of (masks[i] & uncovered) for each i
        # Efficient way: dot product of masks with uncovered (treating bool as int)
        # But masks is bool, uncovered is bool. 
        # np.sum(masks & uncovered, axis=1) is O(N^2) per step. N=729, N^2=531k. 73 steps -> 38M ops.
        # This is fast enough in numpy.
        
        scores = np.sum(masks & uncovered, axis=1)
        best_idx = np.argmax(scores)
        
        if scores[best_idx] == 0:
            # Should not happen if code is complete, but break to avoid infinite loop
            break
            
        code.append(all_words[best_idx])
        uncovered = uncovered & ~masks[best_idx]
        
    return code
# EVOLVE-BLOCK-END
