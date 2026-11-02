"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import itertools


def construct(q, n, radius):
    """
    Greedy set-cover algorithm using numpy for efficiency.
    Precomputes coverage matrix and uses vectorized operations for updates.
    """
    if n <= 0:
        return [[]]
    
    total_words = q ** n
    
    # Generate all words
    words = []
    for i in range(total_words):
        w = [0] * n
        x = i
        for j in range(n - 1, -1, -1):
            w[j] = x % q
            x //= q
        words.append(w)
    
    # Convert words to numpy array for vectorized distance calculation
    words_arr = np.array(words, dtype=np.int8) # shape (N, n)
    
    # Precompute coverage matrix: cov[i, j] = 1 if dist(words[i], words[j]) <= radius
    cov = np.zeros((total_words, total_words), dtype=np.int8)
    for i in range(total_words):
        diff = (words_arr != words_arr[i]).sum(axis=1)
        cov[i] = (diff <= radius).astype(np.int8)
    
    # Greedy Set Cover
    uncovered = np.ones(total_words, dtype=np.int8)
    gains = cov.sum(axis=1).astype(np.int32)
    
    codewords = []
    
    while uncovered.sum() > 0:
        best_idx = np.argmax(gains)
        if gains[best_idx] <= 0:
            break
            
        codewords.append(words[best_idx])
        
        # Update uncovered
        newly_covered = uncovered & cov[best_idx]
        uncovered -= newly_covered
        
        # Update gains
        gains -= cov.dot(newly_covered)
        
        # Prevent re-selection
        gains[best_idx] = -1

    return codewords
# EVOLVE-BLOCK-END
