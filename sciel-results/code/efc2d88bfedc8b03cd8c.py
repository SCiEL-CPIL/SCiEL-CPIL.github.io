"""Greedy set-cover for ternary covering code K_3(6,1)."""
# EVOLVE-BLOCK-START
import numpy as np


def construct(q, n, radius):
    total = q ** n
    
    # Generate all words as a 2D array of shape (total, n)
    # We use base-q representation.
    # indices: 0 .. total-1
    # word[i] = digits of i in base q, reversed? 
    # Let's just generate them efficiently.
    # Actually, for small n, we can just loop or use broadcasting.
    # Let's use the standard digit extraction.
    all_words = np.zeros((total, n), dtype=np.uint8)
    for i in range(total):
        v = i
        for j in range(n):
            all_words[i, j] = v % q
            v //= q
    
    # Precompute coverage matrix
    # cover[i, j] = True if Hamming distance(all_words[i], all_words[j]) <= radius
    # For radius=1, dist <= 1.
    # We can compute this by checking if words are identical or differ in exactly 1 position.
    
    # Efficient way for radius 1:
    # 1. Identity matrix: all_words[i] == all_words[i]
    # 2. For each position k, check if all_words[i, :k] + all_words[i, k+1:] == all_words[j, :k] + all_words[j, k+1:]
    #    and all_words[i, k] != all_words[j, k]
    
    # Given the small size (729), a simpler vectorized approach might be faster to write and sufficiently fast.
    # Let's compute the full distance matrix or coverage directly.
    
    # Method: For each candidate i, find all j such that dist(i, j) <= 1.
    # We can create a mask for each i.
    # But doing it one by one is O(total^2 * n). 729^2 * 6 is ~3M ops, which is fine in Python if vectorized per i? 
    # Better: Vectorize over all i and j.
    
    # Let's compute the number of mismatches between all pairs.
    # all_words shape (total, n)
    # We want to compare all_words[i] with all_words[j] for all i, j.
    # This creates a tensor of shape (total, total, n).
    # total=729. 729*729*6 = ~3.1 million elements. This fits in memory easily.
    
    # Mismatch matrix: (total, total, n) boolean
    # mismatches[i, j, k] = (all_words[i, k] != all_words[j, k])
    
    # To avoid huge memory if n was large, but here n=6 is small.
    # Let's construct it.
    
    # Reshape all_words to broadcast
    # a = all_words[:, np.newaxis, :]  # (total, 1, n)
    # b = all_words[np.newaxis, :, :]  # (1, total, n)
    # mismatches = (a != b)            # (total, total, n)
    # dist = mismatches.sum(axis=2)    # (total, total)
    # coverage = (dist <= radius)      # (total, total)
    
    a = all_words[:, np.newaxis, :]
    b = all_words[np.newaxis, :, :]
    mismatches = (a != b)
    dist = mismatches.sum(axis=2)
    coverage = (dist <= radius).astype(np.uint8)
    
    # Greedy Set Cover
    # uncovered: boolean array of length total, initially all True
    uncovered = np.ones(total, dtype=bool)
    selected_indices = []
    
    while uncovered.any():
        # Count how many uncovered words each candidate covers
        # coverage is (total, total). coverage[i, j] is 1 if i covers j.
        # We want sum(coverage[i, :] * uncovered) for all i.
        # This is matrix-vector product: coverage @ uncovered
        # But uncovered is bool, coverage is uint8.
        # counts = coverage.dot(uncovered)
        
        counts = coverage.dot(uncovered)
        
        # Find the candidate with the maximum count
        # If multiple have same max, pick the first one (arbitrary)
        best_idx = np.argmax(counts)
        
        # If the best count is 0, we have a problem (should not happen for valid covering radius)
        if counts[best_idx] == 0:
            break
            
        # Select this candidate
        selected_indices.append(best_idx)
        
        # Update uncovered: set to False where this candidate covers
        uncovered &= ~coverage[best_idx, :].astype(bool)
        
    # Convert selected indices to list of lists of integers
    result = [all_words[i].tolist() for i in selected_indices]
    
    return result
# EVOLVE-BLOCK-END
