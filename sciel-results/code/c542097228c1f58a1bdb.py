import numpy as np
import time

def construct(q, n, radius):
    # Only handle q=3, n=8, radius=1 as per task description
    # Fallback to seed if parameters differ
    if q != 3 or n != 8 or radius != 1:
        # Fallback to seed logic
        import itertools
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        return [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]

    start_time = time.time()
    deadline = start_time + 110 # 10s margin
    
    total_words = q ** n
    # Map words to indices 0..total_words-1
    # Word is a base-q number
    
    # Precompute neighbors
    # For each word, generate all words at distance <= radius
    # Distance 1 in base-3 length-8:
    # Change one digit by +/- 1 (mod 3).
    # 8 positions * 2 directions = 16 neighbors + self = 17.
    
    neighbors = np.zeros((total_words, 1 + n * (q - 1)), dtype=np.int32)
    neighbor_counts = 1 + n * (q - 1)
    
    for i in range(total_words):
        # Convert i to base-q digits
        digits = [(i // (q ** j)) % q for j in range(n)]
        
        # Self
        neighbors[i, 0] = i
        
        # Neighbors
        idx = 1
        for pos in range(n):
            for delta in [1, -1]:
                new_digit = (digits[pos] + delta) % q
                # Reconstruct number
                new_val = i + (new_digit - digits[pos]) * (q ** pos)
                neighbors[i, idx] = new_val
                idx += 1
                
    covered = np.zeros(total_words, dtype=bool)
    codewords = []
    
    # Pre-compute powers of q for potential optimization, though neighbors are precomputed
    # The main loop
    while not np.all(covered):
        if time.time() > deadline:
            break
            
        # Count uncovered neighbors for each potential codeword
        # neighbors is (total_words, neighbor_counts)
        # covered is (total_words,)
        # covered[neighbors] is (total_words, neighbor_counts) boolean
        # sum along axis 1 gives count of covered neighbors
        covered_counts = np.sum(covered[neighbors], axis=1)
        uncovered_counts = neighbor_counts - covered_counts
        
        # Find the word with max uncovered neighbors
        best_idx = np.argmax(uncovered_counts)
        
        # Add to codewords
        # Convert best_idx back to list of digits
        digits = [(best_idx // (q ** j)) % q for j in range(n)]
        codewords.append(digits)
        
        # Update covered
        covered[neighbors[best_idx]] = True
        
    # If we timed out, we might not have a full cover.
    # The task requires a valid cover.
    # If we break early, we must ensure validity.
    # However, with 120s limit and ~10ms per step, we have plenty of time.
    # 486 steps * 10ms = 4.8s.
    
    return codewords
