import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 55.0
    
    # Seed: First-order Reed-Muller code on 16 coordinates, extended by two zero bits.
    # This gives 32 words of length 17 with distance 8.
    words = []
    for slope in range(16):
        for intercept in range(2):
            word = 0
            for point in range(16):
                bit = ((slope & point).bit_count() + intercept) % 2
                word |= bit << point
            # The 17th bit (index 16) is implicitly 0.
            words.append(word)
    
    code = set(words)
    N = 17
    MIN_DIST = 6
    TOTAL = 1 << N  # 131072
    
    # Convert code to numpy array for vectorized operations
    code_arr = np.array(list(code), dtype=np.uint32)
    
    # Precompute all candidates as numpy array
    all_candidates = np.arange(TOTAL, dtype=np.uint32)
    
    # Initialize min distances
    min_dists = np.full(TOTAL, 100, dtype=np.uint8)
    
    # Mark seed words as used (distance 0)
    for w in code:
        min_dists[w] = 0
    
    # Calculate initial min distances using vectorized operations
    # Process in batches to manage memory
    BATCH_SIZE = 16384
    for batch_start in range(0, TOTAL, BATCH_SIZE):
        batch_end = min(batch_start + BATCH_SIZE, TOTAL)
        batch = all_candidates[batch_start:batch_end]
        
        # Compute distances from each candidate in batch to each codeword
        # Use XOR and bit_count via numpy
        # For efficiency, compute distance to each codeword separately and take min
        batch_min = np.full(len(batch), 100, dtype=np.uint8)
        
        for w in code:
            dists = np.array([(c ^ w).bit_count() for c in batch], dtype=np.uint8)
            batch_min = np.minimum(batch_min, dists)
        
        min_dists[batch_start:batch_end] = batch_min
    
    # Create a list of valid candidate indices (those with min_dist >= MIN_DIST)
    # We'll maintain this as a sorted structure for efficient max-finding
    # Use a heap or simply track the maximum
    
    # Greedy loop with periodic re-evaluation
    while time.time() < deadline:
        # Find the candidate with the maximum min distance that is still valid
        valid_mask = min_dists >= MIN_DIST
        if not np.any(valid_mask):
            break
        
        # Find index of max min_dist among valid candidates
        valid_indices = np.where(valid_mask)[0]
        valid_dists = min_dists[valid_indices]
        best_local = np.argmax(valid_dists)
        best_idx = valid_indices[best_local]
        best_d = valid_dists[best_local]
        
        if best_d < MIN_DIST:
            break
        
        new_word = int(all_candidates[best_idx])
        code.add(new_word)
        min_dists[best_idx] = 0  # Mark as used/invalid
        
        # Update min distances for all other candidates using vectorized ops
        # Only update candidates that are still valid (min_dist >= MIN_DIST)
        new_dists = np.array([(c ^ new_word).bit_count() for c in all_candidates], dtype=np.uint8)
        min_dists = np.minimum(min_dists, new_dists)
        
        # Periodic check to ensure we're making progress
        if len(code) % 50 == 0:
            pass  # Could add logging or adaptive strategies here
    
    return list(code)
