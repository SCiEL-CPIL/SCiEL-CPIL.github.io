import numpy as np
import time

def construct():
    N = 17
    MIN_DIST = 6
    start_time = time.time()
    DEADLINE = 55.0  # Safety margin
    
    # Initial seed: 32 words from the current program
    words = []
    for slope in range(16):
        for intercept in range(2):
            word = 0
            for point in range(16):
                bit = ((slope & point).bit_count() + intercept) % 2
                word |= bit << point
            words.append(word)
    
    # Convert to numpy array for vectorized operations
    # Shape: (M, N)
    def to_bits(w):
        return np.array([(w >> i) & 1 for i in range(N)], dtype=np.uint8)
        
    words_bits = np.array([to_bits(w) for w in words])
    
    # Pre-compute all candidates bits
    # This is 131072 x 17. ~2MB.
    all_candidates = np.array([to_bits(w) for w in range(2**N)], dtype=np.uint8)
    
    # We will greedily add the candidate with the maximum minimum distance
    # We maintain a set of available candidates
    available = np.ones(2**N, dtype=bool)
    
    current_size = len(words)
    
    while True:
        if time.time() - start_time > DEADLINE:
            break
        
        if not np.any(available):
            break
            
        # Get indices of available candidates
        avail_indices = np.where(available)[0]
        
        # If too many, process in batches to avoid memory issues?
        # 131k * 256 * 17 * 8 bytes = 360MB. Might be okay, but let's batch.
        # Batch size 10000.
        best_candidate_idx = -1
        best_min_dist = -1
        
        batch_size = 10000
        for i in range(0, len(avail_indices), batch_size):
            batch_indices = avail_indices[i:i+batch_size]
            batch_candidates = all_candidates[batch_indices]
            
            # Calculate distances
            # batch_candidates: (B, N)
            # words_bits: (M, N)
            # diff: (B, M, N)
            diff = batch_candidates[:, None, :] != words_bits[None, :, :]
            dists = np.sum(diff, axis=2)
            min_dists = np.min(dists, axis=1)
            
            # Find best in this batch
            local_best_idx = np.argmax(min_dists)
            local_best_dist = min_dists[local_best_idx]
            
            if local_best_dist > best_min_dist:
                best_min_dist = local_best_dist
                best_candidate_idx = batch_indices[local_best_idx]
                
        if best_min_dist < MIN_DIST:
            break
            
        # Add the best candidate
        words.append(best_candidate_idx)
        new_word_bits = to_bits(best_candidate_idx)
        words_bits = np.vstack([words_bits, new_word_bits])
        available[best_candidate_idx] = False
        current_size += 1
        
    return words
