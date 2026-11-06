import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 55.0
    
    N = 17
    MIN_DIST = 6
    TOTAL_WORDS = 1 << N
    
    # Generate all words
    all_words = np.arange(TOTAL_WORDS, dtype=np.uint32)
    
    # Precompute bits for all words?
    # 131072 x 17 uint8 array = ~2MB. OK.
    bits_all = np.zeros((TOTAL_WORDS, N), dtype=np.uint8)
    for i in range(TOTAL_WORDS):
        w = all_words[i]
        for j in range(N):
            if w & (1 << j):
                bits_all[i, j] = 1
                
    # Generate RM(1,4) seed (32 words)
    seed_words = []
    for slope in range(16):
        for intercept in range(2):
            word = 0
            for point in range(16):
                bit = ((slope & point).bit_count() + intercept) % 2
                if bit:
                    word |= (1 << point)
            seed_words.append(word)
            
    # Map seed words to indices in all_words
    seed_indices = []
    for w in seed_words:
        idx = np.where(all_words == w)[0][0]
        seed_indices.append(idx)
        
    best_code_indices = []
    best_size = 0
    
    # Run multiple greedy attempts
    attempts = 0
    while time.time() < deadline:
        attempts += 1
        # Randomize tie-breaking by shuffling the initial order?
        # Or just use a random permutation of indices for tie-breaking in argmax?
        
        # Initialize min_dist_to_code
        # For seed words, dist is 0. For others, inf.
        min_dist = np.full(TOTAL_WORDS, np.inf, dtype=np.float32)
        
        current_code_indices = seed_indices[:]
        
        # Update min_dist for all words against seed words
        # Vectorized:
        # bits_seed = bits_all[seed_indices] (32 x 17)
        # dists = sum(bits_all[:, None, :] ^ bits_seed[None, :, :], axis=2) (131072 x 32)
        # min_dist = min(dists, axis=1)
        
        bits_seed = bits_all[seed_indices]
        # To save memory, we can do this in chunks or just let it allocate (131072*32*8 = 33MB). OK.
        dists_seed = np.sum(bits_all[:, None, :] ^ bits_seed[None, :, :], axis=2)
        min_dist = np.min(dists_seed, axis=1)
        
        # Mark seed words as used (dist 0)
        min_dist[seed_indices] = 0
        
        # Greedy loop
        while time.time() < deadline:
            # Find index with max min_dist
            # We want the one with max min_dist, provided min_dist >= MIN_DIST
            # If max min_dist < MIN_DIST, break
            
            # To avoid picking used words (dist 0), we can set them to -1
            # But they are already 0, and we want max.
            # If we have valid words, their dist is >= 6.
            # So argmax will pick a valid one if any exist.
            
            # Optimization: Only consider words with min_dist >= MIN_DIST
            # valid_mask = min_dist >= MIN_DIST
            # if not np.any(valid_mask): break
            # candidate_idx = np.argmax(np.where(valid_mask, min_dist, -1))
            
            # Faster:
            max_dist = np.max(min_dist)
            if max_dist < MIN_DIST:
                break
                
            # Find all with max_dist? Or just one?
            # Random tie-breaking:
            candidates = np.where(min_dist == max_dist)[0]
            candidate_idx = random.choice(candidates)
            
            # Add candidate
            current_code_indices.append(candidate_idx)
            
            # Update min_dist for all OTHER words
            # dists_new = sum(bits_all[:, None, :] ^ bits_all[candidate_idx][None, :, :], axis=1)
            # min_dist = min(min_dist, dists_new)
            # But we must not update min_dist for words already in the code?
            # Actually, min_dist for code words should stay 0 (or whatever).
            # If we update them, they might become < 6, but we don't care because we won't pick them again?
            # We should mark them as "used" so they aren't picked.
            # Setting min_dist[used] = -1 is safe.
            
            bits_cand = bits_all[candidate_idx]
            dists_new = np.sum(bits_all[:, None, :] ^ bits_cand[None, :, :], axis=1)
            min_dist = np.minimum(min_dist, dists_new)
            
            # Mark used words to prevent re-selection
            min_dist[current_code_indices] = -1
            
            # Check time
            if time.time() > deadline - 5:
                break
                
        # End of greedy attempt
        size = len(current_code_indices)
        if size > best_size:
            best_size = size
            best_code_indices = current_code_indices[:]
            
        # If we found 256, we can stop early?
        if best_size >= 256:
            break
            
    # Convert indices back to integers
    return [int(all_words[i]) for i in best_code_indices]
