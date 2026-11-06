"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Leave 5s margin for 60s limit
    
    N = 17
    MIN_DIST = 6
    
    # Generate the 32-word Reed-Muller seed: RM(1,4) extended to 17 bits
    seed_words = []
    for slope in range(16):
        for intercept in range(2):
            word = 0
            for point in range(16):
                bit = ((slope & point).bit_count() + intercept) % 2
                if bit:
                    word |= (1 << point)
            seed_words.append(word)
            
    # Convert to numpy array for vectorized operations
    def get_bits(words):
        n_words = len(words)
        bits = np.zeros((n_words, N), dtype=np.uint8)
        for i, w in enumerate(words):
            for j in range(N):
                if w & (1 << j):
                    bits[i, j] = 1
        return bits
    
    def min_dist_to_code(candidate_int, bits):
        cand_bits = np.zeros(N, dtype=np.uint8)
        for j in range(N):
            if candidate_int & (1 << j):
                cand_bits[j] = 1
        xor = bits ^ cand_bits
        dists = xor.sum(axis=1)
        return np.min(dists)
    
    def is_valid(candidate_int, bits):
        return min_dist_to_code(candidate_int, bits) >= MIN_DIST
    
    def int_to_bits(w):
        bits = np.zeros(N, dtype=np.uint8)
        for j in range(N):
            if w & (1 << j):
                bits[j] = 1
        return bits
    
    # Initial code
    current_words = seed_words[:]
    current_bits = get_bits(current_words)
    
    best_words = current_words[:]
    best_size = len(current_words)
    
    # Local Search / Greedy Expansion
    attempts_no_add = 0
    max_no_add = 300
    
    while time.time() < deadline:
        # Try to add a random word
        candidate = random.getrandbits(N)
        if is_valid(candidate, current_bits):
            current_words.append(candidate)
            current_bits = np.vstack([current_bits, int_to_bits(candidate)])
            
            if len(current_words) > best_size:
                best_size = len(current_words)
                best_words = current_words[:]
            
            attempts_no_add = 0
        else:
            attempts_no_add += 1
            
            # If stuck, try a swap
            if attempts_no_add > max_no_add:
                idx = random.randint(0, len(current_words) - 1)
                old_word = current_words[idx]
                
                if random.random() < 0.5:
                    candidate = random.getrandbits(N)
                else:
                    flips = random.randint(1, 3)
                    mask = 0
                    for _ in range(flips):
                        pos = random.randint(0, N - 1)
                        mask |= (1 << pos)
                    candidate = old_word ^ mask
                
                cand_bits = int_to_bits(candidate)
                xor = current_bits ^ cand_bits
                dists = xor.sum(axis=1)
                bad_indices = np.where(dists < MIN_DIST)[0]
                
                if len(bad_indices) == 0:
                    current_words.append(candidate)
                    current_bits = np.vstack([current_bits, cand_bits])
                    if len(current_words) > best_size:
                        best_size = len(current_words)
                        best_words = current_words[:]
                    attempts_no_add = 0
                elif len(bad_indices) == 1 and bad_indices[0] == idx:
                    current_words[idx] = candidate
                    current_bits[idx] = cand_bits
                    attempts_no_add = 0
                else:
                    pass
                
                attempts_no_add = 0  # Reset to allow more swaps
    
    return best_words
# EVOLVE-BLOCK-END
