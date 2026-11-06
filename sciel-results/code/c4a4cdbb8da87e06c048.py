"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 55.0
    
    N = 17
    MIN_DIST = 6
    
    # Generate the 32-word Reed-Muller seed (RM(1,4) extended to 17 bits)
    seed_words = []
    for slope in range(16):
        for intercept in range(2):
            word = 0
            for point in range(16):
                bit = ((slope & point).bit_count() + intercept) % 2
                if bit:
                    word |= (1 << point)
            seed_words.append(word)
    
    def get_bits(words):
        n_words = len(words)
        bits = np.zeros((n_words, N), dtype=np.uint8)
        for i, w in enumerate(words):
            for j in range(N):
                if w & (1 << j):
                    bits[i, j] = 1
        return bits
    
    def int_to_bits(w):
        b = np.zeros(N, dtype=np.uint8)
        for j in range(N):
            if w & (1 << j):
                b[j] = 1
        return b
    
    def min_dist_to_code(candidate_int, bits):
        cand_bits = int_to_bits(candidate_int)
        xor = bits ^ cand_bits
        dists = xor.sum(axis=1)
        return int(np.min(dists))
    
    def is_valid(candidate_int, bits):
        return min_dist_to_code(candidate_int, bits) >= MIN_DIST
    
    # Initial code
    current_words = seed_words[:]
    current_bits = get_bits(current_words)
    
    best_words = current_words[:]
    best_size = len(current_words)
    
    # Strategy: Greedy add with random restarts and swaps
    # Key insight: We need to find a large clique in the compatibility graph.
    # The 256-word linear code is a large clique. We can try to find it or a larger nonlinear one.
    # Since random search is slow, we use a guided approach:
    # 1. Start with the 32-word seed.
    # 2. Try to add words that are "close" to being valid (distance 5 or 6).
    # 3. Use swaps to escape local optima.
    
    # To make the search more effective, we will prioritize candidates that are
    # at a specific weight or have a specific structure.
    # The [17,8,6] QR code has words of various weights.
    
    # Let's implement a more aggressive local search:
    # - Keep a list of "near-miss" candidates (those with min dist 5).
    # - When stuck, try to resolve conflicts by swapping.
    
    attempts_no_add = 0
    max_no_add = 200  # Lower threshold for more frequent swaps
    
    # Precompute a set of current words for fast lookup? No, we need distances.
    
    # To speed up the search, we can use a heuristic for candidate generation.
    # Instead of purely random, we can bias towards words with weight around 8 or 9.
    # The average weight in a random 17-bit word is 8.5.
    
    def generate_candidate():
        # Generate a random word with weight 7, 8, or 9
        # This is more likely to be compatible with existing words than extreme weights.
        w = random.randint(7, 9)
        if w == 0:
            return 0
        # Choose w positions out of 17
        positions = random.sample(range(N), w)
        word = 0
        for p in positions:
            word |= (1 << p)
        return word
    
    while time.time() < deadline:
        
        # Phase 1: Try to add a valid word
        # Use a mix of random and structured candidates
        if random.random() < 0.7:
            candidate = generate_candidate()
        else:
            candidate = random.getrandbits(N)
            
        if is_valid(candidate, current_bits):
            # Add it
            current_words.append(candidate)
            current_bits = np.vstack([current_bits, int_to_bits(candidate)])
            
            if len(current_words) > best_size:
                best_size = len(current_words)
                best_words = current_words[:]
            
            attempts_no_add = 0
        else:
            attempts_no_add += 1
            
            # Phase 2: If stuck, try a swap
            if attempts_no_add > max_no_add:
                # Pick a random index to swap
                idx = random.randint(0, len(current_words) - 1)
                old_word = current_words[idx]
                
                # Generate candidate
                # Perturb old word or random
                if random.random() < 0.5:
                    candidate = generate_candidate()
                else:
                    # Flip 1-3 bits
                    flips = random.randint(1, 3)
                    mask = 0
                    for _ in range(flips):
                        pos = random.randint(0, N - 1)
                        mask |= (1 << pos)
                    candidate = old_word ^ mask
                
                # Check if candidate is valid against all words EXCEPT idx
                cand_bits = int_to_bits(candidate)
                xor = current_bits ^ cand_bits
                dists = xor.sum(axis=1)
                
                # Find indices where dist < MIN_DIST
                bad_indices = np.where(dists < MIN_DIST)[0]
                
                if len(bad_indices) == 0:
                    # Valid! Add it (this case should have been caught in Phase 1, but maybe random seed changed?)
                    # Actually, if Phase 1 failed, this shouldn't happen unless we modified bits?
                    # No, Phase 1 checked is_valid. If it failed, bad_indices > 0.
                    pass
                elif len(bad_indices) == 1 and bad_indices[0] == idx:
                    # Valid against all except the one we are swapping out!
                    # Perform swap
                    current_words[idx] = candidate
                    current_bits[idx] = cand_bits
                    attempts_no_add = 0 # Reset counter, we made progress (change)
                else:
                    # Not a valid swap
                    pass
                    
                # Reset attempts_no_add occasionally to keep trying swaps?
                # If we don't reset, we will keep swapping until we find a valid one or give up?
                # Let's reset it to 0 after a swap attempt to allow more swap attempts before giving up on this "stuck" state.
                # But if we do many bad swaps, we might waste time.
                # Let's limit swap attempts per "stuck" period.
                # For now, just let it run.
                
                # Safety: if we are spending too much time in swaps, break?
                # The loop condition handles time.

    return best_words

# EVOLVE-BLOCK-END
