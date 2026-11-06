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
    
    # Fast bit extraction using numpy
    def int_to_bits_array(words):
        n_words = len(words)
        if n_words == 0:
            return np.zeros((0, N), dtype=np.uint8)
        words_arr = np.array(words, dtype=np.uint32)
        bits = np.zeros((n_words, N), dtype=np.uint8)
        for j in range(N):
            mask = 1 << j
            bits[:, j] = (words_arr & mask) >> j
        return bits
    
    def bits_to_int(bits_row):
        result = 0
        for j in range(N):
            if bits_row[j]:
                result |= (1 << j)
        return result
    
    # Initial code
    current_words = seed_words[:]
    current_bits = int_to_bits_array(current_words)
    
    best_words = current_words[:]
    best_size = len(current_words)
    
    # Precompute lookup for fast distance checks
    def check_candidate(candidate_int, bits):
        """Check if candidate is at least MIN_DIST from all codewords in bits."""
        cand_bits = np.zeros(N, dtype=np.uint8)
        for j in range(N):
            if candidate_int & (1 << j):
                cand_bits[j] = 1
        xor = bits ^ cand_bits
        min_d = xor.sum(axis=1).min()
        return min_d >= MIN_DIST, min_d
    
    def try_add(candidate_int, bits, words):
        """Try to add candidate to the code. Returns True if successful."""
        valid, _ = check_candidate(candidate_int, bits)
        if valid:
            words.append(candidate_int)
            new_bits = int_to_bits_array([candidate_int])
            bits_new = np.vstack([bits, new_bits])
            return True, bits_new
        return False, bits
    
    # Phase 1: Greedy random insertion
    no_add_count = 0
    max_no_add = 500
    
    while time.time() < deadline:
        candidate = random.getrandbits(N)
        valid, _ = check_candidate(candidate, current_bits)
        if valid:
            current_words.append(candidate)
            new_bits = int_to_bits_array([candidate])
            current_bits = np.vstack([current_bits, new_bits])
            
            if len(current_words) > best_size:
                best_size = len(current_words)
                best_words = current_words[:]
            
            no_add_count = 0
        else:
            no_add_count += 1
            if no_add_count >= max_no_add:
                no_add_count = 0
    
    # Phase 2: Local search with replacement moves
    # Try to replace one codeword with two new ones (net gain of 1)
    replacement_attempts = 0
    max_repl_attempts = 10000
    
    while time.time() < deadline and replacement_attempts < max_repl_attempts:
        replacement_attempts += 1
        
        # Pick a random codeword to potentially replace
        idx = random.randint(0, len(current_words) - 1)
        
        # Generate a candidate that's close to the existing word
        if random.random() < 0.3:
            # Random word
            candidate = random.getrandbits(N)
        else:
            # Flip 1-4 bits of the existing word
            old_word = current_words[idx]
            n_flips = random.randint(1, 4)
            positions = random.sample(range(N), n_flips)
            mask = 0
            for pos in positions:
                mask |= (1 << pos)
            candidate = old_word ^ mask
        
        # Check if candidate is valid when replacing word at idx
        # Remove word at idx, check candidate against remaining
        if len(current_words) <= 1:
            continue
            
        remaining_indices = [i for i in range(len(current_words)) if i != idx]
        remaining_bits = current_bits[remaining_indices]
        
        valid, min_d = check_candidate(candidate, remaining_bits)
        
        if valid:
            # Replace: remove idx, add candidate
            new_words = [current_words[i] for i in remaining_indices]
            new_words.append(candidate)
            new_bits = int_to_bits_array(new_words)
            
            current_words = new_words
            current_bits = new_bits
            
            if len(current_words) > best_size:
                best_size = len(current_words)
                best_words = current_words[:]
    
    # Phase 3: Try to add more words with different strategies
    no_add_count = 0
    max_no_add = 2000
    
    while time.time() < deadline:
        # Try various candidate generation strategies
        strategy = random.random()
        
        if strategy < 0.4:
            # Pure random
            candidate = random.getrandbits(N)
        elif strategy < 0.7:
            # Flip 1-2 bits of a random existing word
            if current_words:
                base = current_words[random.randint(0, len(current_words) - 1)]
                n_flips = random.randint(1, 2)
                positions = random.sample(range(N), n_flips)
                mask = 0
                for pos in positions:
                    mask |= (1 << pos)
                candidate = base ^ mask
            else:
                candidate = random.getrandbits(N)
        else:
            # Flip 3-5 bits of a random existing word
            if current_words:
                base = current_words[random.randint(0, len(current_words) - 1)]
                n_flips = random.randint(3, 5)
                positions = random.sample(range(N), n_flips)
                mask = 0
                for pos in positions:
                    mask |= (1 << pos)
                candidate = base ^ mask
            else:
                candidate = random.getrandbits(N)
        
        valid, _ = check_candidate(candidate, current_bits)
        if valid:
            current_words.append(candidate)
            new_bits = int_to_bits_array([candidate])
            current_bits = np.vstack([current_bits, new_bits])
            
            if len(current_words) > best_size:
                best_size = len(current_words)
                best_words = current_words[:]
            
            no_add_count = 0
        else:
            no_add_count += 1
            if no_add_count >= max_no_add:
                break
    
    return best_words
# EVOLVE-BLOCK-END
