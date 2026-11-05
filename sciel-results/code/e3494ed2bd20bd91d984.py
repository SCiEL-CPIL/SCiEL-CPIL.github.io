import numpy as np
import time
import random

def construct():
    N = 17
    MIN_DIST = 6
    deadline = time.time() + 85  # Safety margin before 90s kill
    
    # Step 1: Construct the 256-word linear code
    # Generator polynomial for [17,8,5] QR code
    def poly_mul(p1, p2):
        res = [0] * (len(p1) + len(p2) - 1)
        for i, c1 in enumerate(p1):
            if c1:
                for j, c2 in enumerate(p2):
                    if c2:
                        res[i+j] ^= 1
        return res

    f1 = [1, 0, 0, 1, 1, 1, 0, 0, 1]
    x_plus_1 = [1, 1]
    g = poly_mul(f1, x_plus_1)
    
    basis = []
    for i in range(8):
        vec = [0] * 17
        for j in range(len(g)):
            vec[i+j] = g[j]
        basis.append(vec)
    
    words_17 = []
    for mask in range(256):
        word = [0] * 17
        for i in range(8):
            if mask & (1 << i):
                for j in range(17):
                    word[j] ^= basis[i][j]
        words_17.append(word)
    
    words_18 = []
    for w in words_17:
        parity = sum(w) % 2
        words_18.append(w + [parity])
    
    final_words = []
    for w18 in words_18:
        if w18[0] == 0:
            new_word = w18[1:]
            val = 0
            for i, bit in enumerate(new_word):
                if bit:
                    val |= (1 << i)
            final_words.append(val)
    
    code = set(final_words)
    code_list = list(code)
    
    # Step 2: Nonlinear local search - try to add words or swap
    # Convert to numpy for vectorized distance calculations
    def to_bits(val, n=N):
        return np.array([(val >> i) & 1 for i in range(n)], dtype=np.uint8)
    
    def hamming_dist(a, b):
        return np.sum(a ^ b)
    
    # Build matrix of all codewords
    code_matrix = np.array([to_bits(w) for w in code_list], dtype=np.uint8)
    
    best_size = len(code_list)
    best_code = code_list[:]
    
    # Try adding random words
    all_words = set(range(2**N))
    used = set(code_list)
    candidates = list(all_words - used)
    random.shuffle(candidates)
    
    # Precompute bit representations for speed
    def compute_min_dist_to_code(new_word_bits, code_mat):
        # Compute Hamming distance from new_word to all words in code_mat
        dists = np.sum(code_mat ^ new_word_bits, axis=1)
        return np.min(dists)
    
    # Hill climbing: try to add words, or swap one word for another
    improvement_found = True
    while improvement_found and time.time() < deadline:
        improvement_found = False
        
        # Try adding random candidate words
        for cand in candidates[:500]:
            if time.time() > deadline:
                break
            cand_bits = to_bits(cand)
            min_d = compute_min_dist_to_code(cand_bits, code_matrix)
            if min_d >= MIN_DIST:
                code_matrix = np.vstack([code_matrix, cand_bits.reshape(1, -1)])
                code_list.append(cand)
                candidates.remove(cand)
                best_size = len(code_list)
                best_code = code_list[:]
                improvement_found = True
                break
        
        # Try swapping: remove one word, try to add two
        if not improvement_found and len(code_list) > 100:
            # Pick a random word to potentially remove
            idx = random.randint(0, len(code_list) - 1)
            removed_word = code_list[idx]
            removed_bits = code_matrix[idx]
            
            # Temporarily remove
            new_code_list = code_list[:idx] + code_list[idx+1:]
            new_code_matrix = np.delete(code_matrix, idx, axis=0)
            
            # Try to add two words that were previously too close
            added_count = 0
            for cand in candidates[:100]:
                if time.time() > deadline:
                    break
                cand_bits = to_bits(cand)
                if cand in used:
                    continue
                min_d = compute_min_dist_to_code(cand_bits, new_code_matrix)
                if min_d >= MIN_DIST:
                    new_code_matrix = np.vstack([new_code_matrix, cand_bits.reshape(1, -1)])
                    new_code_list.append(cand)
                    added_count += 1
                    if added_count >= 2:
                        break
            
            if added_count >= 2:
                code_list = new_code_list
                code_matrix = new_code_matrix
                best_size = len(code_list)
                best_code = code_list[:]
                improvement_found = True
    
    return best_code
