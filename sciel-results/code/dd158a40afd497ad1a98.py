import numpy as np
import time
import random

def construct():
    N = 18
    MIN_DIST = 6
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin
    
    # Generate initial Reed-Muller code (32 words)
    initial_words = []
    for slope in range(16):
        for intercept in range(2):
            word = 0
            for point in range(16):
                bit = ((slope & point).bit_count() + intercept) % 2
                word |= bit << point
            # Extend with 2 zeros (bits 16 and 17 are 0)
            initial_words.append(word)
    
    best_code = list(initial_words)
    best_size = len(best_code)
    
    if best_size == 512:
        return best_code

    # Precompute bit representations for speed? 
    # Or just convert on the fly.
    
    # We will try multiple random orders to find the maximal code
    # Since 512 is the target, we stop if we reach it.
    
    for attempt in range(20): # Max 20 attempts
        if time.time() > deadline:
            break
            
        code_set = set(initial_words)
        # Store as list of ints for set, and list of bit arrays for numpy
        code_bits_list = []
        for w in initial_words:
            bits = np.array([(w >> i) & 1 for i in range(N)], dtype=np.uint8)
            code_bits_list.append(bits)
        code_arr = np.array(code_bits_list)
        
        candidates = list(range(2**N))
        random.seed(attempt) # Deterministic randomness for reproducibility
        random.shuffle(candidates)
        
        for c in candidates:
            if time.time() > deadline:
                break
                
            if c in code_set:
                continue
                
            # Convert c to bits
            c_bits = np.array([(c >> i) & 1 for i in range(N)], dtype=np.uint8)
            
            # Calculate Hamming distances to all current codewords
            # XOR and popcount
            # code_arr is (K, 18), c_bits is (18,)
            # Broadcasting works: code_arr != c_bits -> (K, 18)
            diffs = code_arr != c_bits
            dists = np.sum(diffs, axis=1)
            
            if dists.min() >= MIN_DIST:
                code_set.add(c)
                code_bits_list.append(c_bits)
                code_arr = np.vstack((code_arr, c_bits[None, :]))
                
                if len(code_set) > best_size:
                    best_size = len(code_set)
                    best_code = list(code_set)
                    
                if best_size == 512:
                    return best_code

    return best_code
