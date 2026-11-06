import numpy as np
import time
import random

def construct():
    N = 18
    MIN_DIST = 6
    deadline = time.time() + 85.0
    
    # 1. Initialize with Reed-Muller RM(1,4) extended
    base_words = []
    for slope in range(16):
        for intercept in range(2):
            word = 0
            for point in range(16):
                bit = ((slope & point).bit_count() + intercept) % 2
                word |= bit << point
            base_words.append(word)
    
    all_candidates = list(range(1 << N))
    
    best_code = list(base_words)
    best_size = len(best_code)
    
    # Precompute bit indices for fast extraction
    bit_indices = np.arange(N, dtype=np.uint32)
    
    # We will run multiple greedy passes
    # Each pass uses a different random shuffle of candidates
    # We start each pass from the base_words to avoid bias from previous poor choices
    # Actually, starting from base_words is good because it's a known valid core.
    
    pass_num = 0
    while time.time() < deadline - 5.0:
        pass_num += 1
        random.seed(pass_num) # Different shuffle for each pass
        shuffled_cands = all_candidates[:]
        random.shuffle(shuffled_cands)
        
        current_code = list(base_words)
        
        # Convert current_code to numpy array of bits for initial state
        # We will update this array as we add words? 
        # No, we update `current_code` list, and rebuild `bits_code` for each batch.
        # Rebuilding is cheap.
        
        batch_size = 4096 # Larger batch to reduce loop overhead
        
        for i in range(0, len(shuffled_cands), batch_size):
            if time.time() > deadline - 2.0:
                break
                
            batch = shuffled_cands[i:i+batch_size]
            if not batch:
                break
                
            # Convert batch to numpy array of bits (B, N)
            arr = np.array(batch, dtype=np.uint32)
            bits_batch = (arr[:, None] >> bit_indices) & 1
            
            # Current code bits
            if not current_code:
                current_code.extend(batch)
                continue
                
            code_arr = np.array(current_code, dtype=np.uint32)
            bits_code = (code_arr[:, None] >> bit_indices) & 1
            
            # Compute distances
            # Memory check: B=4096, K=~200, N=18
            # 4096 * 200 * 18 = 14.7M elements.
            # uint8: 14.7 MB. OK.
            
            xor = bits_batch[:, None, :] ^ bits_code[None, :, :]
            dists = xor.sum(axis=2) 
            min_dists = dists.min(axis=1)
            
            valid_mask = min_dists >= MIN_DIST
            valid_indices = np.where(valid_mask)[0]
            
            if len(valid_indices) > 0:
                new_words = [batch[j] for j in valid_indices]
                current_code.extend(new_words)
                
                if len(current_code) > best_size:
                    best_size = len(current_code)
                    best_code = current_code
    
    return best_code
