import numpy as np
import time

def construct():
    start_time = time.time()
    deadline = start_time + 80.0
    
    N = 18
    MIN_DIST = 6
    
    # Helper to convert int to bits array
    def int_to_bits(x):
        return np.array([(x >> i) & 1 for i in range(N)], dtype=np.uint8)
    
    # 1. Generate RM(2,4) words
    rm2_words = []
    for mask in range(2048):
        word = 0
        for point in range(16):
            val = 0
            if mask & 1: val ^= 1
            if mask & 2: val ^= (point >> 0) & 1
            if mask & 4: val ^= (point >> 1) & 1
            if mask & 8: val ^= (point >> 2) & 1
            if mask & 16: val ^= (point >> 3) & 1
            if mask & 32: val ^= ((point >> 0) & 1) * ((point >> 1) & 1)
            if mask & 64: val ^= ((point >> 0) & 1) * ((point >> 2) & 1)
            if mask & 128: val ^= ((point >> 0) & 1) * ((point >> 3) & 1)
            if mask & 256: val ^= ((point >> 1) & 1) * ((point >> 2) & 1)
            if mask & 512: val ^= ((point >> 1) & 1) * ((point >> 3) & 1)
            if mask & 1024: val ^= ((point >> 2) & 1) * ((point >> 3) & 1)
            word |= val << point
        rm2_words.append(word)
        
    # Initialize code with RM(2,4) words greedily
    code_ints = []
    code_bits = []
    
    # Sort by weight to potentially get better packing?
    rm2_words.sort(key=lambda x: x.bit_count())
    
    for w in rm2_words:
        if len(code_ints) == 0:
            code_ints.append(w)
            code_bits.append(int_to_bits(w))
        else:
            w_bits = int_to_bits(w)
            # Check against existing
            # dists = np.sum(np.array(code_bits) != w_bits, axis=1)
            # This is slow if done one by one.
            # But 2048 is small.
            
            valid = True
            for cb in code_bits:
                if np.sum(cb != w_bits) < MIN_DIST:
                    valid = False
                    break
            if valid:
                code_ints.append(w)
                code_bits.append(w_bits)
                
    code_bits_np = np.array(code_bits, dtype=np.uint8)
    
    # 2. Add random words
    while time.time() < deadline:
        batch_size = 500
        batch_ints = np.random.randint(0, 2**N, size=batch_size)
        batch_bits = np.array([int_to_bits(w) for w in batch_ints], dtype=np.uint8)
        
        # Vectorized check
        # code_bits_np: (M, 18)
        # batch_bits: (B, 18)
        
        # XOR: (M, 1, 18) ^ (1, B, 18) -> (M, B, 18)
        # This creates a large array. M can be ~1000, B=500. 1000*500*18 = 9M elements.
        # uint8 -> 9MB. This is fine.
        
        xor_result = code_bits_np[:, np.newaxis, :] ^ batch_bits[np.newaxis, :, :]
        dists = np.sum(xor_result, axis=2)
        
        # For each word in batch, check min distance
        min_dists = np.min(dists, axis=0)
        
        valid_indices = np.where(min_dists >= MIN_DIST)[0]
        
        if len(valid_indices) > 0:
            for idx in valid_indices:
                w_int = int(batch_ints[idx])
                code_ints.append(w_int)
                code_bits.append(batch_bits[idx])
                
            # Update code_bits_np
            code_bits_np = np.array(code_bits, dtype=np.uint8)
            
    return code_ints
