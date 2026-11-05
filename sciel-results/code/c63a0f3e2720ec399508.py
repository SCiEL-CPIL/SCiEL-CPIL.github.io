import numpy as np
import time

def construct():
    N = 17
    MIN_D = 6
    deadline = time.time() + 55  # safety margin before 60s limit
    
    # Precompute popcount table for O(1) Hamming distance lookups
    # For 17-bit numbers, we can split into two 8-bit halves + 1 bit
    # popcount[x] = number of 1-bits in x
    popcount = np.zeros(256, dtype=np.uint8)
    for i in range(256):
        popcount[i] = bin(i).count('1')
    
    def popcount17(x):
        """Fast popcount for 17-bit integers"""
        return popcount[x & 0xFF] + popcount[(x >> 8) & 0xFF] + popcount[(x >> 16) & 0xFF]
    
    # Step 1: Generate the 256-word linear [17,8,6] code
    # f1(x) = x^8 + x^4 + x^3 + x^2 + 1
    f1 = [1, 0, 0, 1, 1, 1, 0, 0, 1]
    
    basis = []
    for i in range(9):
        vec = [0] * 17
        for j in range(len(f1)):
            if i + j < 17:
                vec[i+j] = f1[j]
        basis.append(vec)
        
    words_17 = []
    for mask in range(512):
        word = [0] * 17
        for i in range(9):
            if mask & (1 << i):
                for j in range(17):
                    word[j] ^= basis[i][j]
        words_17.append(word)
        
    # Extend to 18 bits
    words_18 = []
    for w in words_17:
        parity = sum(w) % 2
        words_18.append(w + [parity])
        
    # Shorten: keep words where bit 0 is 0
    codewords = []
    for w18 in words_18:
        if w18[0] == 0:
            new_word = w18[1:]
            val = 0
            for i, bit in enumerate(new_word):
                if bit:
                    val |= (1 << i)
            codewords.append(val)
    
    # Step 2: Greedily add any 17-bit word at distance >= 6 from all existing
    # Use popcount-based distance for speed
    
    def hamming_dist(a, b):
        """Hamming distance between two 17-bit integers using popcount table"""
        return popcount17(a ^ b)
    
    # Track which candidates have been checked
    checked = np.zeros(2**N, dtype=bool)
    
    # Iteratively try to add words
    added_any = True
    while added_any and time.time() < deadline:
        added_any = False
        
        # For each unchecked candidate, check if it can be added
        # We need to check all pairwise distances with existing codewords
        
        n_codes = len(codewords)
        if n_codes == 0:
            # Add the first word
            codewords.append(0)
            checked[0] = True
            added_any = True
            continue
        
        # Precompute weights of existing codewords for distance formula
        # dist(a,b) = weight(a) + weight(b) - 2*popcount(a & b)
        code_weights = np.array([popcount17(c) for c in codewords], dtype=np.uint16)
        
        # Process candidates in batches
        batch_size = 50000
        for start in range(0, 2**N, batch_size):
            if time.time() >= deadline:
                break
            end = min(start + batch_size, 2**N)
            
            # Get unchecked candidates in this batch
            batch_indices = []
            for i in range(start, end):
                if not checked[i]:
                    batch_indices.append(i)
            
            if not batch_indices:
                continue
            
            batch_indices = np.array(batch_indices, dtype=np.int64)
            batch_vals = batch_indices  # The actual integer values are the indices
            
            # For each candidate, compute min distance to all codewords
            # dist = weight(cand) + weight(code) - 2*popcount(cand & code)
            
            batch_weights = np.array([popcount17(v) for v in batch_vals], dtype=np.uint16)
            
            # Compute popcount(cand & code) for all pairs
            # This is the expensive part - do it in sub-batches to manage memory
            sub_batch_size = 10000
            for sub_start in range(0, len(batch_vals), sub_batch_size):
                if time.time() >= deadline:
                    break
                sub_end = min(sub_start + sub_batch_size, len(batch_vals))
                sub_vals = batch_vals[sub_start:sub_end]
                sub_weights = batch_weights[sub_start:sub_end]
                
                # For each code word, compute popcount(sub_val & code)
                min_dists = np.full(len(sub_vals), 100, dtype=np.uint16)
                
                for j, code in enumerate(codewords):
                    # Compute popcount(sub_val & code) for all sub_vals
                    # sub_val & code gives a 17-bit number, use popcount table
                    and_vals = sub_vals & code
                    pc = np.array([popcount17(v) for v in and_vals], dtype=np.uint16)
                    
                    # Distance = weight(sub) + weight(code) - 2*pc
                    dists = sub_weights + code_weights[j] - 2 * pc
                    min_dists = np.minimum(min_dists, dists)
                
                # Find candidates with min_dist >= MIN_D
                valid_mask = min_dists >= MIN_D
                valid_sub_indices = np.where(valid_mask)[0]
                
                for idx in valid_sub_indices:
                    if time.time() >= deadline:
                        break
                    candidate = sub_vals[idx]
                    codewords.append(int(candidate))
                    checked[candidate] = True
                    added_any = True
                    code_weights = np.append(code_weights, popcount17(int(candidate)))
    
    return codewords
