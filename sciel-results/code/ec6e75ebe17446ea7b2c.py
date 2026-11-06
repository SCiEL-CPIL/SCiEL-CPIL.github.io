"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    deadline = time.time() + 55  # Safety margin before 60s limit (evaluator kills at 90s, but we want to be safe)
    
    # 1. Construct the extended quadratic residue code of length 18
    # This gives us a strong linear seed of size 512 with min distance 6.
    p = 19
    qr_set = set((x * x) % p for x in range(1, p))
    
    G = [[0] * 17 for _ in range(9)]
    for i in range(9):
        for j in range(17):
            val = (j - i) % p
            if val in qr_set:
                G[i][j] = 1
    
    # Generate 512 codewords
    seed_codewords = []
    for mask in range(512):
        word = 0
        for i in range(9):
            if (mask >> i) & 1:
                for j in range(17):
                    if G[i][j]:
                        word ^= (1 << j)
        # Extend with parity bit (bit 17)
        parity = word.bit_count() % 2
        word |= (parity << 17)
        seed_codewords.append(word)
    
    # Verify seed validity quickly
    # For QR code, min distance is known to be 6, but let's ensure no duplicates
    seed_codewords = list(set(seed_codewords))
    
    # If seed is invalid or empty (shouldn't happen), fallback to simple construction
    if not seed_codewords:
        return [0]
        
    # 2. Greedy Set Packing Search
    # State: list of integers
    code = seed_codewords[:]
    n = len(code)
    
    # Precompute numpy array for fast distance checks
    # We will maintain a numpy array of the code for vectorized operations
    code_arr = np.array(code, dtype=np.uint16)
    
    # Helper to check if a new word can be added
    # Returns True if valid, False otherwise
    def try_add(word_int):
        nonlocal code, code_arr, n
        if word_int in seen:
            return False
        # Check distance against all existing words
        # Convert word to numpy array
        new_word = np.array([word_int], dtype=np.uint16)
        # XOR
        xors = code_arr ^ new_word[0]
        # Popcount
        dists = np.array([x.bit_count() for x in xors], dtype=np.uint8)
        if np.min(dists) >= 6:
            code.append(word_int)
            code_arr = np.append(code_arr, word_int)
            n += 1
            seen.add(word_int)
            return True
        return False
    
    seen = set(code)
    
    # Time-budgeted search
    # Strategy:
    # 1. Try to add random words.
    # 2. Try to add words derived from existing code (e.g., weight-1, weight-2 mutations)
    #    to explore local structure.
    
    rng = np.random.default_rng(42)
    
    # Candidate generation strategies:
    # A. Random uniform
    # B. Mutations of existing codewords (flip 1, 2, 3 bits)
    # C. Complements (if code is not closed under complement, but QR is linear so it is)
    
    # To speed up, we can batch-check candidates.
    # However, Python loop for candidate generation + NumPy check is often bottlenecked by Python loop.
    # Let's try a hybrid: generate a batch of candidates, check them all at once against the code.
    
    batch_size = 1000
    max_iterations = 500000 # Hard cap on iterations just in case
    
    for _ in range(max_iterations):
        if time.time() > deadline:
            break
            
        # Generate batch of candidates
        candidates = []
        
        # 1. Random candidates
        num_random = 500
        random_words = rng.integers(0, 1 << 18, size=num_random, dtype=np.uint16)
        candidates.extend(random_words.tolist())
        
        # 2. Mutations of a few random codewords
        # Pick 10 random codewords from current code
        idxs = rng.integers(0, n, size=10)
        for idx in idxs:
            base = code[idx]
            # Flip 1 bit
            for bit in range(18):
                candidates.append(base ^ (1 << bit))
            # Flip 2 bits
            # Generate random pairs
            for _ in range(5):
                b1 = rng.integers(0, 18)
                b2 = rng.integers(0, 18)
                if b1 != b2:
                    candidates.append(base ^ (1 << b1) ^ (1 << b2))
            # Flip 3 bits
            for _ in range(2):
                b1 = rng.integers(0, 18)
                b2 = rng.integers(0, 18)
                b3 = rng.integers(0, 18)
                # Ensure distinct
                if b1 != b2 and b1 != b3 and b2 != b3:
                    candidates.append(base ^ (1 << b1) ^ (1 << b2) ^ (1 << b3))
        
        # Filter duplicates within batch and against seen
        unique_candidates = []
        for c in candidates:
            if c not in seen:
                unique_candidates.append(c)
        
        if not unique_candidates:
            continue
            
        # Vectorized check
        # Convert unique_candidates to numpy array
        cand_arr = np.array(unique_candidates, dtype=np.uint16)
        num_cands = len(cand_arr)
        
        # We need to check min distance for each candidate against all code words.
        # This is an outer product of XOR and popcount.
        # Memory: num_cands * n * 2 bytes.
        # If n=512 and num_cands=1000, that's ~1M entries, 2MB. Fine.
        # If n grows to 1000, still fine.
        
        # Reshape for broadcasting
        # cand_arr: (num_cands, 1)
        # code_arr: (1, n)
        
        # Check in chunks if memory is a concern, but 18 bits is small.
        # Let's do it in one go if size is reasonable.
        
        # Optimization: If n is large, we might want to check candidates one by one
        # or in small batches to avoid huge intermediate arrays if num_cands is large.
        # Here num_cands is ~1000-2000. n is ~512+.
        
        # Calculate distances
        # dists[i, j] = popcount(cand_arr[i] ^ code_arr[j])
        
        # We can use a trick: 
        # dists = np.array([np.array([c.bit_count() for c in (cand ^ code_arr)]) for cand in cand_arr])
        # This is slow in Python.
        
        # Better: Use numpy bit operations.
        # popcount for numpy array of uint16 can be done via lookup table or bit tricks.
        # NumPy doesn't have a native popcount for arrays in older versions, but we can use:
        # x = x - ((x >> 1) & 0x5555)
        # x = (x & 0x3333) + ((x >> 2) & 0x3333)
        # x = (x + (x >> 4)) & 0x0F0F
        # x = (x + (x >> 8)) & 0x00FF
        # This is for 32-bit. For 16-bit, simpler.
        
        # Actually, since we are in Python, iterating over candidates and doing vectorized check against code_arr
        # is often faster than a full 2D array creation if the number of candidates is moderate.
        
        added_count = 0
        for c in unique_candidates:
            if time.time() > deadline:
                break
            # Check distance
            xors = code_arr ^ c
            # Fast popcount for numpy array of uint16
            # Using the bit trick or just calling .bit_count() on python ints is slow.
            # Let's use a lookup table for bytes or similar?
            # Or just use the fact that we have uint16.
            # np.array([x.bit_count() for x in xors]) is O(n) python calls.
            
            # Alternative: Use np.unpackbits?
            # bits = np.unpackbits(xors.astype(np.uint8), axis=1)
            # dists = bits.sum(axis=1)
            # This creates a large array (n * 16).
            
            # Let's try the lookup table approach for speed.
            # Precompute popcount for 0-255
            # popcount_table = np.array([i.bit_count() for i in range(256)], dtype=np.uint8)
            # xors_low = (xors & 0xFF).astype(np.uint8)
            # xors_high = (xors >> 8).astype(np.uint8)
            # dists = popcount_table[xors_low] + popcount_table[xors_high]
            
            # This is very fast.
            
            # Let's implement this inside the loop for now, but it's better to vectorize the whole batch if possible.
            # Given the deadline, let's stick to the loop but optimize the inner check.
            
            # Optimization: Early exit if min dist < 6.
            # But we need to check all.
            
            # Let's use the lookup table method for the batch if we can.
            # Actually, doing it per candidate in a Python loop is okay if n is ~500-1000.
            # 1000 candidates * 500 checks = 500k bit_counts.
            # 500k bit_counts in Python is ~0.1s? Maybe.
            
            # Let's use the vectorized lookup table for the whole batch to be safe.
            # But we need to handle the "add if valid" logic which modifies the state.
            # We can't easily vectorize the "add" part because adding changes the set for subsequent candidates in the same batch?
            # Actually, if we check all candidates against the *current* state of the code, and then add all valid ones,
            # we might add two candidates that are too close to *each other*.
            # So we must check sequentially or ensure pairwise distance among new candidates.
            
            # Sequential check is safer.
            
            # To speed up the sequential check, let's use the lookup table.
            
            # We'll define the lookup table outside the loop.
            
            pass # Placeholder for logic below
            
        # Re-implementing the inner loop with optimization
        # We need a global popcount table
        # Let's move the table creation to outside the main loop.
        
    # The logic above was a bit messy. Let's rewrite the search loop cleanly.
    
    # Precompute popcount table for 0-255
    popcount_table = np.array([i.bit_count() for i in range(256)], dtype=np.uint8)
    
    # Reset code to seed
    code = seed_codewords[:]
    seen = set(code)
    code_arr = np.array(code, dtype=np.uint16)
    n = len(code)
    
    for _ in range(max_iterations):
        if time.time() > deadline:
            break
            
        candidates = []
        
        # 1. Random
        random_words = rng.integers(0, 1 << 18, size=500, dtype=np.uint16)
        candidates.extend(random_words.tolist())
        
        # 2. Mutations
        idxs = rng.integers(0, n, size=10)
        for idx in idxs:
            base = code[idx]
            for bit in range(18):
                candidates.append(base ^ (1 << bit))
            for _ in range(5):
                b1 = rng.integers(0, 18)
                b2 = rng.integers(0, 18)
                if b1 != b2:
                    candidates.append(base ^ (1 << b1) ^ (1 << b2))
        
        # Deduplicate candidates
        unique_candidates = []
        for c in candidates:
            if c not in seen:
                unique_candidates.append(c)
                
        # Check each candidate sequentially
        for c in unique_candidates:
            if time.time() > deadline:
                break
                
            # Check distance against all code words
            xors = code_arr ^ c
            # Use lookup table for popcount
            # xors are uint16
            low = (xors & 0xFF).astype(np.uint8)
            high = (xors >> 8).astype(np.uint8)
            dists = popcount_table[low] + popcount_table[high]
            
            if np.min(dists) >= 6:
                code.append(c)
                code_arr = np.append(code_arr, c)
                n += 1
                seen.add(c)
                
    return code

# EVOLVE-BLOCK-END
