"""Construct a large binary code of length 17 with minimum distance 6."""
import numpy as np
import time
import heapq
import random

def construct():
    N = 17
    MIN_DIST = 6
    START_TIME = time.time()
    DEADLINE = START_TIME + 85.0  # safety margin before 90s kill

    # Precompute all vectors as uint8 bitsets for fast XOR and popcount
    all_vectors = np.arange(1 << N, dtype=np.uint16)
    
    # We'll use a bit-parallel approach with numpy for speed
    # Precompute popcount for all 2^17 values using lookup table
    popcount_table = np.zeros(1 << N, dtype=np.int32)
    for i in range(1, 1 << N):
        popcount_table[i] = popcount_table[i >> 1] + (i & 1)
    
    # Better: use numpy's bit_count equivalent via unpacking or a C-optimized approach
    # Actually, let's use a lookup table in chunks for numpy
    # Create popcount for all values using a vectorized approach
    # For 17 bits, we can split into 8-bit and 9-bit parts
    lo_mask = 0xFF
    hi_mask = 0x180  # bits 8-16 (9 bits)
    
    pop_lo = np.zeros(256, dtype=np.int32)
    for i in range(256):
        pop_lo[i] = bin(i).count('1')
    
    pop_hi = np.zeros(512, dtype=np.int32)
    for i in range(512):
        pop_hi[i] = bin(i).count('1')
    
    lo_idx = all_vectors & lo_mask
    hi_idx = (all_vectors >> 8) & 0x1FF
    
    dist_table = pop_lo[lo_idx] + pop_hi[hi_idx]
    
    # Now we need to build a code. Use greedy with conflict tracking.
    # Start with a seed: the [18,9,6] shortened code has 256 words.
    # Let's construct it first, then try to add more.
    
    # Construct the shortened quadratic residue code
    # The QR code of length 17: we'll use a known construction
    # Actually, let's use a simpler approach: start with the RM(1,4) extended code (32 words, dist 8)
    # and try to greedily add words.
    
    # Better strategy: Use a greedy maximum independent set approach
    # with a priority queue based on degree (number of conflicts)
    
    # First, let's get a good starting set. Use the RM(1,4) code (32 words, distance 8)
    code_set = set()
    for slope in range(16):
        for intercept in range(2):
            word = 0
            for point in range(16):
                bit = ((slope & point).bit_count() + intercept) % 2
                word |= bit << point
            code_set.add(word)
    
    # Now try to add more words greedily
    # For efficiency, track for each candidate whether it conflicts with the current code
    # A candidate conflicts if its distance to any codeword is < MIN_DIST
    
    # Use a more efficient approach: 
    # Maintain a "forbidden" mask - positions within distance < MIN_DIST of any codeword
    # But this is complex. Instead, use iterative greedy:
    
    best_code = set(code_set)
    best_size = len(best_code)
    
    # Try to add words using a greedy approach with randomization
    # To speed up, we'll check candidates in order of their "density" of conflicts
    
    # Precompute distances from each codeword to all candidates would be too slow
    # Instead, use a probabilistic/greedy approach:
    
    # Let's try a different structural approach:
    # Use a local search / hill climbing starting from the 256-word code
    
    # First, construct the 256-word shortened QR code properly
    # The [17,8,5] QR code extended... actually let me use a known 256-word code
    
    # Use the shortened [18,9,6] QR code:
    # The QR code of length 18 has 2^9=512 words, distance 6
    # Shortening by one coordinate gives at most 2^8=256 words
    
    # Let me construct it using the parity-check matrix approach
    # Actually, let's just use a greedy construction from scratch with better heuristics
    
    # Strategy: Start empty, add words one by one, choosing the word with fewest conflicts
    # that doesn't conflict with existing words
    
    code = set()
    remaining = set(range(1 << N))
    
    # For each candidate, we need to check if it's within distance < MIN_DIST of any codeword
    # To speed this up, we can use a spatial hashing approach or just brute force with numpy
    
    # Let's use a different approach: iterative greedy with numpy vectorization
    # For each step, compute distances from the next candidate to all current codewords
    
    # Actually, let me use a smarter approach:
    # 1. Start with a small code
    # 2. At each step, find the best candidate to add
    # 3. Use numpy for batch distance computations
    
    # Precompute all vectors
    all_vecs = np.array(list(range(1 << N)), dtype=np.uint16)
    
    # Function to check if a candidate is valid (distance >= MIN_DIST to all codewords)
    def is_valid(candidate, current_code):
        if not current_code:
            return True
        current_arr = np.array(list(current_code), dtype=np.uint16)
        # XOR with candidate
        xor_vals = np.bitwise_xor(current_arr, np.uint16(candidate))
        # Compute popcount
        lo = xor_vals & 0xFF
        hi = (xor_vals >> 8) & 0x1FF
        dists = pop_lo[lo] + pop_hi[hi]
        return np.min(dists) >= MIN_DIST
    
    # Greedy construction with randomization
    # To make it faster, let's use a batch approach
    
    # Let me try a completely different approach: use scipy for distance computation
    from scipy.spatial.distance import cdist
    
    # Actually, let's stick with numpy but be smarter
    
    # New strategy: Use a priority-based greedy
    # Start with the 32-word RM code
    code = set()
    for slope in range(16):
        for intercept in range(2):
            word = 0
            for point in range(16):
                bit = ((slope & point).bit_count() + intercept) % 2
                word |= bit << point
            code.add(word)
    
    # Now greedily add words
    # To speed up, maintain a list of "candidate scores"
    
    # For each step, we need to find a word that has distance >= 6 to all current codewords
    # Let's use a rejection sampling approach with numpy
    
    code_list = list(code)
    code_arr = np.array(code_list, dtype=np.uint16)
    
    added = 0
    max_add = 500  # try to add up to 500 more
    
    while time.time() < DEADLINE - 5 and added < max_add:
        # Try to add a random candidate
        # To be more systematic, try candidates in a structured way
        
        # Generate candidates: all words not in code
        # Check if they're valid
        
        # For efficiency, check in batches
        batch_size = 10000
        found = False
        
        # Try random candidates first
        for attempt in range(100):
            if time.time() > DEADLINE - 5:
                break
            candidate = random.randint(0, (1 << N) - 1)
            if candidate in code:
                continue
            
            # Check distance to all current codewords
            xor_vals = np.bitwise_xor(code_arr, np.uint16(candidate))
            lo = xor_vals & 0xFF
            hi = (xor_vals >> 8) & 0x1FF
            dists = pop_lo[lo] + pop_hi[hi]
            
            if np.min(dists) >= MIN_DIST:
                code.add(candidate)
                code_list.append(candidate)
                code_arr = np.array(code_list, dtype=np.uint16)
                added += 1
                found = True
                break
        
        if not found:
            # Try a more systematic search
            # Find the candidate with maximum minimum distance to current code
            # This is expensive, so do it in chunks
            
            if time.time() > DEADLINE - 10:
                break
            
            best_candidate = None
            best_min_dist = MIN_DIST - 1  # we need >= MIN_DIST
            
            # Sample candidates
            num_samples = min(50000, (1 << N) - len(code))
            samples = np.random.randint(0, 1 << N, size=num_samples, dtype=np.uint16)
            
            for candidate in samples:
                if candidate in code:
                    continue
                
                xor_vals = np.bitwise_xor(code_arr, np.uint16(candidate))
                lo = xor_vals & 0xFF
                hi = (xor_vals >> 8) & 0x1FF
                dists = pop_lo[lo] + pop_hi[hi]
                min_dist = np.min(dists)
                
                if min_dist >= MIN_DIST:
                    best_candidate = candidate
                    break
            
            if best_candidate is not None:
                code.add(best_candidate)
                code_list.append(best_candidate)
                code_arr = np.array(code_list, dtype=np.uint16)
                added += 1
            else:
                # Can't add more with random sampling, try exhaustive in smaller ranges
                break
    
    return list(code)
