import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 55.0
    
    # Seed: First-order Reed-Muller code on 16 coordinates, extended by two zero bits.
    # This gives 32 words of length 17 with distance 8.
    words = []
    for slope in range(16):
        for intercept in range(2):
            word = 0
            for point in range(16):
                bit = ((slope & point).bit_count() + intercept) % 2
                word |= bit << point
            # The 17th bit (index 16) is implicitly 0.
            words.append(word)
    
    code = set(words)
    N = 17
    MIN_DIST = 6
    TOTAL = 1 << N  # 131072
    
    # Precompute binary representation of all candidates for vectorized distance calc
    # unpackbits returns uint8, shape (TOTAL, 17)
    # We need to be careful with memory. 131072 * 17 * 1 byte = ~2.2 MB. This is fine.
    all_candidates = np.arange(TOTAL, dtype=np.uint32)
    # unpackbits requires uint8 or similar? It works on integers.
    # Let's convert to uint8 for unpackbits to be safe, or just use the integer array if supported.
    # np.unpackbits works on uint8. So we need to cast.
    # Actually, we can compute distance using XOR and then bit count.
    # A faster way in numpy for bit count of an array of ints is not directly available without looping or unpacking.
    # Unpacking is the best bet.
    
    # Create a 2D array of bits for all candidates
    # Note: unpackbits pads to multiple of 8. We want exactly 17 bits.
    # We can create a (TOTAL, 17) array.
    bits_candidates = np.zeros((TOTAL, N), dtype=np.uint8)
    # This loop is slow. Let's do it vectorized if possible?
    # No, unpackbits is the way.
    # We can unpack all candidates at once.
    # unpackbits(all_candidates) gives shape (TOTAL, 17) if we handle the padding correctly?
    # unpackbits pads to 8. So 17 bits -> 24 bits? No, it unpacks to 8 bits per byte.
    # If input is uint32, it might unpack 32 bits?
    # Let's stick to a robust method:
    # Convert candidates to a list of bytes? No.
    
    # Let's use the fact that we can compute Hamming distance between two arrays of bits.
    # We will maintain `code_bits` as a (|C|, 17) uint8 array.
    # We will maintain `candidate_bits` as a (TOTAL, 17) uint8 array.
    
    # Initialize candidate bits
    # Using a loop for initialization is acceptable if done once.
    # Or better:
    # We can generate bits using bitwise operations in a loop over bit positions?
    # for i in range(17): bits[:, i] = (all_candidates >> i) & 1
    # This is vectorized over candidates for each bit. 17 iterations of vectorized ops on 131k elements. Fast.
    
    candidate_bits = np.zeros((TOTAL, N), dtype=np.uint8)
    for i in range(N):
        candidate_bits[:, i] = (all_candidates >> i) & 1
    
    # Initialize code bits
    code_list = list(code)
    code_bits = np.zeros((len(code_list), N), dtype=np.uint8)
    for j, w in enumerate(code_list):
        for i in range(N):
            code_bits[j, i] = (w >> i) & 1
    
    # min_dists[i] is the min distance from candidate i to any word in code
    min_dists = np.full(TOTAL, 100, dtype=np.uint8)
    
    # Calculate initial min distances
    # Distance between candidate_bits (TOTAL, N) and code_bits (C, N)
    # dist = sum(candidate_bits[:, None, :] ^ code_bits[None, :, :], axis=2)
    # This creates a (TOTAL, C) array. 131072 * 32 * 1 byte = ~4 MB. Fine.
    # As code grows, C grows. 131072 * 256 * 1 = 33 MB. Still fine.
    # 131072 * 512 * 1 = 67 MB. Fine.
    # 131072 * 1024 * 1 = 134 MB. Getting large but manageable.
    # We can process in batches of candidates to keep memory low.
    
    BATCH_SIZE = 4096
    for batch_start in range(0, TOTAL, BATCH_SIZE):
        if time.time() > deadline - 5: break
        batch_end = min(batch_start + BATCH_SIZE, TOTAL)
        batch_bits = candidate_bits[batch_start:batch_end]
        
        # Compute distances to all current codewords
        # batch_bits: (B, N), code_bits: (C, N)
        # xor: (B, C, N) -> sum along N -> (B, C)
        # This might be memory heavy if C is large and B is large.
        # B=4096, C=256 -> 4096*256*17 = 17M bytes = 17MB. OK.
        
        # More efficient:
        # dists = np.sum(batch_bits[:, None, :] ^ code_bits[None, :, :], axis=2)
        # min_dists[batch_start:batch_end] = np.min(dists, axis=1)
        
        # Optimization: If we have many codewords, we can compute in chunks of codewords too?
        # For now, assume C is small enough.
        
        # Use a helper to compute min dist
        # We can also use the property that min(A, B) is associative.
        
        if len(code_bits) > 0:
             # Calculate XOR and sum
             # This creates a temporary array of size B x C x N
             xor_matrix = batch_bits[:, None, :] ^ code_bits[None, :, :]
             dists = np.sum(xor_matrix, axis=2)
             batch_mins = np.min(dists, axis=1)
             min_dists[batch_start:batch_end] = batch_mins
        else:
             min_dists[batch_start:batch_end] = 100

    # Mark seed words as used (distance 0)
    for w in code:
        min_dists[w] = 0

    # Greedy loop
    # We need to efficiently find the candidate with max min_dist among those with min_dist >= MIN_DIST.
    # And update min_dists after adding a new word.
    
    # To speed up updates, we only need to update min_dists for candidates that are still valid?
    # No, a candidate that was invalid (dist < 6) will remain invalid because min_dist only decreases.
    # So we can maintain a mask of valid candidates.
    
    valid_mask = min_dists >= MIN_DIST
    
    while time.time() < deadline - 5:
        if not np.any(valid_mask):
            break
        
        # Find best candidate among valid ones
        # We want argmax of min_dists where valid_mask is True.
        # We can set min_dists of invalid ones to -1 or 0 and argmax, but we need to be careful.
        # Let's create a masked array or just filter.
        
        valid_indices = np.where(valid_mask)[0]
        if len(valid_indices) == 0:
            break
            
        valid_dists = min_dists[valid_indices]
        best_local_idx = np.argmax(valid_dists)
        best_global_idx = valid_indices[best_local_idx]
        best_dist = valid_dists[best_local_idx]
        
        if best_dist < MIN_DIST:
            break
            
        new_word_idx = best_global_idx
        new_word_int = int(all_candidates[new_word_idx])
        
        # Add to code
        code.add(new_word_int)
        
        # Update code_bits
        # Append new row
        new_row = candidate_bits[new_word_idx]
        code_bits = np.vstack([code_bits, new_row])
        
        # Mark as invalid
        min_dists[new_word_idx] = 0
        valid_mask[new_word_idx] = False
        
        # Update min_dists for all *other* valid candidates
        # We only need to update candidates that are still valid.
        # Let's get the indices of currently valid candidates (excluding the one we just added, which is now invalid)
        # Actually, we can just update all, but it's wasted effort for invalid ones.
        # Let's update only valid ones to save time.
        
        current_valid_indices = np.where(valid_mask)[0]
        if len(current_valid_indices) == 0:
            break
            
        # Get bits for these candidates
        # We need to compute distance from these candidates to the NEW word only, and update min.
        # min_dists[i] = min(min_dists[i], dist(i, new_word))
        
        # Get bits of valid candidates
        valid_bits = candidate_bits[current_valid_indices]
        # new_word_bits is the last row of code_bits
        new_word_bits = code_bits[-1]
        
        # Compute distances
        # valid_bits: (V, N), new_word_bits: (N,)
        # xor: (V, N)
        # sum: (V,)
        dists_to_new = np.sum(valid_bits ^ new_word_bits, axis=1)
        
        # Update min_dists
        old_mins = min_dists[current_valid_indices]
        new_mins = np.minimum(old_mins, dists_to_new)
        min_dists[current_valid_indices] = new_mins
        
        # Update valid_mask
        # Candidates that dropped below MIN_DIST become invalid
        newly_invalid = new_mins < MIN_DIST
        if np.any(newly_invalid):
            # Get the global indices of these newly invalid candidates
            newly_invalid_global = current_valid_indices[newly_invalid]
            valid_mask[newly_invalid_global] = False
            
    return list(code)
