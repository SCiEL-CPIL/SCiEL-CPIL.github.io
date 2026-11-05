"""Combined approach: construct a large binary code of length 18, distance >= 6.

Strategy:
  1) Start from the Reed-Muller RM(1,4) code (32 words, distance 8) as a seed.
  2) Greedily add words to maximize the code size, checking pairwise distances
     using integer XOR and population count.
  3) Use a deadline guard to ensure we finish within 90s wall-clock.
  4) Return the largest verified code found.
"""

import time
import random

N = 18
MIN_DISTANCE = 6
FULL_SPACE = 1 << N  # 262144


def _popcount(x):
    """Population count for a single integer."""
    return bin(x).count('1')


def _hamming_dist(a, b):
    """Hamming distance between two integers (XOR then popcount)."""
    return _popcount(a ^ b)


def _construct_reed_muller_seed():
    """First-order Reed-Muller code on 16 coordinates, extended by two zero bits.
    Every affine Boolean function on F_2^4 is a word: 32 words, distance 8."""
    words = []
    for slope in range(16):
        for intercept in range(2):
            word = 0
            for point in range(16):
                bit = ((slope & point).bit_count() + intercept) % 2
                word |= bit << point
            words.append(word)
    return words


def _verify_code(words):
    """Verify all pairs have Hamming distance >= MIN_DISTANCE."""
    n = len(words)
    for i in range(n):
        for j in range(i + 1, n):
            if _hamming_dist(words[i], words[j]) < MIN_DISTANCE:
                return False
    return True


def _greedy_add_words(seed_words, t0, deadline):
    """Greedily add words to the code, maintaining minimum distance constraint.
    
    Uses a candidate pool and checks distances efficiently.
    Returns the final list of words.
    """
    code = list(seed_words)
    code_set = set(code)
    
    # Pre-compute distances from each seed word to all candidates
    # We'll use a bucket approach: for each candidate, track min distance to current code
    
    # To be efficient, we'll iterate through candidates in random order
    # and add those that satisfy the distance constraint
    
    # Build a candidate list (all words not in code)
    # For memory efficiency, we'll process in chunks
    
    # Shuffle for diversity
    candidates = list(range(FULL_SPACE))
    random.shuffle(candidates)
    
    # Remove words already in code
    candidates = [c for c in candidates if c not in code_set]
    
    # For each candidate, we need to check against all current code words
    # This is O(|code|) per candidate, which is expensive for large codes
    
    # Optimization: use the fact that we can early-exit if distance < MIN_DISTANCE
    
    added = 0
    for candidate in candidates:
        if time.time() > deadline:
            break
        
        # Check if candidate has distance >= MIN_DISTANCE to all code words
        valid = True
        for cw in code:
            if _hamming_dist(candidate, cw) < MIN_DISTANCE:
                valid = False
                break
        
        if valid:
            code.append(candidate)
            code_set.add(candidate)
            added += 1
            
            # Periodic verification (expensive, so do it sparingly)
            if added % 50 == 0:
                if not _verify_code(code):
                    # Should not happen, but just in case
                    print(f"Verification failed at size {len(code)}")
                    # Remove the last added word
                    code.pop()
                    code_set.discard(candidate)
    
    return code


def _greedy_add_words_optimized(seed_words, t0, deadline):
    """Optimized greedy addition using distance buckets.
    
    For each candidate, compute min distance to current code.
    Only add candidates with min distance >= MIN_DISTANCE.
    """
    code = list(seed_words)
    code_set = set(code)
    
    # For efficiency, we'll use numpy if available, otherwise fall back
    try:
        import numpy as np
        use_numpy = True
    except ImportError:
        use_numpy = False
    
    if use_numpy:
        # Pre-compute popcount LUT
        lut = np.array([bin(i).count('1') for i in range(256)], dtype=np.uint8)
        
        def popcount_array(X):
            """Popcount for an array of ints (up to 24 bits)."""
            return (lut[X & 255].astype(np.int16) +
                    lut[(X >> 8) & 255].astype(np.int16) +
                    lut[(X >> 16) & 255].astype(np.int16))
        
        # Process candidates in chunks
        candidates = np.array(list(range(FULL_SPACE)), dtype=np.int64)
        # Shuffle
        np.random.shuffle(candidates)
        # Remove words already in code
        code_arr_init = np.array(list(code_set), dtype=np.int64)
        # Simple filter: keep candidates not in code_set
        # This is O(n) but acceptable for 262144
        mask = np.array([c not in code_set for c in candidates], dtype=bool)
        candidates = candidates[mask]
        
        code_arr = np.array(code, dtype=np.int64)
        
        added = 0
        chunk_size = 10000
        
        for start in range(0, len(candidates), chunk_size):
            if time.time() > deadline:
                break
            
            chunk = candidates[start:start+chunk_size]
            
            # Compute min distance from each chunk element to current code
            # Shape: (chunk_len, code_len)
            # This can be large, so use sub-chunks
            sub_size = 512
            valid_indices = []
            
            for s in range(0, len(chunk), sub_size):
                if time.time() > deadline:
                    break
                wc = chunk[s:s+sub_size]
                # (len(wc), 1) XOR (1, len(code_arr)) -> (len(wc), len(code_arr))
                xor_matrix = wc[:, None] ^ code_arr[None, :]
                pc = popcount_array(xor_matrix)
                min_dists = pc.min(axis=1)
                
                valid_mask = min_dists >= MIN_DISTANCE
                valid_idx = np.where(valid_mask)[0]
                if len(valid_idx) > 0:
                    valid_indices.extend([s + idx for idx in valid_idx])
            
            # Add valid candidates
            for idx in valid_indices:
                if time.time() > deadline:
                    break
                candidate = int(chunk[idx])
                if candidate not in code_set:
                    code.append(candidate)
                    code_set.add(candidate)
                    code_arr = np.append(code_arr, candidate)
                    added += 1
        
        return code
    else:
        # Fallback to simple greedy
        return _greedy_add_words(seed_words, t0, deadline)


def construct():
    """Construct the largest possible collection of 18-bit words with pairwise
    Hamming distance at least 6."""
    t0 = time.time()
    deadline = t0 + 80.0  # 80s safety margin (90s limit)
    
    # Start with Reed-Muller seed
    seed_words = _construct_reed_muller_seed()
    
    # Verify seed
    if not _verify_code(seed_words):
        # Should not happen, but just in case
        return seed_words[:2]
    
    # Try to add more words greedily
    try:
        final_code = _greedy_add_words_optimized(seed_words, t0, deadline)
    except Exception as e:
        print(f"Error in optimized greedy: {e}")
        final_code = seed_words
    
    # Final verification
    if not _verify_code(final_code):
        # If verification fails, fall back to seed
        print("Final verification failed, returning seed")
        return seed_words
    
    return final_code
