import time
import numpy as np

def construct():
    N = 18
    MIN_DIST = 6
    TOTAL_BUDGET = 55.0
    t_start = time.time()
    deadline = t_start + TOTAL_BUDGET

    # --- Attempt 1: Deterministic construction of [18,9,6] code ---
    # Use the extended quadratic residue code approach via proper GF(2) arithmetic.
    # The [23,12,7] QR code has generator polynomial g(x) = 1 + x + x^5 + x^6 + x^7 + x^9 + x^11
    # We'll construct it correctly using convolution.
    
    # Generator polynomial bits (bit i set means x^i present)
    g_poly = 0
    for bit in (0, 1, 5, 6, 7, 9, 11):
        g_poly |= 1 << bit
    
    # Generate all 4096 codewords of the [23,12,7] QR code
    qr_words = []
    for m in range(4096):
        c = 0
        mm = m
        gp = g_poly
        while mm:
            if mm & 1:
                c ^= gp
            mm >>= 1
            gp <<= 1
        qr_words.append(c & ((1 << 23) - 1))
    
    # Shorten by 3 positions: keep codewords with bits 0,1,2 == 0
    # This gives a [20,9,7] subcode with 512 words
    shortened = []
    for w in qr_words:
        if (w & 0b111) == 0:
            new_w = (w >> 3) & ((1 << 20) - 1)
            shortened.append(new_w)
    
    if len(shortened) == 512:
        # Puncture 2 positions (bits 0,1) to get [18,9,6]
        punctured = []
        for w in shortened:
            new_w = (w >> 2) & ((1 << 18) - 1)
            punctured.append(new_w)
        
        # Verify: check all pairs have distance >= 6
        # Use numpy for speed
        arr = np.array(punctured, dtype=np.uint32)
        n = len(arr)
        valid = True
        for i in range(n):
            if time.time() > deadline - 5.0:
                valid = False
                break
            xors = np.bitwise_xor(arr[i+1:], arr[i])
            # Compute popcount for each xor
            counts = np.array([x.bit_count() for x in xors], dtype=np.int32)
            if counts.min() < MIN_DIST:
                valid = False
                break
        
        if valid:
            return punctured

    # --- Attempt 2: Accelerated greedy with local search ---
    # Use a bitset-based approach for speed
    pop6 = np.array([i.bit_count() for i in range(64)], dtype=np.uint8)
    
    def popcount18(arr):
        lo = (arr & 0x3F).astype(np.intp)
        mid = ((arr >> 6) & 0x3F).astype(np.intp)
        hi = ((arr >> 12) & 0x3F).astype(np.intp)
        return pop6[lo].astype(np.uint8) + pop6[mid].astype(np.uint8) + pop6[hi].astype(np.uint8)
    
    # Generate candidates sorted by weight (lighter first for greedy)
    all_candidates = list(range(1, 2**N))
    all_candidates.sort(key=lambda x: (x.bit_count(), x))
    
    best_words = [0]
    best_len = 1
    rng = np.random.default_rng(42)
    
    # Try multiple restarts with different orderings
    for attempt in range(20):
        if time.time() > deadline - 5.0:
            break
        
        if attempt == 0:
            candidates = all_candidates
        else:
            candidates = all_candidates[:]
            # Shuffle within weight classes
            i = 0
            while i < len(candidates):
                j = i + 1
                while j < len(candidates) and candidates[j].bit_count() == candidates[i].bit_count():
                    j += 1
                rng.shuffle(candidates[i:j])
                i = j
        
        words_list = [0]
        words_arr = np.array(words_list, dtype=np.uint32)
        
        for c in candidates:
            if time.time() > deadline - 5.0:
                break
            
            xor_vals = np.bitwise_xor(words_arr, np.uint32(c))
            bit_counts = popcount18(xor_vals)
            
            if bit_counts.min() >= MIN_DIST:
                words_list.append(c)
                words_arr = np.append(words_arr, np.uint32(c))
        
        if len(words_list) > best_len:
            best_len = len(words_list)
            best_words = words_list[:]
        
        if best_len >= 500:
            break
    
    return best_words
