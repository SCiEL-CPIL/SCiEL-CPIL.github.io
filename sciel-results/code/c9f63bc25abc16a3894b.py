import time
import numpy as np

def construct():
    N = 18
    MIN_DIST = 6
    TOTAL_BUDGET = 55.0
    t_start = time.time()
    deadline = t_start + TOTAL_BUDGET

    # Precompute popcount lookup for 9-bit values
    pop9 = np.zeros(512, dtype=np.uint8)
    for i in range(512):
        pop9[i] = i.bit_count()

    def fast_popcount18(arr):
        """Vectorized popcount for numpy array of 18-bit values"""
        low = (arr & 0x1FF).astype(np.intp)
        high = ((arr >> 9) & 0x1FF).astype(np.intp)
        return pop9[low] + pop9[high]

    # Generate RM(1,4) codewords of length 16
    rm16 = []
    for mask in range(32):
        word = 0
        for pos in range(16):
            x1 = pos & 1
            x2 = (pos >> 1) & 1
            x3 = (pos >> 2) & 1
            x4 = (pos >> 3) & 1
            bit = 0
            if mask & 1: bit ^= 1
            if mask & 2: bit ^= x1
            if mask & 4: bit ^= x2
            if mask & 8: bit ^= x3
            if mask & 16: bit ^= x4
            if bit:
                word |= (1 << pos)
        rm16.append(word)

    # Try all 4 possible 2-bit extensions, pick the one that allows most greedy additions
    best_words = None
    best_count = 0

    for ext in range(4):
        if time.time() > deadline - 5.0:
            break

        # Build 18-bit words from RM(1,4) with 2-bit extension
        seed_words = [w | (ext << 16) for w in rm16]
        seed_set = set(seed_words)

        # Generate candidates sorted by weight
        all_candidates = list(range(2**N))
        all_candidates.sort(key=lambda x: (x.bit_count(), x))
        candidates = [c for c in all_candidates if c not in seed_set]

        words = list(seed_words)
        words_set = seed_set
        words_arr = np.array(words, dtype=np.uint32)

        for c in candidates:
            if time.time() > deadline - 2.0:
                break

            c_val = np.uint32(c)
            xor_vals = np.bitwise_xor(words_arr, c_val).astype(np.uint32)
            bit_counts = fast_popcount18(xor_vals)

            if np.min(bit_counts) < MIN_DIST:
                continue

            words.append(c)
            words_set.add(c)
            words_arr = np.append(words_arr, np.uint32(c))

        if len(words) > best_count:
            best_count = len(words)
            best_words = words

    if best_words is None:
        best_words = [w | 0 for w in rm16]

    return best_words
