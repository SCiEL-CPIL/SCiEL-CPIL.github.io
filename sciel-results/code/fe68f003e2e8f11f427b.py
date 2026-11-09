import time
import numpy as np

def construct():
    N = 18
    MIN_DIST = 6
    TOTAL_BUDGET = 55.0
    t_start = time.time()
    deadline = t_start + TOTAL_BUDGET

    # Pre-compute popcount for all 18-bit values for fast fallback
    # 2^18 = 262144, fits in memory easily
    popcount_all = np.array([i.bit_count() for i in range(2**N)], dtype=np.uint8)

    def popcount18(arr):
        """
        arr: numpy array of uint32 (or int) values < 2^18
        Returns: numpy array of popcounts
        """
        # Convert to intp for indexing if necessary, but popcount_all is indexed by int
        # Ensure arr is int array
        return popcount_all[arr.astype(np.intp)]

    # ============================================================
    # Construction 1: Extended Quadratic Residue Code [18,9,6]
    # ============================================================
    # Generator polynomial for QR(17) [17,9,5] code
    # g(x) = x^8 + x^7 + x^5 + x^4 + x^3 + x + 1
    # Bits set at positions: 0, 1, 3, 4, 5, 7, 8
    g_poly = 0
    for bit_pos in [0, 1, 3, 4, 5, 7, 8]:
        g_poly |= (1 << bit_pos)

    def poly_mul_17(m, g):
        """Multiply m(x) * g(x) mod (x^17 - 1). m < 2^9, g < 2^9."""
        result = 0
        # Since deg(m) < 9 and deg(g) < 9, deg(product) < 17.
        # No reduction mod x^17-1 needed for the bits themselves, 
        # but we mask to 17 bits just in case.
        for i in range(9):
            if m & (1 << i):
                result ^= (g << i)
        return result & 0x1FFFF

    qr_codewords_17 = []
    for m in range(512):
        cw = poly_mul_17(m, g_poly)
        qr_codewords_17.append(cw)

    # Extend to length 18 by appending parity bit
    qr_codewords_18 = []
    for cw in qr_codewords_17:
        parity = cw.bit_count() % 2
        extended = cw | (parity << 17)
        qr_codewords_18.append(extended)

    # Verify the QR construction
    if len(qr_codewords_18) == 512:
        # Check min weight (linear code property)
        min_w = min(cw.bit_count() for cw in qr_codewords_18 if cw != 0)
        
        if min_w >= MIN_DIST:
            # Full pairwise verification to be absolutely sure (512 words is small)
            valid = True
            words_arr = np.array(qr_codewords_18, dtype=np.uint32)
            for i in range(len(words_arr)):
                if time.time() > deadline - 5.0:
                    valid = False
                    break
                xor_vals = np.bitwise_xor(words_arr, np.uint32(words_arr[i]))
                bit_counts = popcount18(xor_vals)
                if bit_counts.min() < MIN_DIST:
                    valid = False
                    break
            
            if valid:
                return qr_codewords_18

    # ============================================================
    # Fallback: Optimized Greedy Search
    # ============================================================
    # If QR construction failed or wasn't valid, use greedy search.
    # We use the pre-computed popcount array for speed.
    
    # Sort candidates by weight, then by value
    # We exclude 0 as we start with it
    all_candidates = list(range(1, 2**N))
    all_candidates.sort(key=lambda x: (popcount_all[x], x))

    best_words = [0]
    best_len = 1
    
    rng = np.random.default_rng(42)
    
    # Try a few shuffles within weight classes to find better local optima
    for attempt in range(50):
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
                while j < len(candidates) and popcount_all[candidates[j]] == popcount_all[candidates[i]]:
                    j += 1
                rng.shuffle(candidates[i:j])
                i = j
        
        words_list = [0]
        words_arr = np.array(words_list, dtype=np.uint32)
        
        for c in candidates:
            if time.time() > deadline - 5.0:
                break
            
            c_val = np.uint32(c)
            xor_vals = np.bitwise_xor(words_arr, c_val)
            bit_counts = popcount18(xor_vals)
            
            if bit_counts.min() >= MIN_DIST:
                words_list.append(c)
                # Update words_arr for next iteration
                # To avoid frequent array resizing/copying, we can just keep the list
                # and convert to array only when checking, but that's O(N) per check.
                # Better: maintain a list, and for the check, create the array.
                # Since N is small (max 512), this is fast.
                words_arr = np.array(words_list, dtype=np.uint32)
        
        if len(words_list) > best_len:
            best_len = len(words_list)
            best_words = words_list[:]
            
        if best_len >= 512:
            break

    return best_words
