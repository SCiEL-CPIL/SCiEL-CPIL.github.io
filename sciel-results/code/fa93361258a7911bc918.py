import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin for 60s limit

    # Pre-compute popcount table for 17-bit integers
    popcount = [0] * (1 << 17)
    for i in range(1, 1 << 17):
        popcount[i] = popcount[i >> 1] + (i & 1)

    def popcount_fast(x):
        return popcount[x & 0x1FFFF]

    def is_valid_code(words):
        for i in range(len(words)):
            wi = words[i]
            for j in range(i + 1, len(words)):
                if popcount_fast(wi ^ words[j]) < 6:
                    return False
        return True

    # Step 1: Build the [17,8,6] linear code (256 words)
    # This is the shortened version of the [18,9,6] extended QR(17) code.
    
    gf_poly = 0x1d  # x^8+x^4+x^3+x^2+1, primitive in GF(2^8)
    
    def gf_mul(a, b):
        res = 0
        for _ in range(8):
            if b & 1:
                res ^= a
            b >>= 1
            if a & 0x80:
                a = ((a << 1) ^ gf_poly) & 0xFF
            else:
                a = (a << 1) & 0xFF
        return res

    def gf_pow(a, n):
        res = 1
        while n:
            if n & 1:
                res = gf_mul(res, a)
            a = gf_mul(a, a)
            n >>= 1
        return res

    # Find element of order 17
    alpha = None
    for candidate in range(1, 256):
        if gf_pow(candidate, 17) == 1 and candidate != 1:
            alpha = candidate
            break

    if alpha is None:
        return [0]

    # Quadratic residues mod 17
    Q = [1, 2, 4, 8, 9, 13, 15, 16]
    
    # Build generator polynomial g(x) of degree 8: g(x) = prod_{i in Q} (x - alpha^i)
    g = [1]
    for i in Q:
        root = gf_pow(alpha, i)
        new_g = [0] * (len(g) + 1)
        for j in range(len(g)):
            new_g[j] ^= gf_mul(g[j], root)
            new_g[j + 1] ^= g[j]
        g = new_g

    # Generate all 512 words of the [17,9,5] code
    words_17_9_5 = []
    for m in range(512):
        c = [0] * 17
        for j in range(9):
            if (m >> j) & 1:
                for k in range(9):
                    if (g[k] != 0):
                        pos = (j + k) % 17
                        c[pos] ^= 1
        val = 0
        for bit in c:
            val = (val << 1) | bit
        words_17_9_5.append(val)

    # Extend to [18,9,6] by appending overall parity
    words_18_9_6 = []
    for w in words_17_9_5:
        parity = popcount_fast(w) % 2
        ext_val = w | (parity << 17)
        words_18_9_6.append(ext_val)

    # Shorten: keep words with parity bit = 0, then mask off the 18th coordinate.
    words_17_8_6 = []
    for w in words_18_9_6:
        if (w >> 17) & 1 == 0:
            w_short = w & 0x1FFFF
            words_17_8_6.append(w_short)

    # Deduplicate
    words_17_8_6 = list(set(words_17_8_6))
    base_code = words_17_8_6

    # Verify the base code
    if not is_valid_code(base_code):
        valid_code = [base_code[0]]
        for w in base_code[1:]:
            ok = True
            for v in valid_code:
                if popcount_fast(w ^ v) < 6:
                    ok = False
                    break
            if ok:
                valid_code.append(w)
        base_code = valid_code

    current_code = list(base_code)
    best_code = list(base_code)
    best_size = len(current_code)
    if best_size < 32:
        return [0]

    # Step 2: Greedy extension with targeted candidates
    # Use a set for fast membership checks
    word_set = set(current_code)
    
    def is_valid_candidate(w):
        """Check if w has distance >= 6 from all existing words."""
        for v in current_code:
            if popcount_fast(w ^ v) < 6:
                return False
        return True

    # Strategy 1: Try complements of existing words
    for w in list(current_code):
        if time.time() > deadline - 5.0:
            break
        comp = w ^ 0x1FFFF
        if comp not in word_set:
            if is_valid_candidate(comp):
                current_code.append(comp)
                word_set.add(comp)
                if len(current_code) > best_size:
                    best_size = len(current_code)
                    best_code = list(current_code)

    # Strategy 2: Try words at exactly distance 6 from existing words
    # This is more targeted - generate candidates by flipping exactly 6 bits
    import itertools
    bits = list(range(17))
    
    # Precompute all masks of weight 6 (C(17,6) = 12376 masks)
    weight6_masks = []
    for combo in itertools.combinations(bits, 6):
        mask = 0
        for b in combo:
            mask |= (1 << b)
        weight6_masks.append(mask)
    
    # Sample existing words and try adding words at distance 6 from them
    sample_size = min(50, len(current_code))
    sampled_words = random.sample(current_code, sample_size)
    
    for w in sampled_words:
        if time.time() > deadline - 3.0:
            break
        for mask in weight6_masks:
            if time.time() > deadline - 3.0:
                break
            candidate = w ^ mask
            if candidate not in word_set:
                if is_valid_candidate(candidate):
                    current_code.append(candidate)
                    word_set.add(candidate)
                    if len(current_code) > best_size:
                        best_size = len(current_code)
                        best_code = list(current_code)

    # Strategy 3: Random search for additional words
    random.seed(42)
    attempts = 0
    max_attempts = 200000
    while time.time() < deadline - 2.0 and attempts < max_attempts:
        candidate = random.getrandbits(17)
        if candidate not in word_set:
            if is_valid_candidate(candidate):
                current_code.append(candidate)
                word_set.add(candidate)
                if len(current_code) > best_size:
                    best_size = len(current_code)
                    best_code = list(current_code)
        attempts += 1

    # Final verification and cleanup
    if len(best_code) >= 32:
        if is_valid_code(best_code):
            return best_code
        else:
            # Filter it down
            final = []
            for w in best_code:
                ok = True
                for v in final:
                    if popcount_fast(w ^ v) < 6:
                        ok = False
                        break
                if ok:
                    final.append(w)
            if len(final) >= 32:
                return final
            return base_code
    else:
        return base_code if len(base_code) >= 32 else [0]
