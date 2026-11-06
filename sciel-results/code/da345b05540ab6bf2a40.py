import numpy as np
import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 82.0  # Safety margin before 90s limit

    N = 17
    MIN_DIST = 6

    # GF(256) construction for the QR code.
    # Primitive polynomial: x^8 + x^4 + x^3 + x^2 + 1 (0x11D)

    def gf_mul(a, b, poly=0x11D):
        res = 0
        while b:
            if b & 1:
                res ^= a
            b >>= 1
            a <<= 1
            if a & 0x100:
                a ^= poly
        return res & 0xFF

    def gf_pow(a, n, poly=0x11D):
        res = 1
        base = a
        while n:
            if n & 1:
                res = gf_mul(res, base, poly)
            base = gf_mul(base, base, poly)
            n >>= 1
        return res

    def get_primitive_17th_root():
        # 255 = 15 * 17
        for candidate in range(2, 256):
            if gf_pow(candidate, 17) == 1 and candidate != 1:
                return candidate
        return None

    beta = get_primitive_17th_root()
    current_code = None

    if beta:
        # Build the generator polynomial
        poly_coeffs = [1]
        current_beta = beta
        for _ in range(8):
            new_coeffs = [0] * (len(poly_coeffs) + 1)
            for i in range(len(poly_coeffs)):
                new_coeffs[i+1] = poly_coeffs[i]
            for i in range(len(poly_coeffs)):
                new_coeffs[i] ^= gf_mul(poly_coeffs[i], current_beta)
            poly_coeffs = new_coeffs
            current_beta = gf_mul(current_beta, current_beta)

        g_bits = []
        for c in poly_coeffs:
            if c == 1:
                g_bits.append(1)
            elif c == 0:
                g_bits.append(0)
            else:
                g_bits.append(1)

        if len(g_bits) == 9 and g_bits[0] == 1:
            # Generate [17, 9] code
            codewords_17 = []
            for u in range(512):
                c = 0
                for k in range(9):
                    if (u >> k) & 1:
                        for j in range(9):
                            if g_bits[j]:
                                c ^= 1 << (j + k)
                codewords_17.append(c)

            # Extend to 18
            extended_codewords = []
            for c in codewords_17:
                parity = bin(c).count('1') % 2
                ec = c | (parity << 17)
                extended_codewords.append(ec)

            # Shorten (fix bit 17 to 0)
            shortened_codewords = []
            for ec in extended_codewords:
                if (ec >> 17) & 1 == 0:
                    shortened_codewords.append(ec & ((1 << 17) - 1))

            # Remove duplicates
            shortened_codewords = list(set(shortened_codewords))

            # Verify
            if len(shortened_codewords) >= 256:
                min_d = 100
                for i in range(len(shortened_codewords)):
                    for j in range(i+1, len(shortened_codewords)):
                        d = (shortened_codewords[i] ^ shortened_codewords[j]).bit_count()
                        if d < min_d:
                            min_d = d
                if min_d >= 6:
                    current_code = set(shortened_codewords)

    if current_code is None:
        # Fallback: start with zero word and greedy
        current_code = set()
        current_code.add(0)

    # Convert to numpy array for efficient distance computations
    code_list = list(current_code)
    code_array = np.array(code_list, dtype=np.uint32)

    def min_dist_to_code(candidate, code_array):
        """Compute minimum Hamming distance from candidate to all codewords."""
        xor_vals = code_array ^ candidate
        # Use a lookup table for bit counting on 17-bit values
        # Split into bytes for faster counting
        # Actually, let's use a vectorized approach
        # Count bits by summing over bit positions
        # This is O(N) per candidate but vectorized
        total = np.zeros(len(code_array), dtype=np.uint8)
        for b in range(N):
            total += ((xor_vals >> b) & 1).astype(np.uint8)
        return total.min()

    def min_dist_to_code_fast(candidate, code_array, bit_positions=None):
        """Faster min distance using precomputed bit masks."""
        if bit_positions is None:
            bit_positions = [1 << b for b in range(N)]
        xor_vals = code_array ^ candidate
        total = np.zeros(len(code_array), dtype=np.uint8)
        for mask in bit_positions:
            total += ((xor_vals & mask) != 0).astype(np.uint8)
        return total.min()

    bit_positions = [1 << b for b in range(N)]

    # Simulated Annealing approach to add words
    random.seed(42)
    np.random.seed(42)

    best_code = set(code_list)
    best_size = len(best_code)

    # Phase 1: Try to add random words with repair
    # For each attempt, pick a random candidate, check if it conflicts
    # If it conflicts, try to repair by modifying conflicting codewords

    add_attempts = 0
    max_add_attempts = 50000

    while time.time() < deadline - 5.0 and add_attempts < max_add_attempts:
        add_attempts += 1

        # Pick a random candidate
        cand = random.randint(0, (1 << N) - 1)
        if cand in best_code:
            continue

        # Check distance to all codewords
        cand_arr = np.array([cand], dtype=np.uint32)
        xor_vals = code_array ^ cand
        total = np.zeros(len(code_array), dtype=np.uint8)
        for mask in bit_positions:
            total += ((xor_vals & mask) != 0).astype(np.uint8)

        min_d = total.min()
        if min_d >= MIN_DIST:
            # Can add directly
            best_code.add(cand)
            code_array = np.array(list(best_code), dtype=np.uint32)
            best_size += 1
            continue

        # If min_d < MIN_DIST, try to repair
        # Find the closest codeword(s)
        # Try to flip bits in the candidate or in existing codewords

        # Strategy: try small modifications to the candidate
        # Flip 1 or 2 bits in the candidate and re-check
        found = False
        for _ in range(20):
            # Randomly flip 1-2 bits
            flips = random.sample(range(N), random.randint(1, 2))
            new_cand = cand
            for f in flips:
                new_cand ^= (1 << f)
            if new_cand in best_code:
                continue

            # Check distance
            new_cand_arr = np.array([new_cand], dtype=np.uint32)
            xor_vals2 = code_array ^ new_cand
            total2 = np.zeros(len(code_array), dtype=np.uint8)
            for mask in bit_positions:
                total2 += ((xor_vals2 & mask) != 0).astype(np.uint8)
            if total2.min() >= MIN_DIST:
                best_code.add(new_cand)
                code_array = np.array(list(best_code), dtype=np.uint32)
                best_size += 1
                found = True
                break

        if found:
            continue

    # Phase 2: Local search - try to replace pairs of codewords with better ones
    # Or try to add by removing one and adding two

    # Simulated annealing: try to add a word, and if it conflicts,
    # try to move (swap bits in) existing codewords to make room

    sa_iterations = 0
    max_sa_iterations = 200000
    temperature = 1.0

    while time.time() < deadline - 2.0 and sa_iterations < max_sa_iterations:
        sa_iterations += 1

        # Pick a random candidate
        cand = random.randint(0, (1 << N) - 1)
        if cand in best_code:
            continue

        # Check distance to all codewords
        cand_arr = np.array([cand], dtype=np.uint32)
        xor_vals = code_array ^ cand
        total = np.zeros(len(code_array), dtype=np.uint8)
        for mask in bit_positions:
            total += ((xor_vals & mask) != 0).astype(np.uint8)

        min_d = int(total.min())
        if min_d >= MIN_DIST:
            best_code.add(cand)
            code_array = np.array(list(best_code), dtype=np.uint32)
            best_size += 1
            continue

        # Try to repair: find a codeword to modify
        # Pick a random existing codeword and try flipping 1-3 bits
        # Check if the modified codeword is still valid with all others
        # and if cand becomes valid

        code_list_now = list(best_code)
        idx = random.randint(0, len(code_list_now) - 1)
        old_word = code_list_now[idx]

        # Try modifications to old_word
        success = False
        for _ in range(5):
            flips = random.sample(range(N), random.randint(1, 3))
            new_word = old_word
            for f in flips:
                new_word ^= (1 << f)

            if new_word == old_word or new_word in best_code:
                continue

            # Check new_word against all other codewords (excluding old_word)
            valid = True
            for w in best_code:
                if w == old_word:
                    continue
                if (new_word ^ w).bit_count() < MIN_DIST:
                    valid = False
                    break
            if not valid:
                continue

            # Check cand against all codewords with old_word replaced by new_word
            valid_cand = True
            for w in best_code:
                if w == old_word:
                    if (cand ^ new_word).bit_count() < MIN_DIST:
                        valid_cand = False
                        break
                else:
                    if (cand ^ w).bit_count() < MIN_DIST:
                        valid_cand = False
                        break
            if valid_cand:
                # Accept the swap
                best_code.remove(old_word)
                best_code.add(new_word)
                best_code.add(cand)
                code_array = np.array(list(best_code), dtype=np.uint32)
                best_size += 1
                success = True
                break

        if success:
            temperature *= 0.99

    # Phase 3: Final greedy add
    random.seed(42)
    final_adds = 0
    max_final_adds = 100000
    final_attempts = 0

    while time.time() < deadline - 1.0 and final_attempts < max_final_adds:
        final_attempts += 1
        cand = random.randint(0, (1 << N) - 1)
        if cand in best_code:
            continue

        # Quick check
        valid = True
        for w in best_code:
            if (cand ^ w).bit_count() < MIN_DIST:
                valid = False
                break
        if valid:
            best_code.add(cand)
            final_adds += 1

    return list(best_code)
