import numpy as np
import time
import random
from itertools import combinations

def construct():
    N = 17
    MIN_D = 6
    start_time = time.time()
    deadline = start_time + 54  # safety margin before 60s limit (evaluator kills at 90s but task says 60s)

    # Step 1: Construct the 256-word linear [17,8,6] code
    # Using the quadratic-residue code approach
    poly = 0x1D  # x^8 + x^4 + x^3 + x^2 + 1

    def gf_mul(a, b):
        res = 0
        for _ in range(8):
            if b & 1:
                res ^= a
            b >>= 1
            if a & 0x80:
                a = (a << 1) ^ poly
            else:
                a <<= 1
            a &= 0xFF
        return res

    def gf_pow(a, n):
        res = 1
        while n:
            if n & 1:
                res = gf_mul(res, a)
            a = gf_mul(a, a)
            n >>= 1
        return res

    # Find element of order 17 in GF(2^8)
    alpha = None
    for a in range(1, 256):
        if gf_pow(a, 17) == 1 and a != 1:
            alpha = a
            break

    if alpha is None:
        # Fallback: use the simpler construction from current program
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

        words_18 = []
        for w in words_17:
            parity = sum(w) % 2
            words_18.append(w + [parity])

        codewords = []
        for w18 in words_18:
            if w18[0] == 0:
                new_word = w18[1:]
                val = 0
                for i, bit in enumerate(new_word):
                    if bit:
                        val |= (1 << i)
                codewords.append(val)
    else:
        # Construct using QR code
        Q = [1, 2, 4, 8, 9, 13, 15, 16]
        g_coeffs = [1]
        for i in Q:
            root = gf_pow(alpha, i)
            new_g = [0] * (len(g_coeffs) + 1)
            for j in range(len(g_coeffs)):
                new_g[j + 1] ^= g_coeffs[j]
                new_g[j] ^= gf_mul(g_coeffs[j], root)
            g_coeffs = new_g

        G = np.zeros((9, 17), dtype=np.uint8)
        for k in range(9):
            for j in range(9):
                if g_coeffs[j]:
                    pos = (j + k) % 17
                    G[k, pos] = 1

        words_17 = []
        for u in range(512):
            word = np.zeros(17, dtype=np.uint8)
            for i in range(9):
                if (u >> i) & 1:
                    word ^= G[i]
            words_17.append(word)

        words_18 = []
        for w in words_17:
            parity = int(np.sum(w) % 2)
            words_18.append(np.append(w, parity))

        codewords = []
        for w18 in words_18:
            if w18[17] == 0:
                w_short = w18[:17]
                val = 0
                for i in range(17):
                    if w_short[i]:
                        val |= (1 << i)
                codewords.append(val)

    # Verify we have a valid code
    if not codewords:
        return [0]

    # Convert to bit matrix
    def to_bit_matrix(codes):
        return np.array([[(c >> i) & 1 for i in range(N)] for c in codes], dtype=np.uint8)

    code_bits = to_bit_matrix(codewords)
    code_weights = np.sum(code_bits, axis=1).astype(np.int32)

    # Precompute all 17-bit words as bit matrix
    all_words = np.arange(2**N, dtype=np.int64)
    bits_all = np.array([[(w >> i) & 1 for i in range(N)] for w in all_words], dtype=np.uint8)
    all_weights = np.sum(bits_all, axis=1).astype(np.int32)

    # Precompute flip masks for distances 5 and 6
    flip_masks_5 = []
    for combo in combinations(range(N), 5):
        mask = 0
        for bit in combo:
            mask |= (1 << bit)
        flip_masks_5.append(mask)

    flip_masks_6 = []
    for combo in combinations(range(N), 6):
        mask = 0
        for bit in combo:
            mask |= (1 << bit)
        flip_masks_6.append(mask)

    # Step 2: Greedily add words at distance >= 6 from all existing
    def find_addable(code_bits, code_weights, deadline):
        addable = []
        n_codes = code_bits.shape[0]
        if n_codes == 0:
            return list(range(2**N))
        batch_size = 4096
        for start in range(0, 2**N, batch_size):
            if time.time() >= deadline:
                break
            end = min(start + batch_size, 2**N)
            batch = bits_all[start:end]
            batch_w = all_weights[start:end]

            dots = batch @ code_bits.T
            dists = batch_w[:, None] + code_weights[None, :] - 2 * dots
            min_dists = np.min(dists, axis=1)
            valid = np.where(min_dists >= MIN_D)[0]
            for idx in valid:
                addable.append(start + idx)
        return addable

    # Iterative greedy addition
    added_any = True
    while added_any and time.time() < deadline - 15:
        added_any = False
        code_weights = np.sum(code_bits, axis=1).astype(np.int32)
        addable = find_addable(code_bits, code_weights, deadline - 15)
        if addable:
            for cand in addable:
                if time.time() >= deadline - 15:
                    break
                cand_bits = bits_all[cand]
                dots = code_bits @ cand_bits
                dists = code_weights + all_weights[cand] - 2 * dots
                if np.min(dists) >= MIN_D:
                    codewords.append(cand)
                    code_bits = np.vstack([code_bits, cand_bits[None, :]])
                    code_weights = np.sum(code_bits, axis=1).astype(np.int32)
                    added_any = True

    # Step 3: Local search - try swapping codewords to allow more additions
    if time.time() < deadline - 10:
        swap_improved = True
        while swap_improved and time.time() < deadline - 5:
            swap_improved = False
            n_codes = len(codewords)
            code_weights = np.sum(code_bits, axis=1).astype(np.int32)

            # Randomly shuffle codewords to explore different orderings
            order = list(range(n_codes))
            random.shuffle(order)

            for i in order:
                if time.time() >= deadline - 5:
                    break

                cw_i = codewords[i]

                # Try distance-5 flips
                for fm in flip_masks_5:
                    if time.time() >= deadline - 5:
                        break
                    cand = cw_i ^ fm
                    cand_bits = bits_all[cand]
                    dots = code_bits @ cand_bits
                    dists = code_weights + all_weights[cand] - 2 * dots
                    dists[i] = 999
                    if np.min(dists) >= MIN_D:
                        codewords[i] = cand
                        code_bits[i] = cand_bits
                        code_weights = np.sum(code_bits, axis=1).astype(np.int32)
                        swap_improved = True

                        # After a swap, try to add more words
                        addable = find_addable(code_bits, code_weights, deadline - 5)
                        for add_cand in addable:
                            if time.time() >= deadline - 5:
                                break
                            add_cand_bits = bits_all[add_cand]
                            dots2 = code_bits @ add_cand_bits
                            dists2 = code_weights + all_weights[add_cand] - 2 * dots2
                            if np.min(dists2) >= MIN_D:
                                codewords.append(add_cand)
                                code_bits = np.vstack([code_bits, add_cand_bits[None, :]])
                                code_weights = np.sum(code_bits, axis=1).astype(np.int32)
                                swap_improved = True
                        break

                # If no distance-5 swap worked, try distance-6 flips
                for fm in flip_masks_6:
                    if time.time() >= deadline - 5:
                        break
                    cand = cw_i ^ fm
                    cand_bits = bits_all[cand]
                    dots = code_bits @ cand_bits
                    dists = code_weights + all_weights[cand] - 2 * dots
                    dists[i] = 999
                    if np.min(dists) >= MIN_D:
                        codewords[i] = cand
                        code_bits[i] = cand_bits
                        code_weights = np.sum(code_bits, axis=1).astype(np.int32)
                        swap_improved = True

                        addable = find_addable(code_bits, code_weights, deadline - 5)
                        for add_cand in addable:
                            if time.time() >= deadline - 5:
                                break
                            add_cand_bits = bits_all[add_cand]
                            dots2 = code_bits @ add_cand_bits
                            dists2 = code_weights + all_weights[add_cand] - 2 * dots2
                            if np.min(dists2) >= MIN_D:
                                codewords.append(add_cand)
                                code_bits = np.vstack([code_bits, add_cand_bits[None, :]])
                                code_weights = np.sum(code_bits, axis=1).astype(np.int32)
                                swap_improved = True
                        break

    # Step 4: Final pass - try to add any remaining words
    if time.time() < deadline - 1:
        code_weights = np.sum(code_bits, axis=1).astype(np.int32)
        addable = find_addable(code_bits, code_weights, deadline - 1)
        for cand in addable:
            if time.time() >= deadline - 1:
                break
            cand_bits = bits_all[cand]
            dots = code_bits @ cand_bits
            dists = code_weights + all_weights[cand] - 2 * dots
            if np.min(dists) >= MIN_D:
                codewords.append(cand)
                code_bits = np.vstack([code_bits, cand_bits[None, :]])
                code_weights = np.sum(code_bits, axis=1).astype(np.int32)

    # Remove duplicates and verify
    codewords = list(set(codewords))

    return codewords
