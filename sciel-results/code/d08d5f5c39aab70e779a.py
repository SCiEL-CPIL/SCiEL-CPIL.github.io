import time
import numpy as np
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 55.0

    # Step 1: Construct the 256-word linear code [17,8,6]
    poly = 0x1D

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

    alpha = None
    for a in range(1, 256):
        if gf_pow(a, 17) == 1 and a != 1:
            alpha = a
            break

    if alpha is None:
        return [0]

    Q = [1, 2, 4, 8, 9, 13, 15, 16]

    g_coeffs = [1]
    for i in Q:
        root = gf_pow(alpha, i)
        new_g = [0] * (len(g_coeffs) + 1)
        for j in range(len(g_coeffs)):
            new_g[j + 1] ^= g_coeffs[j]
            new_g[j] ^= gf_mul(g_coeffs[j], root)
        g_coeffs = new_g

    G = np.zeros((9, 17), dtype=np.int64)
    for k in range(9):
        for j in range(9):
            if g_coeffs[j]:
                pos = (j + k) % 17
                G[k, pos] = 1

    words_18 = []
    for u in range(512):
        word = np.zeros(17, dtype=np.int64)
        for i in range(9):
            if (u >> i) & 1:
                word ^= G[i]
        parity = int(np.sum(word) % 2)
        word_ext = np.append(word, parity)
        words_18.append(word_ext)

    words_short = []
    for w in words_18:
        if w[17] == 0:
            w_short = w[:17]
            val = 0
            for i in range(17):
                if w_short[i]:
                    val |= (1 << i)
            words_short.append(val)

    if not words_short:
        return [0]

    # Precompute popcount table
    popcount_table = np.array([i.bit_count() for i in range(2**17)], dtype=np.int8)

    # Base code
    base_code = np.array(words_short, dtype=np.int64)
    best_code = base_code.copy()
    best_size = len(base_code)

    all_words = np.arange(2**17, dtype=np.int64)

    # Precompute all words at each distance from 0 (and by symmetry from any codeword)
    # Words at distance exactly d from 0
    dist6_from_zero = np.array([w for w in range(2**17) if popcount_table[w] == 6], dtype=np.int64)
    dist5_from_zero = np.array([w for w in range(2**17) if popcount_table[w] == 5], dtype=np.int64)
    dist7_from_zero = np.array([w for w in range(2**17) if popcount_table[w] == 7], dtype=np.int64)

    # Helper: generate all words at distance exactly d from a given word
    def words_at_dist_from(word, d, popcount_table, all_words):
        """Return all words at Hamming distance exactly d from word."""
        diffs = all_words ^ word
        dists = popcount_table[diffs]
        return all_words[dists == d]

    # Step 2: Greedy extension using focused candidate pool
    forbidden = np.zeros(2**17, dtype=bool)
    in_code = np.zeros(2**17, dtype=bool)

    # Initialize forbidden set from base code
    for w in base_code:
        diffs = all_words ^ w
        dists = popcount_table[diffs]
        forbidden[dists < 6] = True
    in_code[base_code] = True

    code_list = list(base_code)
    current_size = len(code_list)

    # Build initial candidate pool: words at distance 6 from zero word (which is in the code)
    # These are the most promising candidates for a nonlinear code
    # Also include words at distance 6 from other low-weight codewords
    candidate_pool_set = set()
    # Add words at distance 6 from zero
    for w in dist6_from_zero:
        w_int = int(w)
        if not in_code[w_int] and not forbidden[w_int]:
            candidate_pool_set.add(w_int)

    # Also add words at distance 6 from a few other codewords to expand the pool
    for w in base_code[:16]:
        w_int = int(w)
        diffs = all_words ^ w_int
        dists = popcount_table[diffs]
        for cw in all_words[dists == 6]:
            cw_int = int(cw)
            if not in_code[cw_int] and not forbidden[cw_int]:
                candidate_pool_set.add(cw_int)

    candidate_pool = np.array(list(candidate_pool_set), dtype=np.int64)

    # Greedy: try to add words from the focused pool in random order
    for restart in range(50):
        if time.time() > deadline - 10.0:
            break

        rng = np.random.default_rng(seed=restart * 12345 + 67890)

        if len(candidate_pool) > 0:
            perm = rng.permutation(len(candidate_pool))
            for idx in perm:
                if time.time() > deadline - 10.0:
                    break
                w = int(candidate_pool[idx])
                if in_code[w] or forbidden[w]:
                    continue
                # w can be added
                code_list.append(w)
                in_code[w] = True
                current_size += 1

                # Update forbidden set
                diffs = all_words ^ w
                dists = popcount_table[diffs]
                forbidden[dists < 6] = True

                # Expand candidate pool: add words at distance 6 from newly added word
                new_candidates = all_words[dists == 6]
                for nc in new_candidates:
                    nc_int = int(nc)
                    if not in_code[nc_int] and not forbidden[nc_int]:
                        candidate_pool_set.add(nc_int)

                # Rebuild candidate pool periodically to avoid memory issues
                if len(candidate_pool_set) > 100000:
                    candidate_pool = np.array(list(candidate_pool_set), dtype=np.int64)

        if current_size > best_size:
            best_size = current_size
            best_code = np.array(code_list, dtype=np.int64)

        # If we exhausted the pool, try a broader search
        if len(candidate_pool) > 0 and all(in_code[int(w)] or forbidden[int(w)] for w in candidate_pool[:100]):
            # Fall back to broader random search
            rng2 = np.random.default_rng(seed=restart * 777 + 333)
            broad_candidates = rng2.permutation(all_words)
            for w in broad_candidates:
                if time.time() > deadline - 10.0:
                    break
                w = int(w)
                if in_code[w] or forbidden[w]:
                    continue
                code_list.append(w)
                in_code[w] = True
                current_size += 1
                diffs = all_words ^ w
                dists = popcount_table[diffs]
                forbidden[dists < 6] = True
                if current_size > best_size:
                    best_size = current_size
                    best_code = np.array(code_list, dtype=np.int64)

    # Step 3: Local search - try swaps to enable more additions
    if time.time() < deadline - 5.0:
        code_list = list(best_code)
        in_code = np.zeros(2**17, dtype=bool)
        in_code[code_list] = True
        forbidden = np.zeros(2**17, dtype=bool)
        for w in code_list:
            diffs = all_words ^ w
            dists = popcount_table[diffs]
            forbidden[dists < 6] = True

        # Rebuild focused candidate pool for the current best code
        candidate_pool_set = set()
        for w in code_list:
            w_int = int(w)
            diffs = all_words ^ w_int
            dists = popcount_table[diffs]
            for cw in all_words[dists == 6]:
                cw_int = int(cw)
                if not in_code[cw_int] and not forbidden[cw_int]:
                    candidate_pool_set.add(cw_int)
        candidate_pool = np.array(list(candidate_pool_set), dtype=np.int64)

        for swap_round in range(20):
            if time.time() > deadline - 5.0:
                break

            # Try to add any remaining words from the focused pool
            added_any = False
            rng = np.random.default_rng(seed=swap_round * 999 + 1)
            if len(candidate_pool) > 0:
                perm = rng.permutation(len(candidate_pool))
                for idx in perm:
                    if time.time() > deadline - 5.0:
                        break
                    w = int(candidate_pool[idx])
                    if in_code[w] or forbidden[w]:
                        continue
                    code_list.append(w)
                    in_code[w] = True
                    added_any = True
                    diffs = all_words ^ w
                    dists = popcount_table[diffs]
                    forbidden[dists < 6] = True
                    # Expand pool
                    for nc in all_words[dists == 6]:
                        nc_int = int(nc)
                        if not in_code[nc_int] and not forbidden[nc_int]:
                            candidate_pool_set.add(nc_int)

            if len(code_list) > best_size:
                best_size = len(code_list)
                best_code = np.array(code_list, dtype=np.int64)

            if not added_any:
                # Try swaps: remove one word, see if we can add 2+
                rng = np.random.default_rng(seed=swap_round * 55555 + 2)
                for swap_attempt in range(30):
                    if time.time() > deadline - 5.0:
                        break

                    idx = int(rng.integers(0, len(code_list)))
                    removed = code_list[idx]
                    code_list_new = code_list[:idx] + code_list[idx+1:]
                    in_code[removed] = False

                    # Recompute forbidden set without removed word (incremental: just clear and re-add)
                    # For efficiency, recompute from scratch but only for the relevant subset
                    forbidden_new = np.zeros(2**17, dtype=bool)
                    for w in code_list_new:
                        diffs = all_words ^ w
                        dists = popcount_table[diffs]
                        forbidden_new[dists < 6] = True

                    # Try to add words from focused pool around removed word
                    added_count = 0
                    # Candidates: words at distance 6 from the removed word
                    diffs_rm = all_words ^ removed
                    dists_rm = popcount_table[diffs_rm]
                    candidates2 = all_words[dists_rm == 6]
                    rng2 = np.random.default_rng(seed=swap_round * 111 + swap_attempt)
                    perm2 = rng2.permutation(len(candidates2))
                    for idx2 in perm2:
                        if time.time() > deadline - 5.0:
                            break
                        w = int(candidates2[idx2])
                        if in_code[w] or forbidden_new[w]:
                            continue
                        code_list_new.append(w)
                        in_code[w] = True
                        added_count += 1
                        diffs = all_words ^ w
                        dists = popcount_table[diffs]
                        forbidden_new[dists < 6] = True
                        if added_count >= 2:
                            break

                    if added_count >= 2:
                        code_list = code_list_new
                        forbidden = forbidden_new
                        if len(code_list) > best_size:
                            best_size = len(code_list)
                            best_code = np.array(code_list, dtype=np.int64)
                    else:
                        # Restore
                        in_code[removed] = True

    return list(best_code)
