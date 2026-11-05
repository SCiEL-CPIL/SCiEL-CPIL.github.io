import time
import numpy as np

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

    # Base code as numpy array
    base_code = np.array(words_short, dtype=np.int64)
    best_code = base_code.copy()
    best_size = len(base_code)

    # All possible 17-bit words
    all_words = np.arange(2**17, dtype=np.int64)

    # Use a boolean lookup array for fast membership testing
    in_code = np.zeros(2**17, dtype=bool)
    in_code[base_code] = True

    # Step 2: Vectorized greedy extension
    num_restarts = 40
    batch_size = 10000

    for restart in range(num_restarts):
        if time.time() > deadline - 3.0:
            break

        rng = np.random.default_rng(seed=restart * 12345 + 67890)
        candidates = rng.permutation(all_words)

        # Reset in_code for this restart
        in_code[:] = False
        in_code[base_code] = True

        current_array = base_code.copy()
        current_size = len(current_array)

        for batch_start in range(0, len(candidates), batch_size):
            if time.time() > deadline - 2.0:
                break

            batch_end = min(batch_start + batch_size, len(candidates))
            batch = candidates[batch_start:batch_end]

            # Vectorized filter: exclude words already in code
            valid_mask = ~in_code[batch]
            if not np.any(valid_mask):
                continue
            batch = batch[valid_mask]

            if len(batch) == 0:
                continue

            # Process in sub-batches to manage memory
            sub_batch_size = 4000
            newly_added = []
            for sb_start in range(0, len(batch), sub_batch_size):
                if time.time() > deadline - 2.0:
                    break
                sb_end = min(sb_start + sub_batch_size, len(batch))
                sub_batch = batch[sb_start:sb_end]

                # Compute distance matrix
                diffs = sub_batch[:, None] ^ current_array[None, :]
                dists = popcount_table[diffs]
                min_dists = np.min(dists, axis=1)

                valid = min_dists >= 6
                if np.any(valid):
                    valid_words = sub_batch[valid]
                    newly_added.extend(valid_words.tolist())

            if newly_added:
                new_arr = np.array(newly_added, dtype=np.int64)
                current_array = np.concatenate([current_array, new_arr])
                in_code[new_arr] = True
                current_size += len(new_arr)

        if current_size > best_size:
            best_size = current_size
            best_code = current_array.copy()

    # Step 3: Local search with swaps
    if time.time() < deadline - 5.0:
        for restart in range(15):
            if time.time() > deadline - 3.0:
                break

            rng = np.random.default_rng(seed=restart * 99999 + 1)
            code_list = list(best_code)
            code_set = set(code_list)
            code_array = np.array(code_list, dtype=np.int64)

            for attempt in range(30):
                if time.time() > deadline - 3.0:
                    break

                # Try to add any word that fits
                # Sample candidates not in code
                not_in_code = all_words[~in_code[all_words]] if len(code_array) < 2**17 else []
                if len(not_in_code) == 0:
                    break

                sample_size = min(3000, len(not_in_code))
                sample = rng.choice(not_in_code, size=sample_size, replace=False)

                added_any = False
                for w in sample:
                    if time.time() > deadline - 3.0:
                        break
                    diffs = code_array ^ w
                    dists = popcount_table[diffs]
                    if np.min(dists) >= 6:
                        code_list.append(int(w))
                        code_set.add(int(w))
                        code_array = np.append(code_array, w)
                        in_code[int(w)] = True
                        added_any = True

                if added_any and len(code_list) > best_size:
                    best_size = len(code_list)
                    best_code = code_array.copy()

                # Try swap: remove a random word, see if we can add 2+ new ones
                if time.time() > deadline - 3.0:
                    break
                if len(code_list) > 1:
                    idx = int(rng.integers(0, len(code_list)))
                    removed = code_list[idx]
                    # Temporarily remove
                    code_list_new = code_list[:idx] + code_list[idx+1:]
                    code_array_new = np.array(code_list_new, dtype=np.int64)
                    in_code[removed] = False

                    # Try to add words
                    not_in_code2 = all_words[~in_code[all_words]]
                    if len(not_in_code2) > 0:
                        sample_size2 = min(3000, len(not_in_code2))
                        sample2 = rng.choice(not_in_code2, size=sample_size2, replace=False)

                        added_count = 0
                        for w in sample2:
                            if time.time() > deadline - 3.0:
                                break
                            diffs = code_array_new ^ w
                            dists = popcount_table[diffs]
                            if np.min(dists) >= 6:
                                code_list_new.append(int(w))
                                code_set.add(int(w))
                                code_array_new = np.append(code_array_new, w)
                                in_code[int(w)] = True
                                added_count += 1
                                if added_count >= 2:
                                    break

                        # Only accept swap if we gained net words
                        if added_count >= 2:
                            code_list = code_list_new
                            code_array = code_array_new
                            if len(code_list) > best_size:
                                best_size = len(code_list)
                                best_code = code_array.copy()
                        else:
                            # Restore
                            code_list = list(best_code)
                            code_set = set(code_list)
                            code_array = np.array(code_list, dtype=np.int64)
                            in_code[:] = False
                            in_code[best_code] = True
                            in_code[removed] = True

    return list(best_code)
