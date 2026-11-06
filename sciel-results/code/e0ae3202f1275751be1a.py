import numpy as np
import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 57.0  # Safety margin before 90s kill

    N = 17
    MIN_DIST = 6

    # Quadratic residues mod 19
    qres = set()
    for x in range(1, 19):
        qres.add((x * x) % 19)

    def build_linear_code():
        G = np.zeros((9, 18), dtype=np.uint8)
        for i in range(9):
            for j in range(17):
                pos = (j - i) % 19
                if pos in qres:
                    G[i, j] = 1
            G[i, 17] = G[i, :17].sum() % 2
        codewords = set()
        for mask in range(256):
            cw = np.zeros(18, dtype=np.uint8)
            for i in range(9):
                if (mask >> i) & 1:
                    cw = (cw + G[i]) % 2
            val = 0
            for j in range(18):
                val = (val << 1) | int(cw[j])
            codewords.add(val)
        return codewords, G

    all_cw_18, G = build_linear_code()

    # Shorten by removing coordinate 17 (parity/LSB), keep words with 0 there
    cw_17 = set()
    for val in all_cw_18:
        if (val & 1) == 0:
            cw_17.add(val >> 1)

    cw_list = list(cw_17)
    min_d = 999
    for i in range(len(cw_list)):
        for j in range(i + 1, len(cw_list)):
            d = (cw_list[i] ^ cw_list[j]).bit_count()
            if d < min_d:
                min_d = d

    # Try all shortening coordinates if needed
    if min_d < 6:
        best_shortened = None
        best_min_d = 0
        for coord in range(18):
            cw_short = set()
            for val in all_cw_18:
                if ((val >> (17 - coord)) & 1) == 0:
                    upper = val >> (18 - coord)
                    lower = val & ((1 << (17 - coord)) - 1)
                    cw_short.add((upper << (17 - coord)) | lower)
            if len(cw_short) == 256:
                cl = list(cw_short)
                md = 999
                for i in range(len(cl)):
                    for j in range(i + 1, len(cl)):
                        d = (cl[i] ^ cl[j]).bit_count()
                        if d < md:
                            md = d
                if md > best_min_d:
                    best_min_d = md
                    best_shortened = cw_short
        if best_shortened is not None:
            cw_list = list(best_shortened)
            min_d = best_min_d

    # Precompute popcount for 16-bit chunks
    _pc = np.empty(65536, dtype=np.int8)
    for _i in range(65536):
        _pc[_i] = _i.bit_count()
    popcount_table = _pc

    def get_conflicts(new_word, code_arr):
        """Return boolean mask of codewords with Hamming distance < MIN_DIST."""
        if len(code_arr) == 0:
            return np.array([], dtype=bool)
        xors = code_arr ^ new_word
        low = (xors & 0xFFFF).astype(np.intp)
        high = (xors >> 16).astype(np.intp)
        dists = popcount_table[low].astype(np.intp) + high
        return dists < MIN_DIST

    def check_candidate(candidate, code_arr, skip_idx=-1):
        """Check if candidate has distance >= MIN_DIST from all codewords except skip_idx."""
        if len(code_arr) == 0:
            return True
        xors = code_arr ^ candidate
        low = (xors & 0xFFFF).astype(np.intp)
        high = (xors >> 16).astype(np.intp)
        dists = popcount_table[low].astype(np.intp) + high
        if skip_idx >= 0:
            dists[skip_idx] = 999
        return bool(np.all(dists >= MIN_DIST))

    def try_add_with_repair(new_word, code_list):
        """Try to add new_word, repairing conflicts via bit flips. Returns (success, new_list)."""
        if new_word in code_set_global:
            return False, code_list

        code_arr = np.array(code_list, dtype=np.int64)
        conflicts = get_conflicts(new_word, code_arr)

        if not np.any(conflicts):
            return True, code_list + [new_word]

        new_code = list(code_list)
        new_set = set(code_list)

        for idx in np.where(conflicts)[0]:
            idx = int(idx)
            cw = new_code[idx]
            fixed = False

            # Try single-bit flips
            for bit in range(N):
                new_cw = cw ^ (1 << bit)
                if new_cw == new_word:
                    continue
                if (new_cw ^ new_word).bit_count() < MIN_DIST:
                    continue
                new_code_arr = np.array(new_code, dtype=np.int64)
                if check_candidate(new_cw, new_code_arr, skip_idx=idx):
                    new_set.discard(cw)
                    new_code[idx] = new_cw
                    new_set.add(new_cw)
                    fixed = True
                    break

            if not fixed:
                # Try two-bit flips
                for bit1 in range(N):
                    if fixed:
                        break
                    for bit2 in range(bit1 + 1, N):
                        new_cw = cw ^ (1 << bit1) ^ (1 << bit2)
                        if new_cw == new_word:
                            continue
                        if (new_cw ^ new_word).bit_count() < MIN_DIST:
                            continue
                        new_code_arr = np.array(new_code, dtype=np.int64)
                        if check_candidate(new_cw, new_code_arr, skip_idx=idx):
                            new_set.discard(cw)
                            new_code[idx] = new_cw
                            new_set.add(new_cw)
                            fixed = True
                            break

            if not fixed:
                # Remove the conflicting word
                new_code.pop(idx)
                new_set.discard(cw)
                code_arr2 = np.array(new_code, dtype=np.int64)
                remaining = get_conflicts(new_word, code_arr2)
                if np.any(remaining):
                    return False, code_list

        # Final check
        code_arr_final = np.array(new_code, dtype=np.int64)
        final_conf = get_conflicts(new_word, code_arr_final)
        if np.any(final_conf):
            return False, code_list

        new_code.append(new_word)
        new_set.add(new_word)
        return True, new_code

    # Global best tracking
    global_best = list(cw_list)
    global_best_size = len(cw_list)
    code_set_global = set(cw_list)

    all_words = np.arange(1 << 17, dtype=np.int64)

    # Multiple restarts with different strategies
    n_restarts = 6
    time_per_restart = (deadline - time.time()) / n_restarts

    for restart in range(n_restarts):
        if time.time() > deadline - 2.0:
            break

        # Each restart: fresh SA from the linear code (or best found)
        if restart == 0:
            current_code = list(cw_list)
            current_set = set(cw_list)
        else:
            # Start from global best for later restarts
            current_code = list(global_best)
            current_set = set(global_best)

        rng = random.Random(1000 + restart * 7919)

        T = 8.0
        T_final = 0.01
        cooling = 0.9997
        no_improve = 0
        max_no_improve = 3000

        restart_deadline = time.time() + time_per_restart * 1.2  # Allow some overlap

        while time.time() < restart_deadline and time.time() < deadline - 1.0:
            # Pick a random word to try adding
            new_word = int(rng.choice(all_words))

            if new_word in current_set:
                no_improve += 1
            else:
                success, new_code = try_add_with_repair(new_word, current_code)
                if success:
                    current_code = new_code
                    current_set = set(new_code)
                    no_improve = 0
                    if len(current_code) > global_best_size:
                        global_best = list(current_code)
                        global_best_size = len(current_code)
                        code_set_global = set(global_best)
                else:
                    no_improve += 1

            T = max(T * cooling, T_final)

            if no_improve > max_no_improve:
                # Reset to best and restart temperature
                current_code = list(global_best)
                current_set = set(global_best)
                T = 8.0
                no_improve = 0

    result = global_best

    # Final validation
    assert len(set(result)) == len(result)
    assert all(0 <= w < (1 << N) for w in result)
    for i in range(len(result)):
        for j in range(i + 1, len(result)):
            d = (result[i] ^ result[j]).bit_count()
            assert d >= MIN_DIST, f"Distance {d} < {MIN_DIST}"

    return result
