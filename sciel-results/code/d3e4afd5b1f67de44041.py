import numpy as np
import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 55.0

    N = 17
    MIN_DIST = 6

    # Build the 256-word linear code: shortened [18,9,6] extended QR code
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

    # Try shortening at each coordinate to find one with distance >= 6
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

    if best_shortened is not None and best_min_d >= 6:
        code_list = list(best_shortened)
    else:
        # Fallback: greedy from scratch
        code_list = []
        for w in range(1 << 17):
            if all((w ^ cw).bit_count() >= MIN_DIST for cw in code_list):
                code_list.append(w)
            if len(code_list) >= 256:
                break

    # Convert to numpy array for fast vectorized distance computation
    code_arr = np.array(code_list, dtype=np.uint32)
    best_code = list(code_list)
    best_size = len(code_list)

    rng = random.Random(42)

    def popcount_vec(xor_arr):
        """Vectorized popcount for uint32 array using bit tricks."""
        x = xor_arr.copy()
        x = x - ((x >> 1) & 0x55555555)
        x = (x & 0x33333333) + ((x >> 2) & 0x33333333)
        x = (x + (x >> 4)) & 0x0F0F0F0F
        x = (x * 0x01010101) >> 24
        return x

    def get_conflicts(w, code_arr):
        """Return indices of codewords within distance < MIN_DIST of w."""
        if len(code_arr) == 0:
            return np.array([], dtype=int)
        xor_arr = code_arr ^ w
        dists = popcount_vec(xor_arr)
        return np.where(dists < MIN_DIST)[0]

    # Phase 1: Greedy addition with vectorized checks
    in_code = np.zeros(1 << 17, dtype=bool)
    for w in code_list:
        in_code[w] = True

    remaining = [w for w in range(1 << 17) if not in_code[w]]
    rng.shuffle(remaining)

    added_count = 0
    for w in remaining:
        if time.time() > deadline - 10:
            break
        conf = get_conflicts(w, code_arr)
        if len(conf) == 0:
            code_list.append(w)
            code_arr = np.append(code_arr, w)
            in_code[w] = True
            added_count += 1
            if len(code_list) > best_size:
                best_code = list(code_list)
                best_size = len(code_list)

    # Phase 2: Local search with vectorized conflict detection
    current_list = list(code_list)
    current_arr = np.array(current_list, dtype=np.uint32)
    current_set = set(current_list)

    no_improve = 0
    max_no_improve = 5000
    rng2 = random.Random(123)

    while time.time() < deadline - 3:
        if no_improve > max_no_improve:
            # Restart from best
            current_list = list(best_code)
            current_arr = np.array(current_list, dtype=np.uint32)
            current_set = set(current_list)
            no_improve = 0
            continue

        # Pick a random word to try to add
        w = rng2.randint(0, (1 << 17) - 1)

        if w in current_set:
            no_improve += 1
            continue

        # Vectorized conflict check
        conf = get_conflicts(w, current_arr)

        if len(conf) == 0:
            # Can add directly
            current_list.append(w)
            current_arr = np.append(current_arr, w)
            current_set.add(w)
            no_improve = 0
            if len(current_list) > best_size:
                best_code = list(current_list)
                best_size = len(current_list)
            continue

        # Try repair: for each conflicting codeword, try to flip one bit
        success = False
        # Sort conflicts by distance (closest first) for better repair
        xor_arr = current_arr[conf] ^ w
        conf_dists = popcount_vec(xor_arr)
        sorted_conf = conf[np.argsort(conf_dists)]

        for idx in sorted_conf[:min(len(sorted_conf), 8)]:
            cw = current_list[idx]
            # Try all single-bit flips of cw
            for bit in range(17):
                new_cw = cw ^ (1 << bit)
                if new_cw == w or new_cw in current_set:
                    continue
                if (new_cw ^ w).bit_count() < MIN_DIST:
                    continue
                # Check new_cw against all others (vectorized)
                if len(current_arr) > 1:
                    others = np.delete(current_arr, idx)
                    xor_others = others ^ new_cw
                    dists_others = popcount_vec(xor_others)
                    if np.any(dists_others < MIN_DIST):
                        continue
                # Check w against all others (vectorized)
                if len(current_arr) > 1:
                    xor_w_others = others ^ w
                    dists_w = popcount_vec(xor_w_others)
                    if np.any(dists_w < MIN_DIST):
                        continue
                # Do the swap
                current_set.discard(cw)
                current_list[idx] = new_cw
                current_arr[idx] = new_cw
                current_set.add(new_cw)
                current_list.append(w)
                current_arr = np.append(current_arr, w)
                current_set.add(w)
                success = True
                break
            if success:
                break

        if success:
            no_improve = 0
            if len(current_list) > best_size:
                best_code = list(current_list)
                best_size = len(current_list)
        else:
            # Try 2-swap: remove two conflicting, add two new words
            if len(conf) >= 2 and rng2.random() < 0.3:
                idx1, idx2 = int(conf[0]), int(conf[1])
                removed1 = current_list[idx1]
                removed2 = current_list[idx2]
                # Try to find two replacements
                found = False
                for _ in range(20):
                    w1 = rng2.randint(0, (1 << 17) - 1)
                    w2 = rng2.randint(0, (1 << 17) - 1)
                    if w1 in current_set or w2 in current_set or w1 == w2:
                        continue
                    if (w1 ^ w2).bit_count() < MIN_DIST:
                        continue
                    # Check w1 and w2 against all remaining codewords
                    remaining_arr = np.delete(np.delete(current_arr, idx1), idx2)
                    if len(remaining_arr) > 0:
                        d1 = popcount_vec(remaining_arr ^ w1)
                        d2 = popcount_vec(remaining_arr ^ w2)
                        if np.any(d1 < MIN_DIST) or np.any(d2 < MIN_DIST):
                            continue
                    # Valid! Do the double swap
                    current_set.discard(removed1)
                    current_set.discard(removed2)
                    # Remove both from list (swap-pop both)
                    # Rebuild list for simplicity
                    new_list = []
                    for i, cw in enumerate(current_list):
                        if i != idx1 and i != idx2:
                            new_list.append(cw)
                    new_list.append(w1)
                    new_list.append(w2)
                    current_list = new_list
                    current_arr = np.array(current_list, dtype=np.uint32)
                    current_set = set(current_list)
                    found = True
                    break
                if found:
                    no_improve = 0
                    if len(current_list) > best_size:
                        best_code = list(current_list)
                        best_size = len(current_list)

        no_improve += 1

    result = best_code

    # Verify
    assert len(set(result)) == len(result)
    assert all(0 <= w < (1 << N) for w in result)
    if len(result) > 1:
        for i in range(len(result)):
            for j in range(i + 1, len(result)):
                if (result[i] ^ result[j]).bit_count() < MIN_DIST:
                    assert False, f"Distance < {MIN_DIST} found"

    return result
