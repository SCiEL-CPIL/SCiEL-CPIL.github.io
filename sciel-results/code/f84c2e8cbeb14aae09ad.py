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

    # Now use a fast greedy + local search approach
    # Key: use O(1) swap-pop for list maintenance, avoid numpy allocations in inner loop

    code_set = set(code_list)
    best_code = list(code_list)
    best_size = len(code_list)

    rng = random.Random(42)

    # Precompute: for each word, store it as an integer
    # Fast conflict check: for a candidate word w, check against all in code_set
    # Use early termination

    def has_conflict(w, code_list):
        for cw in code_list:
            if (w ^ cw).bit_count() < MIN_DIST:
                return True
        return False

    def count_conflicts(w, code_list):
        cnt = 0
        for cw in code_list:
            if (w ^ cw).bit_count() < MIN_DIST:
                cnt += 1
        return cnt

    # Phase 1: Greedy addition - repeatedly try to add random words
    # Use swap-pop to maintain a list of "active" candidates

    # Build a list of all words not in code
    all_words = list(range(1 << 17))
    in_code = [False] * (1 << 17)
    for w in code_list:
        in_code[w] = True

    # Shuffle for random ordering
    remaining = [w for w in all_words if not in_code[w]]
    rng.shuffle(remaining)

    # Greedy: scan through remaining, add if no conflict
    added_count = 0
    for w in remaining:
        if time.time() > deadline - 5:
            break
        if not has_conflict(w, code_list):
            code_list.append(w)
            code_set.add(w)
            in_code[w] = True
            added_count += 1
            if len(code_list) > best_size:
                best_code = list(code_list)
                best_size = len(code_list)

    # Phase 2: Local search with swap-pop
    # Try to replace pairs: remove one codeword, add two new ones
    # Or: remove one, add one (with different word)

    # For efficiency, maintain a list of "nearby" candidates for each codeword
    # Actually, let's do a simpler approach: random perturbation + repair

    current_list = list(code_list)
    current_set = set(code_list)

    T0 = 30.0
    T = T0
    cooling = 0.999
    T_final = 0.5
    no_improve = 0
    max_no_improve = 3000

    while time.time() < deadline - 2:
        if no_improve > max_no_improve:
            # Restart from best
            current_list = list(best_code)
            current_set = set(best_code)
            T = T0
            no_improve = 0
            continue

        # Pick a random word to try to add
        w = rng.randint(0, (1 << 17) - 1)

        if w in current_set:
            no_improve += 1
            T = max(T * cooling, T_final)
            continue

        # Count conflicts
        conflicts = []
        for i, cw in enumerate(current_list):
            if (w ^ cw).bit_count() < MIN_DIST:
                conflicts.append(i)

        if not conflicts:
            # Can add directly
            current_list.append(w)
            current_set.add(w)
            no_improve = 0
            if len(current_list) > best_size:
                best_code = list(current_list)
                best_size = len(current_list)
            T = max(T * cooling, T_final)
            continue

        # Try to resolve by moving conflicting codewords
        # For each conflicting codeword, try to flip one bit to get a valid replacement
        success = False
        for idx in conflicts:
            cw = current_list[idx]
            for bit in range(17):
                new_cw = cw ^ (1 << bit)
                if new_cw == w or new_cw in current_set:
                    continue
                if (new_cw ^ w).bit_count() < MIN_DIST:
                    continue
                # Check new_cw against all others
                valid = True
                for j, other in enumerate(current_list):
                    if j == idx:
                        continue
                    if (new_cw ^ other).bit_count() < MIN_DIST:
                        valid = False
                        break
                if not valid:
                    continue
                # Also check w against all others (excluding idx which we're replacing)
                for j, other in enumerate(current_list):
                    if j == idx:
                        continue
                    if (w ^ other).bit_count() < MIN_DIST:
                        valid = False
                        break
                if not valid:
                    continue
                # Do the swap
                current_set.discard(cw)
                current_list[idx] = new_cw
                current_set.add(new_cw)
                current_list.append(w)
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
            # Try removing one conflicting codeword and adding w
            # This keeps size same but might enable future additions
            if len(conflicts) == 1 and rng.random() < 0.1:
                idx = conflicts[0]
                removed = current_list[idx]
                current_set.discard(removed)
                # Remove from list (swap-pop)
                last = current_list.pop()
                if idx < len(current_list):
                    current_list[idx] = last
                current_list.append(w)
                current_set.add(w)

        no_improve += 1
        T = max(T * cooling, T_final)

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
