import time
import random
import numpy as np

_DEADLINE = 140.0

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + _DEADLINE

    total = q ** n
    nwords = (total + 63) // 64

    powers = [q ** (n - 1 - i) for i in range(n)]

    def idx_to_digits(idx):
        digits = []
        x = idx
        for _ in range(n):
            digits.append(x % q)
            x //= q
        return digits[::-1]

    # Precompute ball masks as numpy array (total, nwords) uint64
    ball_masks = np.zeros((total, nwords), dtype=np.uint64)

    for idx in range(total):
        digits = idx_to_digits(idx)
        w = idx // 64
        b = idx % 64
        ball_masks[idx, w] |= (1 << b)

        if radius >= 1:
            for pos in range(n):
                orig = digits[pos]
                for alt in range(q):
                    if alt == orig:
                        continue
                    new_idx = idx + (alt - orig) * powers[pos]
                    w2 = new_idx // 64
                    b2 = new_idx % 64
                    ball_masks[idx, w2] |= (1 << b2)

    # All-ones mask
    all_mask = np.zeros(nwords, dtype=np.uint64)
    all_mask[:] = 0xFFFFFFFFFFFFFFFF
    if total % 64 != 0:
        all_mask[-1] &= (1 << (total % 64)) - 1

    # Popcount for a single uint64
    def pc(x):
        x = int(x)
        return bin(x).count('1')

    # Popcount for a numpy array of uint64
    def pc_arr(arr):
        c = 0
        for v in arr:
            c += bin(int(v)).count('1')
        return c

    # Greedy Set Cover
    def greedy_cover(seed=0):
        rng = random.Random(seed)
        uncovered = all_mask.copy()
        selected = []
        remaining = list(range(total))
        rng.shuffle(remaining)

        while pc_arr(uncovered) > 0:
            if time.time() > deadline - 15:
                break

            best_count = -1
            best_idx = -1

            # Compute gains
            gains = np.zeros(len(remaining), dtype=np.int32)
            for k, c in enumerate(remaining):
                gain = 0
                for j in range(nwords):
                    v = int(ball_masks[c, j] & uncovered[j])
                    gain += bin(v).count('1')
                gains[k] = gain

            max_gain = int(np.max(gains))
            if max_gain <= 0:
                break

            # Pick from top candidates
            threshold = max(1, int(max_gain * 0.95))
            candidates = [remaining[k] for k in range(len(remaining)) if gains[k] >= threshold]
            best_idx = rng.choice(candidates)

            selected.append(best_idx)
            uncovered = uncovered & ~ball_masks[best_idx]
            remaining.remove(best_idx)

        return selected

    # Local Search
    def local_search(C, seed=0):
        if not C:
            return C

        rng = random.Random(seed)

        # counts[w] = number of codewords in C covering w
        # Use the symmetry: w in ball(c) iff c in ball(w)
        # counts[w] = sum_{c in C} [c in ball(w)]
        # We can compute this by for each c in C, adding 1 to all w in ball(c)

        counts = np.zeros(total, dtype=np.int32)
        for c in C:
            # Expand ball_masks[c] to increment counts
            bm = ball_masks[c]
            for j in range(nwords):
                v = int(bm[j])
                base = j * 64
                while v:
                    lsb = v & -v
                    bit = lsb.bit_length() - 1
                    idx = base + bit
                    if idx < total:
                        counts[idx] += 1
                    v ^= lsb

        best_C = list(C)
        best_size = len(C)
        no_improve = 0

        while time.time() < deadline - 8:
            improved = False

            # Squeeze
            indices = list(range(len(C)))
            rng.shuffle(indices)

            for i in indices:
                c = C[i]
                # Check if c is removable: for all w in ball(c), counts[w] > 1
                bm = ball_masks[c]
                essential = False
                for j in range(nwords):
                    v = int(bm[j])
                    base = j * 64
                    while v:
                        lsb = v & -v
                        bit = lsb.bit_length() - 1
                        idx = base + bit
                        if idx < total and counts[idx] == 1:
                            essential = True
                            break
                        v ^= lsb
                    if essential:
                        break

                if not essential:
                    # Remove c
                    bm = ball_masks[c]
                    for j in range(nwords):
                        v = int(bm[j])
                        base = j * 64
                        while v:
                            lsb = v & -v
                            bit = lsb.bit_length() - 1
                            idx = base + bit
                            if idx < total:
                                counts[idx] -= 1
                            v ^= lsb
                    C.pop(i)
                    improved = True
                    if len(C) < best_size:
                        best_size = len(C)
                        best_C = list(C)
                    break

            if improved:
                no_improve = 0
                continue

            # Swap
            swapped = False
            for _ in range(100):
                if not C:
                    break
                i = rng.randint(0, len(C) - 1)
                c = C[i]

                # Find U_c: words w in ball(c) with counts[w] == 1
                Uc = []
                bm = ball_masks[c]
                for j in range(nwords):
                    v = int(bm[j])
                    base = j * 64
                    while v:
                        lsb = v & -v
                        bit = lsb.bit_length() - 1
                        idx = base + bit
                        if idx < total and counts[idx] == 1:
                            Uc.append(idx)
                        v ^= lsb

                if not Uc:
                    # Remove c
                    bm = ball_masks[c]
                    for j in range(nwords):
                        v = int(bm[j])
                        base = j * 64
                        while v:
                            lsb = v & -v
                            bit = lsb.bit_length() - 1
                            idx = base + bit
                            if idx < total:
                                counts[idx] -= 1
                            v ^= lsb
                    C.pop(i)
                    improved = True
                    if len(C) < best_size:
                        best_size = len(C)
                        best_C = list(C)
                    break

                # Find c' that covers all of Uc
                # c' must have all words in Uc in its ball
                # i.e., for all w in Uc, w in ball(c') iff c' in ball(w)
                # So c' must be in intersection of ball(w) for w in Uc

                # Compute intersection of ball(w) for w in Uc
                # Start with all_mask, AND with ball(w) for each w in Uc
                inter = all_mask.copy()
                for w in Uc:
                    inter = inter & ball_masks[w]

                # Find candidates in inter that are not in C (or can be, swapping)
                # Actually we just need any word in inter
                candidates = []
                for j in range(nwords):
                    v = int(inter[j])
                    base = j * 64
                    while v:
                        lsb = v & -v
                        bit = lsb.bit_length() - 1
                        idx = base + bit
                        if idx < total:
                            candidates.append(idx)
                        v ^= lsb

                if candidates:
                    c_new = rng.choice(candidates)
                    # Perform swap
                    # Remove c
                    bm = ball_masks[c]
                    for j in range(nwords):
                        v = int(bm[j])
                        base = j * 64
                        while v:
                            lsb = v & -v
                            bit = lsb.bit_length() - 1
                            idx = base + bit
                            if idx < total:
                                counts[idx] -= 1
                            v ^= lsb
                    # Add c_new
                    bm = ball_masks[c_new]
                    for j in range(nwords):
                        v = int(bm[j])
                        base = j * 64
                        while v:
                            lsb = v & -v
                            bit = lsb.bit_length() - 1
                            idx = base + bit
                            if idx < total:
                                counts[idx] += 1
                            v ^= lsb
                    C[i] = c_new
                    swapped = True
                    break

            if swapped:
                no_improve = 0
                continue

            no_improve += 1
            if no_improve >= 3:
                break

        return best_C

    # Main
    best_result = None
    best_size = float('inf')

    for seed in range(100):
        if time.time() > deadline - 25:
            break

        C = greedy_cover(seed=seed)
        if not C:
            continue

        C = local_search(C, seed=seed)

        if len(C) < best_size:
            best_size = len(C)
            best_result = list(C)

        if best_size <= 170:
            break

    if best_result is None:
        best_result = greedy_cover(seed=0)

    result = []
    for idx in best_result:
        result.append(idx_to_digits(idx))

    return result
