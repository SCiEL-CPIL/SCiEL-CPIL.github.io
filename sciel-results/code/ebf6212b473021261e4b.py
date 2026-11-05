import numpy as np
import time
import random
import itertools

def construct():
    N = 17
    MIN_D = 6
    ALL_ONES = (1 << N) - 1
    deadline = time.time() + 52  # safety margin before 60s limit

    # Precompute popcount table for O(1) Hamming distance lookups
    popcount = np.zeros(256, dtype=np.uint8)
    for i in range(256):
        popcount[i] = bin(i).count('1')

    def popcount17(x):
        """Fast popcount for 17-bit integers"""
        return popcount[x & 0xFF] + popcount[(x >> 8) & 0xFF] + popcount[(x >> 16) & 0xFF]

    # ---------- Step 1: Build the 256-word [17,8,6] shortened extended QR code ----------
    # f1(x) = x^8 + x^4 + x^3 + x^2 + 1
    f1 = [1, 0, 0, 1, 1, 1, 0, 0, 1]

    basis = []
    for i in range(9):
        vec = [0] * 17
        for j in range(len(f1)):
            if i + j < 17:
                vec[i + j] = f1[j]
        basis.append(vec)

    words_17 = []
    for mask in range(512):
        word = [0] * 17
        for i in range(9):
            if mask & (1 << i):
                for j in range(17):
                    word[j] ^= basis[i][j]
        words_17.append(word)

    # Extend to 18 bits (parity)
    words_18 = []
    for w in words_17:
        parity = sum(w) % 2
        words_18.append(w + [parity])

    # Shorten: keep words where bit 0 is 0
    codewords = []
    for w18 in words_18:
        if w18[0] == 0:
            new_word = w18[1:]
            val = 0
            for i, bit in enumerate(new_word):
                if bit:
                    val |= (1 << i)
            codewords.append(val)

    # ---------- Step 2: Fast vectorized greedy extension ----------
    # Maintain code as a numpy array for fast batched distance checks.
    codes = np.array(codewords, dtype=np.uint32)
    code_weights = np.array([popcount17(c) for c in codes], dtype=np.uint16)
    code_set = set(codewords)
    checked = np.zeros(2 ** N, dtype=bool)
    for c in codewords:
        checked[c] = True

    def add_candidate(val):
        """Add a candidate to the code, updating all bookkeeping."""
        nonlocal codes, code_weights
        val = int(val)
        codes = np.append(codes, val)
        code_weights = np.append(code_weights, popcount17(val))
        code_set.add(val)
        checked[val] = True

    def vectorized_greedy_pass():
        """Scan all unchecked candidates in batches, add any with min distance >= MIN_D."""
        nonlocal codes, code_weights
        n_codes = len(codes)
        if n_codes == 0:
            add_candidate(0)
            return

        batch_size = 32768
        for start in range(0, 2 ** N, batch_size):
            if time.time() >= deadline:
                return
            end = min(start + batch_size, 2 ** N)

            # Get unchecked candidates in this batch
            mask = ~checked[start:end]
            if not np.any(mask):
                continue
            batch_vals = np.arange(start, end, dtype=np.int64)[mask]
            if len(batch_vals) == 0:
                continue

            batch_weights = np.array([popcount17(v) for v in batch_vals], dtype=np.uint16)

            sub_batch_size = 8192
            newly_added = []
            for sub_start in range(0, len(batch_vals), sub_batch_size):
                if time.time() >= deadline:
                    return
                sub_end = min(sub_start + sub_batch_size, len(batch_vals))
                sub_vals = batch_vals[sub_start:sub_end]
                sub_weights = batch_weights[sub_start:sub_end]

                min_dists = np.full(len(sub_vals), 100, dtype=np.uint16)

                for j in range(len(codes)):
                    code = int(codes[j])
                    and_vals = sub_vals & code
                    pc = np.array([popcount17(v) for v in and_vals], dtype=np.uint16)
                    dists = sub_weights + code_weights[j] - 2 * pc
                    min_dists = np.minimum(min_dists, dists)

                valid_mask = min_dists >= MIN_D
                valid_sub_indices = np.where(valid_mask)[0]
                for idx in valid_sub_indices:
                    candidate = int(sub_vals[idx])
                    if candidate in code_set:
                        continue
                    add_candidate(candidate)
                    newly_added.append(candidate)

    # Run a full vectorized pass to harvest any immediately-addable words
    vectorized_greedy_pass()

    # ---------- Step 3: Targeted candidate generation (inspiration strategies) ----------
    random.seed(12345)

    # Strategy A: complements of existing words
    for w in list(codes):
        if time.time() >= deadline:
            break
        comp = int(w) ^ ALL_ONES
        if comp not in code_set:
            # quick validity check against all codes
            ok = True
            for c in codes:
                if popcount17(comp ^ int(c)) < MIN_D:
                    ok = False
                    break
            if ok:
                add_candidate(comp)

    # Strategy B: random sampling
    attempts = 0
    max_attempts = 400000
    while time.time() < deadline - 6.0 and attempts < max_attempts:
        candidate = random.getrandbits(N)
        if candidate not in code_set:
            ok = True
            for c in codes:
                if popcount17(candidate ^ int(c)) < MIN_D:
                    ok = False
                    break
            if ok:
                add_candidate(candidate)
        attempts += 1

    # Strategy C: structured perturbations (flip exactly 6 bits from existing words)
    if time.time() < deadline - 4.0:
        sample_words = random.sample(list(codes), min(60, len(codes)))
        for w in sample_words:
            if time.time() >= deadline - 4.0:
                break
            w = int(w)
            for _ in range(30):
                bits = random.sample(range(N), 6)
                mask = 0
                for b in bits:
                    mask |= (1 << b)
                candidate = w ^ mask
                if candidate not in code_set:
                    ok = True
                    for c in codes:
                        if popcount17(candidate ^ int(c)) < MIN_D:
                            ok = False
                            break
                    if ok:
                        add_candidate(candidate)

    # ---------- Step 4: Final vectorized passes to harvest anything left ----------
    while time.time() < deadline - 2.0:
        before = len(codes)
        vectorized_greedy_pass()
        if len(codes) == before:
            break

    return [int(c) for c in codes]
