import time
import random
import numpy as np

N = 18
MIN_DIST = 6
TIME_LIMIT = 60.0

def popcount(x):
    return x.bit_count()

def hamming(a, b):
    return popcount(a ^ b)

def verify_code(words, min_dist=MIN_DIST):
    """Verify all pairwise distances using NumPy for speed."""
    if len(words) <= 1:
        return True
    arr = np.array(words, dtype=np.uint32)
    n = len(arr)
    for i in range(n):
        diffs = arr[i + 1:] ^ arr[i]
        counts = np.array([d.bit_count() for d in diffs])
        if np.any(counts < min_dist):
            return False
    return True

def construct_extended_qr_512():
    """
    Constructs the [17,9,5] QR code for p=17 and extends to 18 bits.
    Returns the 512-word base linear code L (a [18,9] code).
    """
    p = 17
    qr = set((i * i) % p for i in range(p))

    k = (p + 1) // 2  # 9
    G = [[0] * p for _ in range(k)]
    for i in range(k):
        for j in range(p):
            if (j - i) % p in qr:
                G[i][j] = 1

    codewords_17 = []
    for mask in range(1 << k):
        word = 0
        for i in range(k):
            if mask & (1 << i):
                for j in range(p):
                    if G[i][j]:
                        word ^= (1 << j)
        codewords_17.append(word)

    codewords_18 = []
    for w in codewords_17:
        parity = popcount(w) % 2
        codewords_18.append(w | (parity << p))

    return codewords_18

def _min_dist_to_set(v, arr):
    """Minimum Hamming distance from scalar v to the array arr (uint32)."""
    diffs = arr ^ v
    return min(d.bit_count() for d in diffs)

def find_coset_offset(L, time_budget=30.0):
    """
    Searches for a vector v such that min distance from v to L is >= MIN_DIST.
    If found, L U (v+L) is a valid code of size 2*len(L).
    """
    n = len(L)
    L_set = set(L)
    L_arr = np.array(L, dtype=np.uint32)

    deadline = time.time() + time_budget
    random.seed(42)

    from itertools import combinations
    for weight in range(0, 4):
        if time.time() > deadline:
            break
        for combo in combinations(range(N), weight):
            v = 0
            for pos in combo:
                v |= (1 << pos)
            if v in L_set:
                continue
            if _min_dist_to_set(v, L_arr) >= MIN_DIST:
                return v

    while time.time() < deadline:
        v = random.randint(0, (1 << N) - 1)
        if v in L_set:
            continue
        if _min_dist_to_set(v, L_arr) >= MIN_DIST:
            return v

    return None

def flip_search(words, time_budget=0.0):
    """
    Local search: try to add random words and perform single-bit flips of
    existing codewords to free up packing space, accepting moves only when the
    full code stays valid. Returns a (possibly larger) valid code.
    """
    if time_budget <= 0:
        return list(words)

    deadline = time.time() + time_budget
    best = list(words)
    best_len = len(best)
    best_arr = np.array(best, dtype=np.uint32)
    best_set = set(best)

    random.seed(123)

    def is_valid_with(candidate, exclude_idx=None):
        if candidate in best_set:
            return False
        diffs = best_arr ^ candidate
        counts = [d.bit_count() for d in diffs]
        for idx, c in enumerate(counts):
            if idx == exclude_idx:
                continue
            if c < MIN_DIST:
                return False
        return True

    while time.time() < deadline:
        grow = random.random() < 0.4
        if grow:
            candidate = random.randint(0, (1 << N) - 1)
            if candidate in best_set:
                continue
            diffs = best_arr ^ candidate
            if min(d.bit_count() for d in diffs) >= MIN_DIST:
                best.append(candidate)
                best_set.add(candidate)
                best_arr = np.append(best_arr, candidate)
                best_len = len(best)
        else:
            idx = random.randint(0, len(best) - 1)
            bit = random.randint(0, N - 1)
            new_word = best[idx] ^ (1 << bit)
            if new_word in best_set:
                continue
            if is_valid_with(new_word, exclude_idx=idx):
                best[idx] = new_word
                best_set.discard(best[idx] ^ (1 << bit))  # old word removed
                best_set.add(new_word)
                best_arr[idx] = new_word

    return best

def greedy_extend(words, time_budget=0.0):
    """
    Greedy extension with NumPy-accelerated distance checks.
    """
    if time_budget <= 0:
        return list(words)

    deadline = time.time() + time_budget
    words = list(words)
    word_set = set(words)
    words_arr = np.array(words, dtype=np.uint32)
    random.seed(123)

    while time.time() < deadline:
        candidate = random.randint(0, (1 << N) - 1)
        if candidate in word_set:
            continue
        diffs = words_arr ^ candidate
        if min(d.bit_count() for d in diffs) >= MIN_DIST:
            words.append(candidate)
            word_set.add(candidate)
            words_arr = np.append(words_arr, candidate)

    return words

def construct():
    start_time = time.time()
    deadline = start_time + TIME_LIMIT - 3.0  # Safety margin

    # Step 1: Construct the 512-word base linear code L.
    base_code = construct_extended_qr_512()

    if not verify_code(base_code):
        # Fallback to small RM code if QR construction fails (should not happen)
        words = []
        for slope in range(16):
            for intercept in range(2):
                word = 0
                for point in range(16):
                    bit = ((slope & point).bit_count() + intercept) % 2
                    word |= bit << point
                words.append(word)
        return words

    all_ones = (1 << N) - 1
    base_arr = np.array(base_code, dtype=np.uint32)
    base_set = set(base_code)

    # Step 2: If the all-ones vector is a codeword (it is, for p=17),
    # doubling via L U (all_ones + L) yields the full [18,10,6] code of 1024 words.
    if all_ones in base_set:
        if _min_dist_to_set(all_ones, base_arr) >= MIN_DIST:
            full_code = list(base_code)
            for w in base_code:
                full_code.append(w ^ all_ones)
            if verify_code(full_code):
                best_words = full_code
                # Step 3: use remaining time to try to exceed 1024.
                remaining = deadline - time.time()
                if remaining > 2:
                    extended = flip_search(best_words, time_budget=remaining)
                    if len(extended) > len(best_words) and verify_code(extended):
                        best_words = extended
                return best_words

    # Fallback: no all-ones symmetry; try a general coset offset.
    best_words = list(base_code)
    remaining = deadline - time.time()
    if remaining > 5:
        offset = find_coset_offset(base_code, time_budget=remaining - 2.0)
        if offset is not None:
            full_code = list(base_code)
            for w in base_code:
                full_code.append(w ^ offset)
            if verify_code(full_code):
                best_words = full_code

    remaining = deadline - time.time()
    if remaining > 2:
        extended = flip_search(best_words, time_budget=remaining)
        if len(extended) > len(best_words) and verify_code(extended):
            best_words = extended

    return best_words
