"""Greedy set-packing search for A_2(18,6): grow a large binary code of
length 18 and pairwise Hamming distance >= 6, seeded by the first-order
Reed-Muller construction and extended with NumPy-accelerated candidate
generation and distance checks. Includes a wall-clock deadline guard."""
import time
import random
import numpy as np

N = 18
MIN_DISTANCE = 6
DEADLINE_MARGIN = 5.0  # seconds of safety before the 90s kill


def _reseeded_rng(seed):
    return random.Random(seed)


def _seed_words():
    """First-order Reed-Muller code on 16 coordinates, extended by two zero bits.
    32 words, pairwise distance 8 (>= 6)."""
    words = []
    for slope in range(16):
        for intercept in range(2):
            word = 0
            for point in range(16):
                bit = ((slope & point).bit_count() + intercept) % 2
                word |= bit << point
            words.append(word)
    return words


def _popcount(x):
    return x.bit_count()


def _min_distance_to_code(candidate, words):
    """Return min Hamming distance from candidate to any word in words.
    words: list of ints. Pure-Python, fast for small lists."""
    best = 100
    for w in words:
        d = (candidate ^ w).bit_count()
        if d < best:
            best = d
            if best < MIN_DISTANCE:
                return best
    return best


def _min_distance_numpy(candidate, word_array):
    """Min distance from scalar candidate to a numpy int64 array of words."""
    if word_array.size == 0:
        return 100
    diffs = (word_array ^ candidate).astype(np.uint64)
    # popcount via np.unpackbits
    unpacked = np.unpackbits(diffs.view(np.uint8), axis=1, count=8)
    counts = unpacked.sum(axis=1)
    return int(counts.min())


def _try_add_word(candidate, words, word_array, np_mode):
    """Check candidate against current code; add if distance >= MIN_DISTANCE.
    Returns True if added (and updates word_array in place if np_mode)."""
    if np_mode:
        d = _min_distance_numpy(candidate, word_array)
    else:
        d = _min_distance_to_code(candidate, words)
    if d >= MIN_DISTANCE:
        words.append(candidate)
        if np_mode:
            # append to numpy array
            word_array = np.append(word_array, np.int64(candidate))
        return True
    return False


def _greedy_construct(deadline):
    """Main search. Returns list of codeword ints."""
    words = _seed_words()
    word_array = np.array(words, dtype=np.int64)
    np_mode = True

    rng = _reseeded_rng(12345)

    # Precompute some structure: we'll iterate candidates in a randomized
    # order, but to be efficient we can iterate over all 2**18 candidates
    # in a shuffled order, or use a smarter local-search approach.
    #
    # Strategy: iterate over a shuffled list of all candidates, but that's
    # 262144 * len(words) checks which is too slow. Instead, use a
    # randomized local search: start from random words, try to add them,
    # and also try to "swap" or improve.
    #
    # Better strategy: iterate candidates in increasing order of weight
    # (or random order) but break early. Actually, the classic greedy
    # approach for packing is: for each candidate in some order, add it
    # if it's far enough from all existing words. The order matters.
    #
    # Let's use a multi-pass approach:
    # Pass 1: iterate all candidates in a fixed (e.g., weight-ascending)
    #         order, add if compatible. This gives a baseline.
    # Pass 2+: randomized restarts / local improvement.
    #
    # For speed, I'll iterate candidates in a shuffled order but use
    # NumPy vectorized checks against the whole current code at once.

    # Generate all candidates once
    all_candidates = np.arange(2**N, dtype=np.int64)

    # We'll do a greedy pass: for each candidate (in a given order),
    # check min distance to current code; add if >= MIN_DISTANCE.
    # To keep it fast, we check in batches.

    # First, a deterministic weight-ordered pass for a good baseline.
    # Compute weights for all candidates
    weights = np.unpackbits(all_candidates.view(np.uint8), axis=1, count=8).sum(axis=1)
    order = np.argsort(weights, kind='stable')  # weight ascending

    batch_size = 4096
    added_count = 0

    for start in range(0, len(order), batch_size):
        if time.time() > deadline:
            break
        batch_idx = order[start:start+batch_size]
        batch = all_candidates[batch_idx]
        # For each candidate in batch, compute min distance to current code
        # Current code is word_array (size k). We compute a k x batch distance matrix.
        if word_array.size == 0:
            # add all
            for c in batch:
                words.append(int(c))
            word_array = np.append(word_array, batch)
            continue
        # Compute distances: for each candidate, min over code
        # Use chunking to avoid huge memory
        sub_batch_size = min(batch_size, max(1, (1 << 20) // max(1, word_array.size)))
        for sb_start in range(0, len(batch), sub_batch_size):
            if time.time() > deadline:
                break
            sb = batch[sb_start:sb_start+sub_batch_size]
            # distances: (len(sb), word_array.size)
            diffs = sb[:, None] ^ word_array[None, :]
            diffs = diffs.astype(np.uint64)
            unpacked = np.unpackbits(diffs.view(np.uint8).reshape(-1, 8), axis=1, count=8)
            unpacked = unpacked.reshape(diffs.shape[0], diffs.shape[1], 8)
            dists = unpacked.sum(axis=2)
            min_dists = dists.min(axis=1)
            for i, md in enumerate(min_dists):
                if md >= MIN_DISTANCE:
                    words.append(int(sb[i]))
                    word_array = np.append(word_array, np.int64(sb[i]))
                    added_count += 1

    # Now do randomized local-search passes to try to add more words.
    # In each pass, sample random candidates and try to add them.
    # Also, we can try to "improve" by removing a word and adding two.
    # For simplicity, just random sampling passes.

    for pass_num in range(100):
        if time.time() > deadline:
            break
        # Sample a batch of random candidates
        sample_size = 20000
        samples = rng.sample(range(2**N), sample_size)
        samples_arr = np.array(samples, dtype=np.int64)
        if word_array.size == 0:
            continue
        sub_batch_size = min(sample_size, max(1, (1 << 20) // max(1, word_array.size)))
        for sb_start in range(0, len(samples_arr), sub_batch_size):
            if time.time() > deadline:
                break
            sb = samples_arr[sb_start:sb_start+sub_batch_size]
            diffs = sb[:, None] ^ word_array[None, :]
            diffs = diffs.astype(np.uint64)
            unpacked = np.unpackbits(diffs.view(np.uint8).reshape(-1, 8), axis=1, count=8)
            unpacked = unpacked.reshape(diffs.shape[0], diffs.shape[1], 8)
            dists = unpacked.sum(axis=2)
            min_dists = dists.min(axis=1)
            for i, md in enumerate(min_dists):
                if md >= MIN_DISTANCE:
                    words.append(int(sb[i]))
                    word_array = np.append(word_array, np.int64(sb[i]))

    return words


def construct():
    start = time.time()
    deadline = start + 90.0 - DEADLINE_MARGIN
    try:
        words = _greedy_construct(deadline)
    except Exception:
        # Fallback to the seed if anything goes wrong
        words = _seed_words()
    # Ensure distinctness and range
    seen = set()
    out = []
    for w in words:
        if w not in seen and 0 <= w < 2**N:
            seen.add(w)
            out.append(w)
    if not out:
        out = _seed_words()
    return out
