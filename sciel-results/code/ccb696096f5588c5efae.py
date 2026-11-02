"""Ternary covering code K_3(7,1) via vectorized greedy + redundancy pruning."""
import time
import numpy as np
import itertools


def construct(q, n, radius):
    """Return a list of codewords (each a list of n ints in 0..q-1) covering {0..q-1}^n
    within Hamming distance radius. Score is the number of codewords (lower is better)."""
    deadline = time.time() + 115.0  # safety margin before 120s kill

    # Only the known task parameters matter for performance; fall back to a trivial
    # construction for other (q, n, radius) so the interface stays valid.
    if q != 3 or n != 7 or radius != 1:
        # Trivial: all words (valid but huge); better: a simple product code.
        # For safety, return the direct-sum tetracode style if possible, else all words.
        if q == 3 and n >= 4 and radius == 1:
            tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
            return [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]
        # Last resort: every word
        return [list(t) for t in itertools.product(range(q), repeat=n)]

    # --- Specialized solver for q=3, n=7, radius=1 ---
    # Total space size: 3^7 = 2187. Each codeword covers 1 + 7*2 = 15 words.
    # Lower bound: ceil(2187 / 15) = 146. Best known: 186.

    # Precompute the index of each word: word -> index in [0, 2187)
    # word is a tuple of 7 ints in {0,1,2}. Index = w[0]*3^6 + w[1]*3^5 + ... + w[6]
    powers = [3 ** (n - 1 - i) for i in range(n)]  # [729, 243, 81, 27, 9, 3, 1]

    def to_idx(w):
        return sum(w[i] * powers[i] for i in range(n))

    def from_idx(idx):
        w = [0] * n
        for i in range(n):
            w[i] = idx // powers[i]
            idx %= powers[i]
        return w

    # Precompute the neighborhood (sphere of radius 1) for every word.
    # sphere[idx] = list of indices of words within Hamming distance 1 of word idx.
    # Size = 1 + 7*2 = 15.
    sphere = np.zeros((2187, 15), dtype=np.int32)
    for idx in range(2187):
        w = from_idx(idx)
        sphere[idx, 0] = idx
        k = 1
        for pos in range(n):
            orig = w[pos]
            for new_val in range(3):
                if new_val == orig:
                    continue
                w[pos] = new_val
                sphere[idx, k] = to_idx(w)
                k += 1
            w[pos] = orig
        # k should be 15
    assert k == 15

    # Vectorized greedy: at each step, pick the codeword that covers the most uncovered words.
    # We'll use multiple random restarts with random initial codeword and random tie-breaking.

    def greedy_cover(seed_rng):
        """One greedy run. Returns list of codeword indices."""
        uncovered = np.ones(2187, dtype=np.bool_)
        codewords = []
        # Optional: start with a random codeword to break symmetry
        start = seed_rng.integers(0, 2187)
        uncovered[sphere[start]] = False
        codewords.append(start)

        while uncovered.any():
            # For each candidate codeword c, gain = number of currently uncovered words in sphere[c]
            # gain[c] = sum(uncovered[sphere[c]])
            # Vectorized: gains = sphere @ uncovered (but sphere is int, uncovered is bool)
            gains = sphere.astype(np.int32) @ uncovered.astype(np.int32)
            # Pick the candidate with max gain; break ties randomly
            max_gain = gains.max()
            candidates = np.where(gains == max_gain)[0]
            if len(candidates) == 0:
                break
            choice = candidates[seed_rng.integers(0, len(candidates))]
            codewords.append(choice)
            uncovered[sphere[choice]] = False

        return codewords

    def remove_redundant(codewords):
        """Remove codewords that are redundant (their sphere is covered by others)."""
        if len(codewords) <= 1:
            return list(codewords)
        covered = np.zeros(2187, dtype=np.int32)
        for c in codewords:
            covered[sphere[c]] += 1
        result = []
        for c in codewords:
            # Check if removing c leaves all its sphere words still covered
            # i.e., for all w in sphere[c], covered[w] > 1
            if (covered[sphere[c]] > 1).all():
                # c is redundant; don't add it
                covered[sphere[c]] -= 1
            else:
                result.append(c)
        return result

    # Run multiple restarts, keep the best (smallest) result
    best = None
    best_size = 2187
    rng = np.random.default_rng(42)

    # First, try a deterministic greedy (no random start)
    codewords = greedy_cover(np.random.default_rng(0))
    codewords = remove_redundant(codewords)
    if len(codewords) < best_size:
        best_size = len(codewords)
        best = list(codewords)

    # Random restarts
    while time.time() < deadline - 5.0:
        restart_rng = np.random.default_rng(rng.integers(0, 2**31))
        codewords = greedy_cover(restart_rng)
        codewords = remove_redundant(codewords)
        if len(codewords) < best_size:
            best_size = len(codewords)
            best = list(codewords)
            if best_size <= 156:  # can't do better than lower bound
                break

    # Convert indices to list of lists
    return [from_idx(c) for c in best]
