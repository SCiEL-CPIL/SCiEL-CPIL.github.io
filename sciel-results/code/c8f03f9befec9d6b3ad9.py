"""Greedy set-cover for K_q(n, radius) using NumPy for efficient coverage updates."""
# EVOLVE-BLOCK-START
import time
import itertools
import numpy as np


def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 110.0  # safety margin before 120s kill

    total = q ** n
    # Each codeword covers 1 + n*(q-1) words (itself + one mutation per coordinate)
    cover_size = 1 + n * (q - 1)

    # Precompute all words as indices 0..total-1
    # We'll track coverage as a boolean array of length total
    # But for n=6, q=3, total=729, this is small. For larger cases, we need efficiency.

    # Strategy: greedy set cover
    # 1. Start with all words uncovered
    # 2. Repeatedly pick the word that covers the most uncovered words
    # 3. Mark those as covered
    # 4. Stop when all are covered

    # For efficiency, we precompute for each candidate codeword, the set of words it covers.
    # Then we maintain a count of how many times each word is "covered by candidates we haven't picked yet"
    # Actually, simpler: maintain uncovered set, and for each step, score each candidate by how many
    # currently-uncovered words it covers.

    # For n=6, q=3: total=729, cover_size=1+6*2=13
    # Number of candidates = 729
    # We can represent coverage as a bitset or boolean array.

    # Let's use a different approach for small cases:
    # Represent the space as indices 0..total-1
    # For each candidate c, precompute the list of indices it covers
    # Maintain an array 'uncovered_count' but actually just a boolean 'is_uncovered'
    # For scoring, for candidate c, score = sum(is_uncovered[i] for i in covers[c])

    # For total=729, this is 729 * 13 = 9477 operations per step, and we need ~73 steps.
    # That's ~700k operations total, very fast.

    # For larger total, we need to be smarter. Let's handle the general case.

    # Precompute covers: for each word index, what indices does it cover?
    # A word covers itself and all words at Hamming distance 1.

    # Convert index to word and back
    def idx_to_word(idx):
        w = [0] * n
        for i in range(n):
            w[i] = idx % q
            idx //= q
        return w

    def word_to_idx(w):
        idx = 0
        for i in range(n):
            idx = idx * q + w[i]
        return idx

    # Precompute all covers
    all_covers = []
    for idx in range(total):
        w = idx_to_word(idx)
        covered = [idx]
        for pos in range(n):
            orig = w[pos]
            for val in range(q):
                if val == orig:
                    continue
                w[pos] = val
                covered.append(word_to_idx(w))
            w[pos] = orig
        all_covers.append(covered)

    # Convert to numpy arrays for faster summing
    covers_arr = [np.array(c) for c in all_covers]

    # Track uncovered
    is_uncovered = np.ones(total, dtype=np.int8)
    uncovered_count = total

    selected = []

    while uncovered_count > 0:
        if time.time() > deadline:
            # Return what we have (may be invalid if not all covered, but better than timeout)
            break

        # Score each candidate: number of uncovered words it covers
        best_score = -1
        best_idx = -1
        for i in range(total):
            if is_uncovered[i] == 0:
                # This word is already covered, but it could still be a candidate
                # Actually, we should consider all candidates, not just uncovered ones
                pass
            score = np.sum(is_uncovered[covers_arr[i]])
            if score > best_score:
                best_score = score
                best_idx = i

        if best_idx == -1 or best_score == 0:
            # Shouldn't happen if uncovered_count > 0
            break

        # Select this codeword
        selected.append(best_idx)
        # Mark covered words
        for idx in all_covers[best_idx]:
            if is_uncovered[idx]:
                is_uncovered[idx] = 0
                uncovered_count -= 1

    # Convert selected indices to word lists
    codewords = [idx_to_word(idx) for idx in selected]

    # Verify coverage (quick check)
    # For safety, let's verify
    if len(codewords) < total:
        # Quick verification using the covers
        covered_set = set()
        for idx in selected:
            for c in all_covers[idx]:
                covered_set.add(c)
        if len(covered_set) < total:
            # Fallback: add missing words
            missing = [i for i in range(total) if i not in covered_set]
            for idx in missing:
                codewords.append(idx_to_word(idx))

    return codewords
# EVOLVE-BLOCK-END
