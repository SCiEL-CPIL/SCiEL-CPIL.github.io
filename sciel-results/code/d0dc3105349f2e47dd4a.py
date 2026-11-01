"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import itertools
import numpy as np
import time


def _fallback_construct(q, n, radius):
    tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
    return [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]


def _polish_redundant(codewords, covered_indices, total_words, deadline):
    """Remove codewords whose covered words are all covered by other codewords."""
    if not codewords:
        return codewords
    # Map each codeword to its index in the covered_indices array
    # covered_indices is indexed by the integer representation of the word
    # We need to find the integer index for each codeword
    n = len(codewords[0])
    powers = [3 ** (n - 1 - i) for i in range(n)]

    def to_int(word):
        return sum(w * p for w, p in zip(word, powers))

    cw_ints = [to_int(cw) for cw in codewords]
    cw_int_set = set(cw_ints)

    # For each codeword, compute which words it uniquely covers
    # A word is uniquely covered by codeword i if no other codeword covers it
    # We use a count array: for each word in the space, count how many codewords cover it
    # But that's expensive. Instead, we check each codeword's coverage against others.

    # Efficient approach: for each codeword, check if removing it leaves all its words covered.
    # Process codewords in order; skip if already removed.
    active = [True] * len(codewords)
    removed = 0

    # Precompute the covered word indices for each codeword
    cw_covered = [covered_indices[cw_ints[i]] for i in range(len(codewords))]

    # Iterate multiple passes until no more removals
    for _pass in range(3):
        if time.time() > deadline:
            break
        any_removed = False
        for i in range(len(codewords)):
            if not active[i]:
                continue
            # Check if all words covered by codeword i are covered by some other active codeword
            # For each word w in cw_covered[i], check if any other active codeword j != i covers w
            # This is expensive. Use a faster check:
            # Build a set of words covered by codeword i
            # Then for each such word, check if it's in the coverage of another active codeword
            # Instead, use a counting approach per pass:
            pass

        # Better approach: count coverage
        # For each word in the space, count how many active codewords cover it
        # If a word is covered by exactly 1 active codeword, that codeword is essential
        # If all words in codeword i's coverage are covered by >= 2 active codewords, i is removable

        # Build coverage count for all words
        cov_count = np.zeros(total_words, dtype=np.int8)
        for i in range(len(codewords)):
            if active[i]:
                cov_count[cw_covered[i]] += 1

        # Now for each active codeword, check if all its words have cov_count >= 2
        # (i.e., they're covered by at least one other active codeword)
        for i in range(len(codewords)):
            if not active[i]:
                continue
            if time.time() > deadline:
                return [codewords[j] for j in range(len(codewords)) if active[j]]
            # Check if all words in cw_covered[i] have cov_count >= 2
            # If yes, codeword i is redundant
            if np.all(cov_count[cw_covered[i]] >= 2):
                active[i] = False
                any_removed = True
                removed += 1
                # Update cov_count
                cov_count[cw_covered[i]] -= 1

        if not any_removed:
            break

    return [codewords[j] for j in range(len(codewords)) if active[j]]


def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 130

    if q == 3 and n == 8 and radius == 1:
        total_words = 3 ** n
        powers = np.array([3 ** (n - 1 - i) for i in range(n)], dtype=np.int64)

        all_indices = np.arange(total_words, dtype=np.int64)
        digits = np.zeros((total_words, n), dtype=np.int8)
        for i in range(n):
            digits[:, i] = (all_indices // powers[i]) % 3

        num_covered = 1 + n * (q - 1)
        covered_indices = np.zeros((total_words, num_covered), dtype=np.int64)
        for idx in range(total_words):
            covered_indices[idx, 0] = idx
            pos = 1
            for i in range(n):
                d = digits[idx, i]
                for v in range(q):
                    if v != d:
                        covered_indices[idx, pos] = idx + (v - d) * powers[i]
                        pos += 1

        uncovered = np.ones(total_words, dtype=bool)
        codewords = []

        while uncovered.any():
            if time.time() > deadline:
                return _fallback_construct(q, n, radius)

            counts = uncovered[covered_indices].sum(axis=1)
            best_idx = int(np.argmax(counts))
            if counts[best_idx] == 0:
                break

            uncovered[covered_indices[best_idx]] = False
            codewords.append(digits[best_idx].tolist())

        if not uncovered.any():
            # Polish: remove redundant codewords
            codewords = _polish_redundant(codewords, covered_indices, total_words, deadline)
            return codewords
        return _fallback_construct(q, n, radius)

    return _fallback_construct(q, n, radius)
# EVOLVE-BLOCK-END
