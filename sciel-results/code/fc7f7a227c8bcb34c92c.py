"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import itertools
import time
import random
import numpy as np


def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 110  # safety margin before 120s kill

    # Precompute all words as integers for fast Hamming distance
    total = q ** n
    # Convert word to int
    def word_to_int(w):
        v = 0
        for i in range(n):
            v = v * q + w[i]
        return v

    def int_to_word(v):
        w = [0] * n
        for i in range(n - 1, -1, -1):
            w[i] = v % q
            v //= q
        return w

    # Precompute ball membership: for each codeword index, which words it covers
    # ball_mask[cw_idx] is a set of word-ints within radius
    # But total=729, so we can use numpy arrays

    # Precompute all words
    all_words = [list(w) for w in itertools.product(range(q), repeat=n)]
    all_ints = [word_to_int(w) for w in all_words]

    # Precompute Hamming distance between any two words (729x729 matrix)
    # For radius=1, two words are within distance 1 if they differ in at most 1 position
    # Precompute: for each word, the set of words within distance radius
    # Use numpy for speed
    words_arr = np.array(all_words, dtype=np.int32)  # shape (total, n)

    # Compute pairwise Hamming distances
    # For small n, we can do it directly
    # diff[i,j] = number of positions where word_i != word_j
    # Use broadcasting: shape (total, total, n) might be too big for large total
    # For total=729, n=6: 729*729*6 = ~3.2M ints, fine
    diff = (words_arr[:, np.newaxis, :] != words_arr[np.newaxis, :, :]).sum(axis=2)
    # covered[i,j] = True if dist(i,j) <= radius
    covered_matrix = diff <= radius  # shape (total, total)

    # Greedy set cover: repeatedly pick the codeword that covers the most uncovered words
    uncovered = np.ones(total, dtype=bool)
    codewords = []
    covered_set = np.zeros(total, dtype=bool)

    while uncovered.any():
        if time.time() > deadline - 5:
            break
        # For each candidate codeword, count how many currently-uncovered words it covers
        # count[i] = number of j where covered_matrix[i,j] and uncovered[j]
        counts = (covered_matrix & uncovered[np.newaxis, :]).sum(axis=1)
        # Pick the one with max count
        best = int(np.argmax(counts))
        codewords.append(all_words[best])
        covered_set |= covered_matrix[best]
        uncovered = ~covered_set

    # Local search: try to remove codewords if still covered
    def is_covered(cw_list):
        """Check if all words are covered by cw_list"""
        if not cw_list:
            return total == 0
        cw_arr = np.array(cw_list, dtype=np.int32)
        # For each word, check if any codeword is within radius
        # diff2[i,j] = hamming distance between word_i and cw_j
        # Use broadcasting
        d = (words_arr[:, np.newaxis, :] != cw_arr[np.newaxis, :, :]).sum(axis=2)
        return (d.min(axis=1) <= radius).all()

    # Try removing codewords
    improved = True
    while improved:
        if time.time() > deadline - 5:
            break
        improved = False
        for i in range(len(codewords)):
            if time.time() > deadline - 5:
                break
            # Try removing codewords[i]
            candidate = codewords[:i] + codewords[i+1:]
            if is_covered(candidate):
                codewords = candidate
                improved = True
                break

    # Try swaps: replace one codeword with another
    for _ in range(50):
        if time.time() > deadline - 5:
            break
        improved = False
        for i in range(len(codewords)):
            if time.time() > deadline - 5:
                break
            # Try replacing codewords[i] with a random other word
            for _ in range(20):
                if time.time() > deadline - 5:
                    break
                j = random.randint(0, total - 1)
                candidate = codewords[:i] + [all_words[j]] + codewords[i+1:]
                if is_covered(candidate):
                    codewords = candidate
                    improved = True
                    break
            if improved:
                break
        if not improved:
            break

    # Final check: verify coverage
    if not is_covered(codewords):
        # Fallback to direct sum construction
        if q == 3 and n == 6 and radius == 1:
            tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
            return [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]
        # Generic fallback: all words
        return all_words

    return codewords
# EVOLVE-BLOCK-END
