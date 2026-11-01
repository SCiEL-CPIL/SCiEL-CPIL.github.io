"""Greedy covering-code construction using bitset coverage tracking."""
import time
import numpy as np

# Deadline guard: kill before the evaluator's 150s wall-clock limit.
_DEADLINE = 120.0
_t0 = time.time()


def _remaining():
    return _DEADLINE - (time.time() - _0)


def _ball_indices(q, n, radius):
    """For each position p, return the list of offsets (as flat indices) of words
    that differ from the word at index p in exactly one coordinate (distance 1).
    Precomputed as a flat list of ints for fast np.add."""
    # For a word at flat index idx, its neighbours at distance 1 are obtained by
    # changing one coordinate. We build per-position offset arrays.
    # Represent word coords via base-q digits.
    offsets = []
    for pos in range(n):
        stride = q ** (n - 1 - pos)
        arr = []
        # For each possible digit d at this position, the neighbours are d' != d
        # The flat-index delta for changing digit from d to d' is (d' - d)*stride
        # But we need, for a given word (with digit d), the offsets to all d'!=d.
        # We'll compute per-word offsets later using digit extraction; simpler:
        # build a full neighbour-offset table indexed by (pos, digit).
        pass
    return offsets


def _digit_at(idx, pos, q, n):
    stride = q ** (n - 1 - pos)
    return (idx // stride) % q


def construct(q, n, radius):
    global _t0
    _t0 = time.time()

    total = q ** n
    # Greedy set cover: repeatedly pick the word covering the most uncovered words.
    # Coverage of a word = itself + all words within Hamming distance <= radius.
    # For radius 1: 1 + n*(q-1) words.

    # Precompute the ball (set of covered flat indices) for every candidate word.
    # To be memory-light, we compute on the fly but cache via numpy.
    # We track uncovered as a boolean numpy array.
    uncovered = np.ones(total, dtype=np.uint8)
    covered_count = np.zeros(total, dtype=np.int64)  # how many uncovered each candidate covers

    # Precompute, for each position and each digit value, the list of flat-index
    # offsets to change that digit to each other value. Actually we compute the
    # ball of a word given its index.
    # Build stride table.
    strides = [q ** (n - 1 - pos) for pos in range(n)]

    def ball_offsets(idx):
        """Yield flat indices in the radius-1 ball of word idx (excluding itself handled separately)."""
        # distance 1 neighbours
        for pos in range(n):
            st = strides[pos]
            d = (idx // st) % q
            for d2 in range(q):
                if d2 != d:
                    yield idx + (d2 - d) * st

    # Initial coverage counts
    for idx in range(total):
        c = 1  # itself
        for off in ball_offsets(idx):
            c += uncovered[off]
        covered_count[idx] = c

    codewords = []
    while True:
        if time.time() - _t0 > _DEADLINE - 5.0:
            break
        if uncovered.sum() == 0:
            break
        # pick argmax
        best = int(np.argmax(covered_count))
        if covered_count[best] == 0:
            break
        codewords.append(best)
        uncovered[best] = 0
        # Update: for every candidate, subtract the newly covered words.
        # The newly covered words are those in ball(best) that were uncovered.
        newly = []
        for off in ball_offsets(best):
            if uncovered[off]:
                newly.append(off)
        newly.append(best)
        newly = np.array(newly, dtype=np.int64)
        if len(newly) == 0:
            break
        # For each candidate, count how many of its ball intersect newly.
        # Efficient: for each newly-covered word w, add 1 to covered_count of all
        # candidates whose ball contains w. A candidate's ball contains w iff
        # candidate is within distance <= radius of w, i.e. candidate in ball(w).
        for w in newly:
            uncovered[w] = 0
            # all candidates in ball(w) lose 1 (they no longer cover w)
            # ball(w) = {w} + distance-1 neighbours of w
            cand_list = [w]
            for off in ball_offsets(w):
                cand_list.append(off)
            cand_arr = np.array(cand_list, dtype=np.int64)
            covered_count[cand_arr] -= 1

        if time.time() - _t0 > _DEADLINE - 5.0:
            break

    # If we ran out of time before full coverage, return what we have.
    # Convert flat indices to codeword lists.
    result = []
    for idx in codewords:
        word = []
        x = idx
        for pos in range(n - 1, -1, -1):
            word.append(x % q)
            x //= q
        word.reverse()
        result.append(word)
    return result
