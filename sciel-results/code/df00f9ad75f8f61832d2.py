# EVOLVE-BLOCK-START
import time
import itertools
import numpy as np


def _base_cover(q, n, radius):
    # Direct sum of the ternary tetracode (9 words of length 4, perfect covering radius 1)
    # with every word on the remaining n - 4 coordinates: 9 * 3**(n - 4) words.
    tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(q) for b in range(q)]
    if n <= 4:
        return [list(h[:n]) for h in tetracode]
    return [h + list(tail) for h in tetracode
            for tail in itertools.product(range(q), repeat=n - 4)]


def _is_cover(codewords, q, n, radius):
    if not codewords:
        return n == 0
    C = np.array(codewords, dtype=np.int32).reshape(-1, n)
    words = np.array(list(itertools.product(range(q), repeat=n)), dtype=np.int32)
    diff = (words[:, None, :] != C[None, :, :])
    dist = diff.sum(axis=2)
    return bool(np.all(dist.min(axis=1) <= radius))


def _greedy_cover(q, n, radius, deadline):
    if n == 0:
        return [[]]

    words = np.array(list(itertools.product(range(q), repeat=n)), dtype=np.int32)
    M = words.shape[0]
    # Compute pairwise Hamming distances efficiently
    # For each position, compute which pairs differ
    diff = (words[:, None, :] != words[None, :, :])
    dist = diff.sum(axis=2)
    del diff
    ball = (dist <= radius)
    del dist
    del words

    covered = np.zeros(M, dtype=bool)
    code_idx = []
    remaining = M

    # Precompute ball sizes for initial greedy choice
    ball_sizes = ball.sum(axis=1)

    while remaining > 0:
        if time.time() > deadline:
            break
        # Choose the codeword that covers the most uncovered words
        gains = ball_sizes.copy()
        gains[covered] = -1
        j = int(np.argmax(gains))
        if gains[j] <= 0:
            # Pick any uncovered word
            uncovered = np.where(~covered)[0]
            if len(uncovered) == 0:
                break
            j = int(uncovered[0])
        code_idx.append(j)
        newcov = ball[j] & ~covered
        new_count = int(newcov.sum())
        if new_count > 0:
            covered |= newcov
            remaining -= new_count
            # Update ball sizes: for each ball, subtract newly covered words
            ball_sizes = ball.sum(axis=1)

    return [list(b) for b in ball[code_idx, :]]


def _local_search_polish(codewords, q, n, radius, deadline):
    """Try to reduce codeword count by removing redundant codewords."""
    if not codewords:
        return codewords
    M = q ** n
    if M > 100000:
        return codewords  # Too large for this approach

    words = np.array(list(itertools.product(range(q), repeat=n)), dtype=np.int32)
    C = np.array(codewords, dtype=np.int32)
    num_cw = len(C)

    # Compute coverage: for each word, which codewords cover it
    diff = (words[:, None, :] != C[None, :, :])
    dist = diff.sum(axis=2)
    del diff
    covers = dist <= radius  # M x num_cw boolean matrix

    # For each word, count how many codewords cover it
    cover_counts = covers.sum(axis=1)  # M array

    best_code = codewords[:]
    improved = True
    while improved and time.time() < deadline:
        improved = False
        # Try removing each codeword
        for i in range(len(best_code)):
            if time.time() > deadline:
                break
            # Check if removing codeword i still covers everything
            # A word is still covered if it has at least one other codeword covering it
            # covers[:, i] is True for words covered by codeword i
            # We need: for all words, (cover_counts[word] - (1 if covers[word,i] else 0)) >= 1
            # i.e., no word is exclusively covered by codeword i
            exclusive = covers[:, i] & (cover_counts == 1)
            if not exclusive.any():
                # Safe to remove
                new_covers = covers[:, list(range(len(best_code))) if False else [j for j in range(len(best_code)) if j != i]]
                # Actually, let's just mark it and rebuild
                remove_idx = i
                best_code = [best_code[j] for j in range(len(best_code)) if j != i]
                # Recompute cover matrix
                C_new = np.array(best_code, dtype=np.int32)
                diff_new = (words[:, None, :] != C_new[None, :, :])
                dist_new = diff_new.sum(axis=2)
                del diff_new
                covers = dist_new <= radius
                del dist_new
                cover_counts = covers.sum(axis=1)
                improved = True
                break
    return best_code


def construct(q, n, radius):
    start = time.time()
    deadline = start + 110.0          # explicit guard, safety margin vs 120s wall-clock

    if n == 0:
        return [[]]

    # Baseline that is guaranteed valid and always available to return.
    base = _base_cover(q, n, radius)
    base = base[:1000000]             # keep memory bounded for large inputs
    if not _is_cover(base, q, n, radius):
        # Fallback: exhaustive (valid but large); only for pathological small q,n.
        base = [list(w) for w in itertools.product(range(q), repeat=n)]
    best = base

    # Greedy refinement: try to beat the baseline within the time budget.
    try:
        cand = _greedy_cover(q, n, radius, deadline)
        if len(cand) < len(best) and _is_cover(cand, q, n, radius):
            best = cand
    except Exception:
        pass

    # Local search polish: try to remove redundant codewords
    if time.time() < deadline:
        try:
            polished = _local_search_polish(best, q, n, radius, deadline)
            if len(polished) < len(best) and _is_cover(polished, q, n, radius):
                best = polished
        except Exception:
            pass

    if time.time() > deadline:
        return best
    return best
# EVOLVE-BLOCK-END
