"""Rewritten program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import itertools
import random
import numpy as np

_DEADLINE = 120.0  # hard wall-clock limit; keep a safety margin
_START = time.time()


def _elapsed():
    return time.time() - _START


def _remaining():
    return _DEADLINE - _elapsed() - 2.0  # 2s safety margin


def _hamming_dist(a, b):
    return sum(1 for x, y in zip(a, b) if x != y)


def _all_neighbors(word, q, n):
    """Yield all words within Hamming distance <= 1 of `word`."""
    yield word
    for pos in range(n):
        for v in range(q):
            if v != word[pos]:
                w = list(word)
                w[pos] = v
                yield tuple(w)


def _is_valid_cover(codewords, q, n, radius):
    """Check that every word in {0..q-1}^n is within radius of some codeword."""
    if radius != 1:
        # Fallback: brute force
        covered = set()
        for cw in codewords:
            for word in itertools.product(range(q), repeat=n):
                if sum(1 for i in range(n) if word[i] != cw[i]) <= radius:
                    covered.add(word)
        return len(covered) == q ** n
    # Fast check for radius 1
    covered = set()
    for cw in codewords:
        for nb in _all_neighbors(cw, q, n):
            covered.add(nb)
    return len(covered) == q ** n


def _coverage_count(codewords, q, n):
    """Count how many words are covered (within distance 1)."""
    covered = set()
    for cw in codewords:
        for nb in _all_neighbors(cw, q, n):
            covered.add(nb)
    return len(covered)


def _tetracode_sum(q, n, radius):
    """Direct sum construction: tetracode on first 4 coords, all on rest."""
    if n < 4:
        return [list(w) for w in itertools.product(range(q), repeat=n)]
    tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
    result = []
    for h in tetracode:
        for tail in itertools.product(range(3), repeat=n - 4):
            result.append(h + list(tail))
    return result


def _greedy_cover(q, n, radius, max_time=5.0):
    """Greedy set-cover: repeatedly add the codeword that covers the most uncovered words."""
    all_words = [list(w) for w in itertools.product(range(q), repeat=n)]
    total = len(all_words)
    covered = [False] * total
    word_index = {tuple(w): i for i, w in enumerate(all_words)}
    codewords = []
    uncovered = total

    # Precompute neighbor lists for each word (distance <= 1)
    neighbors = []
    for w in all_words:
        nbs = set()
        nbs.add(tuple(w))
        for pos in range(n):
            for v in range(q):
                if v != w[pos]:
                    nw = list(w)
                    nw[pos] = v
                    nbs.add(tuple(nw))
        neighbors.append(nbs)

    start = time.time()
    while uncovered > 0 and (time.time() - start) < max_time:
        best_idx = -1
        best_gain = 0
        for i, w in enumerate(all_words):
            gain = sum(1 for nb in neighbors[i] if not covered[word_index[nb]])
            if gain > best_gain:
                best_gain = gain
                best_idx = i
            if best_gain == q * n + 1:  # max possible, can't do better
                break
        if best_idx < 0 or best_gain == 0:
            break
        codewords.append(all_words[best_idx])
        for nb in neighbors[best_idx]:
            idx = word_index[nb]
            if not covered[idx]:
                covered[idx] = True
                uncovered -= 1
    return codewords


def _local_search(codewords, q, n, radius, max_time=60.0):
    """
    Local search with add-remove heuristic.
    Start from a valid cover, try to remove a codeword and see if still covered.
    If not, try replacing it with a different word that restores coverage with less overlap.
    """
    if not _is_valid_cover(codewords, q, n, radius):
        return codewords

    all_words = [tuple(w) for w in itertools.product(range(q), repeat=n)]
    word_index = {w: i for i, w in enumerate(all_words)}
    total = len(all_words)

    # Precompute neighbors (distance <= 1)
    neighbors = []
    for w in all_words:
        nbs = set()
        nbs.add(w)
        for pos in range(n):
            for v in range(q):
                if v != w[pos]:
                    nw = list(w)
                    nw[pos] = v
                    nbs.add(tuple(nw))
        neighbors.append(nbs)

    cw_set = set(tuple(cw) for cw in codewords)
    # Count how many times each word is covered
    cover_count = np.zeros(total, dtype=np.int32)
    for cw in codewords:
        idx = word_index[tuple(cw)]
        for nb in neighbors[idx]:
            cover_count[word_index[nb]] += 1

    uncovered = np.sum(cover_count == 0)
    if uncovered > 0:
        return codewords

    start = time.time()
    improved = True
    while improved and (time.time() - start) < max_time:
        improved = False
        n_cw = len(codewords)
        for i in range(n_cw):
            if _remaining() < 5.0:
                return codewords
            cw_i = tuple(codewords[i])
            # Try removing codeword i
            for nb in neighbors[word_index[cw_i]]:
                cover_count[word_index[nb]] -= 1
            still_covered = np.all(cover_count > 0)
            if still_covered:
                # Remove it
                codewords.pop(i)
                cw_set.discard(cw_i)
                improved = True
                # Restart loop since indices changed
                break
            else:
                # Restore
                for nb in neighbors[word_index[cw_i]]:
                    cover_count[word_index[nb]] += 1
                # Try replacing with a word that covers the uncovered ones
                uncovered_words = [w for w in all_words if cover_count[word_index[w]] == 1]
                if not uncovered_words:
                    continue
                # Find a replacement word that covers all uncovered_words
                best_rep = None
                best_overlap = float('inf')
                # Sample some candidate replacements for speed
                candidates = random.sample(all_words, min(500, total))
                for cand in candidates:
                    cand_nbs = neighbors[word_index[cand]]
                    # Check if cand covers all uncovered_words
                    if all(u in cand_nbs for u in uncovered_words):
                        # Count overlap with existing codewords
                        overlap = 0
                        for j, cw_j in enumerate(codewords):
                            if j == i:
                                continue
                            if tuple(cw_j) in cand_nbs:
                                overlap += 1
                        if overlap < best_overlap:
                            best_overlap = overlap
                            best_rep = cand
                if best_rep is not None:
                    # Perform replacement
                    for nb in neighbors[word_index[cw_i]]:
                        cover_count[word_index[nb]] -= 1
                    for nb in neighbors[word_index[best_rep]]:
                        cover_count[word_index[nb]] += 1
                    codewords[i] = list(best_rep)
                    cw_set.discard(cw_i)
                    cw_set.add(best_rep)
                    improved = True
                    break
        if _remaining() < 5.0:
            break
    return codewords


def construct(q, n, radius):
    global _START
    _START = time.time()

    # Initial solution: tetracode sum
    codewords = _tetracode_sum(q, n, radius)

    # Local search to reduce size
    codewords = _local_search(codewords, q, n, radius, max_time=_remaining() - 10.0)

    # If still too big, try greedy from scratch with remaining time
    if len(codewords) > 73 and _remaining() > 15.0:
        greedy_cw = _greedy_cover(q, n, radius, max_time=_remaining() - 10.0)
        if len(greedy_cw) < len(codewords):
            codewords = greedy_cw
            codewords = _local_search(codewords, q, n, radius, max_time=_remaining() - 5.0)

    # Final validation
    if not _is_valid_cover(codewords, q, n, radius):
        # Fallback to tetracode sum
        codewords = _tetracode_sum(q, n, radius)

    return codewords
# EVOLVE-BLOCK-END
