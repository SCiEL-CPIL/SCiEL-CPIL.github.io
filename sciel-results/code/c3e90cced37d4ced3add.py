"""Ternary covering code K_3(6,1): greedy + local-search (swap/insert/delete).
Target: q=3, n=6, radius=1.  Best known = 73 words.
Strategy:
  1. Fast greedy construction of a valid cover (usually ~85-90 words).
  2. Local search: try single-word swaps, single deletions, and insertions
     to reduce the number of codewords while preserving full coverage.
  3. Deadline guard to stay within 120 s.
"""
import time
import itertools
import random
import numpy as np

_DEADLINE = None


def _now():
    return time.perf_counter()


def _word_to_idx(word, q, n):
    idx = 0
    for v in word:
        idx = idx * q + v
    return idx


def _idx_to_word(idx, q, n):
    word = [0] * n
    for k in range(n - 1, -1, -1):
        word[k] = idx % q
        idx //= q
    return word


def _ball_indices(word, q, n, powers):
    """Return list of flat indices for the word and all words at Hamming distance 1."""
    base = 0
    for k, v in enumerate(word):
        base = base * q + v
    out = [base]
    for i in range(n):
        old = word[i]
        p = powers[i]
        for new in range(q):
            if new != old:
                out.append(base + (new - old) * p)
    return out


def _build_cover(codewords, q, n, powers):
    """Return a boolean coverage array of length q**n."""
    total = q ** n
    covered = np.zeros(total, dtype=np.bool_)
    for w in codewords:
        idxs = _ball_indices(w, q, n, powers)
        covered[np.array(idxs, dtype=np.int64)] = True
    return covered


def _greedy_construct(q, n, radius, powers, deadline):
    """Greedy set-cover: repeatedly add the word whose ball covers the most uncovered words."""
    total = q ** n
    uncovered = np.ones(total, dtype=np.bool_)
    codewords = []
    # Precompute all words as a list of lists
    all_words = [_idx_to_word(i, q, n) for i in range(total)]
    # Precompute ball index arrays for every word (memory: 729 * 19 ints, fine)
    ball_arrays = []
    for i in range(total):
        w = all_words[i]
        idxs = _ball_indices(w, q, n, powers)
        ball_arrays.append(np.array(idxs, dtype=np.int64))

    while np.any(uncovered) and _now() < deadline:
        # Find the word whose ball covers the most uncovered positions
        best_gain = -1
        best_idx = -1
        # Sample a subset if total is large; for 729 it's fine to scan all
        for i in range(total):
            gain = int(np.count_nonzero(uncovered[ball_arrays[i]]))
            if gain > best_gain:
                best_gain = gain
                best_idx = i
            if gain == 0:
                continue
        if best_idx < 0 or best_gain <= 0:
            break
        codewords.append(all_words[best_idx])
        uncovered[ball_arrays[best_idx]] = False

    return codewords, uncovered


def _local_search(codewords, q, n, powers, deadline, rng):
    """Local search: try deletions, swaps, and insertions to minimize |codewords|.
    Returns the best codeword list found."""
    total = q ** n
    all_words = [_idx_to_word(i, q, n) for i in range(total)]
    ball_arrays = []
    for i in range(total):
        w = all_words[i]
        idxs = _ball_indices(w, q, n, powers)
        ball_arrays.append(np.array(idxs, dtype=np.int64))

    # Build index->word map
    word_to_i = {}
    for i, w in enumerate(all_words):
        word_to_i[tuple(w)] = i

    def _is_valid(cw_list):
        covered = np.zeros(total, dtype=np.bool_)
        for w in cw_list:
            i = word_to_i[tuple(w)]
            covered[ball_arrays[i]] = True
        return bool(np.all(covered))

    def _try_delete(cw_list):
        """Try removing each codeword; if still valid, remove it."""
        m = len(cw_list)
        for i in range(m):
            if _now() > deadline:
                return cw_list, False
            candidate = cw_list[:i] + cw_list[i + 1:]
            if _is_valid(candidate):
                return candidate, True
        return cw_list, False

    def _try_swap(cw_list):
        """Try replacing one codeword with another to reduce size or keep same size but enable future deletions.
        Here we only do swap if it allows a subsequent deletion (i.e., swap+delete)."""
        m = len(cw_list)
        for i in range(m):
            if _now() > deadline:
                return cw_list, False
            for new_w in all_words:
                if new_w == cw_list[i]:
                    continue
                # Try: replace cw_list[i] with new_w, then try to delete one word
                candidate = cw_list[:i] + [new_w] + cw_list[i + 1:]
                # Try deleting each word in candidate
                for j in range(len(candidate)):
                    candidate2 = candidate[:j] + candidate[j + 1:]
                    if _is_valid(candidate2):
                        return candidate2, True
        return cw_list, False

    def _try_insert(cw_list):
        """Try inserting a new word then deleting two existing words (net -1).
        Only useful if we can find a word that, when added, allows removing 2 others."""
        m = len(cw_list)
        if m < 2:
            return cw_list, False
        for new_w in all_words:
            if _now() > deadline:
                return cw_list, False
            if new_w in cw_list:
                continue
            candidate = cw_list + [new_w]
            # Try removing pairs
            for i in range(len(candidate)):
                for j in range(i + 1, len(candidate)):
                    candidate2 = candidate[:i] + candidate[i + 1:j] + candidate[j + 1:]
                    if _is_valid(candidate2):
                        return candidate2, True
        return cw_list, False

    best = list(codewords)
    improved = True
    while improved and _now() < deadline:
        improved = False
        # Try deletions
        cand, ok = _try_delete(best)
        if ok:
            best = cand
            improved = True
            continue
        # Try swaps (with subsequent deletion)
        cand, ok = _try_swap(best)
        if ok:
            best = cand
            improved = True
            continue
        # Try insert+double-delete
        cand, ok = _try_insert(best)
        if ok:
            best = cand
            improved = True
            continue
    return best


def construct(q, n, radius):
    global _DEADLINE
    start = _now()
    deadline = start + 115.0  # safety margin
    _DEADLINE = deadline

    if q == 3 and n == 6 and radius == 1:
        powers = [q ** (n - 1 - i) for i in range(n)]
        rng = random.Random(42)

        # Step 1: Greedy construction
        codewords, uncovered = _greedy_construct(q, n, radius, powers, deadline)

        # Ensure full coverage
        if not np.all(~uncovered):
            # Add remaining uncovered words (shouldn't happen with greedy)
            for i in range(q ** n):
                if uncovered[i]:
                    codewords.append(_idx_to_word(i, q, n))
            uncovered = np.zeros(q ** n, dtype=np.bool_)

        # Step 2: Local search
        if _now() < deadline:
            codewords = _local_search(codewords, q, n, powers, deadline, rng)

        # Final validity check
        covered = _build_cover(codewords, q, n, powers)
        if not np.all(covered):
            # Emergency: add uncovered words
            for i in range(q ** n):
                if not covered[i]:
                    codewords.append(_idx_to_word(i, q, n))

        return codewords

    # Generic fallback
    total = q ** n
    if total <= 200000:
        return [_idx_to_word(i, q, n) for i in range(total)]
    return [_idx_to_word(i, q, n) for i in range(total)]
