"""Greedy covering-code construction for K_q(n, radius) with a time guard."""
# EVOLVE-BLOCK-START
import time
import numpy as np


def _all_covered(q, n, radius, code, space):
    """Return True if every space word is within `radius` of some codeword."""
    if not code:
        return space.size == 0
    covered = np.zeros(space.shape, dtype=bool)
    for c in code:
        idx = _ball_indices(c, q, n, radius, space)
        covered[idx] = True
        if covered.all():
            return True
    return covered.all()


def _ball_indices(word, q, n, radius, space):
    """Indices (into `space`) of all words within Hamming distance `radius` of `word`."""
    if radius <= 0:
        return np.array([space[word]], dtype=np.int64)
    if radius >= n:
        return np.arange(space.size, dtype=np.int64)
    base = np.array(word, dtype=np.int64)
    acc = np.zeros(space.size, dtype=bool)
    # distance 0
    acc[space[base]] = True
    # build up to radius using one-position changes
    prev = [base]
    for _ in range(radius):
        new = []
        for w in prev:
            for pos in range(n):
                for v in range(q):
                    if v == w[pos]:
                        continue
                    nw = w.copy()
                    nw[pos] = v
                    new.append(nw)
        # dedupe
        seen = set()
        uniq = []
        for w in new:
            key = int(np.dot(w, np.array([q ** i for i in range(n)], dtype=np.int64)))
            if key not in seen:
                seen.add(key)
                uniq.append(w)
        for w in uniq:
            acc[space[w]] = True
        prev = uniq
    return np.nonzero(acc)[0]


def _init_covered(code, q, n, radius, space):
    cov = np.zeros(space.size, dtype=bool)
    for c in code:
        cov[_ball_indices(c, q, n, radius, space)] = True
    return cov


def construct(q, n, radius):
    start = time.time()
    deadline = start + 115.0  # safety margin vs 120s kill

    if n == 0:
        return [[]]

    # base positional weights for word -> integer index
    weights = np.array([q ** i for i in range(n)], dtype=np.int64)
    space = np.empty(q ** n, dtype=np.int64)
    for w in range(q ** n):
        idx = 0
        x = w
        for i in range(n):
            idx += (x % q) * weights[i]
            x //= q
        space[w] = idx

    # candidate words: all space words
    all_words = []
    for w in range(q ** n):
        vec = []
        x = w
        for i in range(n):
            vec.append(x % q)
            x //= q
        all_words.append(vec)

    code = []
    covered = np.zeros(space.size, dtype=bool)
    remaining = space.size

    # Precompute ball index arrays for every candidate once (fast, one-time).
    balls = np.empty((q ** n, 0), dtype=np.int64)
    ball_list = []
    for w in range(q ** n):
        bi = _ball_indices(all_words[w], q, n, radius, space)
        ball_list.append(bi)

    # Greedy: repeatedly pick the candidate covering the most uncovered points.
    while remaining > 0:
        if time.time() > deadline:
            break
        best_i = -1
        best_gain = -1
        for i in range(q ** n):
            bi = ball_list[i]
            gain = np.count_nonzero(~covered[bi])
            if gain > best_gain:
                best_gain = gain
                best_i = i
        if best_i < 0 or best_gain <= 0:
            break
        code.append(all_words[best_i])
        covered[ball_list[best_i]] = True
        remaining -= best_gain

    # Final safety: if not fully covered within budget, fall back to the whole space
    # (always a valid covering code).
    if remaining > 0:
        # try a quick completion if we have a little time left
        if time.time() < deadline - 5.0:
            for i in range(q ** n):
                if time.time() > deadline - 2.0:
                    break
                if not covered.all():
                    bi = ball_list[i]
                    if np.any(~covered[bi]):
                        code.append(all_words[i])
                        covered[bi] = True
        if not covered.all():
            # ultimate fallback: return every word (guaranteed valid)
            return all_words

    # dedupe (greedy picks distinct words, but be safe)
    seen = set()
    out = []
    for c in code:
        key = tuple(c)
        if key not in seen:
            seen.add(key)
            out.append(list(c))
    return out
# EVOLVE-BLOCK-END
