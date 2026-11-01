import time
import numpy as np
import random
import itertools


def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 112.0

    if q == 3 and n == 8 and radius == 1:
        return _solve_ternary_8_1(deadline)

    # General fallback: greedy set cover
    return _general_greedy(q, n, radius, deadline)


def _general_greedy(q, n, radius, deadline):
    total = q ** n
    if total > 200000:
        # Too large for full-space approach; use trivial (all words)
        return [list(c) for c in itertools.product(range(q), repeat=n)]

    # Encode all words
    words = np.array(list(itertools.product(range(q), repeat=n)), dtype=np.uint8)

    # Precompute distance matrix (words vs words)
    # dist[i][j] = hamming distance
    # Vectorized: diff = |words[:, None, :] - words[None, :, :]| summed over last axis
    diff = np.abs(words[:, None, :].astype(np.int16) - words[None, :, :].astype(np.int16))
    dist = diff.sum(axis=2)  # shape (total, total)

    # Balls: ball[i] = set of word indices within distance radius of word i
    # For memory, represent as boolean matrix
    ball_mask = dist <= radius  # (total, total) bool

    # Greedy set cover
    remaining = np.ones(total, dtype=np.uint8)
    selected = []

    while remaining.sum() > 0:
        if time.time() > deadline - 5.0:
            break
        # Gain of each candidate: number of remaining words it covers
        gains = (ball_mask & remaining[None, :].astype(np.bool_)).sum(axis=1)
        max_gain = gains.max()
        if max_gain == 0:
            break
        candidates = np.where(gains == max_gain)[0]
        # Random tie-break for diversity
        pick = candidates[random.randint(0, len(candidates) - 1)]
        selected.append(pick)
        remaining[ball_mask[pick]] = 0

    # Verify coverage
    if remaining.sum() > 0:
        # Add uncovered words directly (ensures validity)
        for i in np.where(remaining)[0]:
            selected.append(int(i))

    return [words[i].tolist() for i in selected]


def _solve_ternary_8_1(deadline):
    n = 8
    q = 3
    total = q ** n  # 6561

    # Try a known algebraic construction first
    det_code = _deterministic_486()
    if _is_valid_cover(det_code, q, n, 1):
        return det_code

    # Greedy set-cover
    words = np.array(list(itertools.product(range(q), repeat=n)), dtype=np.uint8)

    # Precompute balls
    diff = np.abs(words[:, None, :].astype(np.int16) - words[None, :, :].astype(np.int16))
    dist = diff.sum(axis=2)
    ball_mask = dist <= 1  # (6561, 6561) bool

    # Greedy phase
    remaining = np.ones(total, dtype=np.uint8)
    selected = []

    while remaining.sum() > 0:
        if time.time() > deadline - 20.0:
            break
        gains = (ball_mask & remaining[None, :].astype(np.bool_)).sum(axis=1)
        max_gain = gains.max()
        if max_gain == 0:
            break
        candidates = np.where(gains == max_gain)[0]
        pick = candidates[random.randint(0, len(candidates) - 1)]
        selected.append(pick)
        remaining[ball_mask[pick]] = 0

    # Add any uncovered
    if remaining.sum() > 0:
        for i in np.where(remaining)[0]:
            selected.append(int(i))

    # Local search: try to remove redundant codewords
    if time.time() < deadline - 10.0:
        selected = _local_remove(words, ball_mask, selected, deadline - 10.0)

    # Final validity check; if invalid, add missing
    if not _is_valid_cover([words[i].tolist() for i in selected], q, n, 1):
        for i in np.where(remaining)[0]:
            if int(i) not in selected:
                selected.append(int(i))

    # Deduplicate
    seen = set()
    unique = []
    for i in selected:
        t = tuple(words[i])
        if t not in seen:
            seen.add(t)
            unique.append(i)

    return [words[i].tolist() for i in unique]


def _deterministic_486():
    """Try several algebraic constructions; return first that looks plausible."""
    n = 8
    codewords = []
    seen = set()

    patterns = [
        # (a,b,c,d, f4, f5, f6, f7) each fi is a linear form
        lambda a,b,c,d: [a,b,c,d, (a+b)%3, (b+c)%3, (c+d)%3, (a+b+c+d)%3],
        lambda a,b,c,d: [a,b,c,d, (a+2*b)%3, (2*b+c)%3, (c+2*d)%3, (a+2*b+c+2*d)%3],
        lambda a,b,c,d: [a,b,c,d, (2*a+b)%3, (b+2*c)%3, (2*c+d)%3, (2*a+b+2*c+d)%3],
        lambda a,b,c,d: [a,b,c,d, (a+2*b)%3, (b+2*c)%3, (c+2*d)%3, (a+2*b+2*c+d)%3],
        lambda a,b,c,d: [a,b,c,d, (2*a+b)%3, (2*b+c)%3, (2*c+d)%3, (2*a+2*b+2*c+d)%3],
        lambda a,b,c,d: [a,b,c,d, (a+2*b)%3, (2*b+c)%3, (2*c+d)%3, (a+b+c+d)%3],
        lambda a,b,c,d: [a,b,c,d, (2*a+b)%3, (b+2*c)%3, (c+2*d)%3, (2*a+2*b+c+d)%3],
        lambda a,b,c,d: [a,b,c,d, (a+b)%3, (b+2*c)%3, (2*c+d)%3, (a+b+2*c+d)%3],
    ]

    for pat in patterns:
        for a in range(3):
            for b in range(3):
                for c in range(3):
                    for d in range(3):
                        w = pat(a,b,c,d)
                        k = tuple(w)
                        if k not in seen:
                            seen.add(k)
                            codewords.append(list(w))

    if len(codewords) == 486:
        return codewords
    return codewords  # return whatever we have; validity checked by caller


def _is_valid_cover(codewords, q, n, radius):
    """Check that every word in {0..q-1}^n is within radius of some codeword."""
    if len(codewords) == 0:
        return False

    code_arr = np.array(codewords, dtype=np.int16)
    words = np.array(list(itertools.product(range(q), repeat=n)), dtype=np.int16)

    # For each word, check if min distance to any codeword <= radius
    # Vectorized in chunks to save memory
    chunk = 1000
    ok = True
    for start in range(0, len(words), chunk):
        w = words[start:start+chunk]  # (chunk, n)
        c = code_arr  # (m, n)
        d = np.abs(w[:, None, :] - c[None, :, :]).sum(axis=2)  # (chunk, m)
        if (d.min(axis=1) > radius).any():
            ok = False
            break
    return ok


def _local_remove(words, ball_mask, selected, deadline):
    """Try to remove each codeword if coverage is preserved."""
    n_sel = len(selected)
    # coverage_count[w] = number of selected codewords whose ball contains w
    coverage_count = np.zeros(words.shape[0], dtype=np.int32)
    for i in selected:
        coverage_count[ball_mask[i]] += 1

    removed = 0
    changed = True
    while changed and time.time() < deadline:
        changed = False
        for idx in range(n_sel - 1, -1, -1):
            i = selected[idx]
            # Check if removing i leaves any word with coverage 0
            new_count = coverage_count - ball_mask[i].astype(np.int32)
            if (new_count > 0).all():
                coverage_count = new_count
                selected.pop(idx)
                n_sel -= 1
                removed += 1
                changed = True
    return selected
