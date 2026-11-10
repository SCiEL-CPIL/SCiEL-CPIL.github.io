# EVOLVE-BLOCK-START
import time
import itertools
import numpy as np

_T1 = time.time()
_HARD = _T1 + 150.0


def _deadline(margin):
    return (time.time() + margin) < _HARD


def _fallback_base(q, n, radius):
    """Guaranteed-valid base: direct-sum of tetracode for q=3, n>=4, radius>=1.
    Otherwise the full space (always a valid cover)."""
    if q != 3 or n < 4 or radius < 1:
        return [list(t) for t in itertools.product(range(q), repeat=n)]
    tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
    tail_len = n - 4
    out = []
    for tail in itertools.product(range(q), repeat=tail_len):
        for h in tetracode:
            out.append(h + list(tail))
    return out


def _build_neighbors(q, n):
    """Return (all_words, word_idx, nbr_int).
    nbr_int is an int32 array of shape (N, 1+2q): each row is the index of the
    word plus its single-coordinate flips (the radius-1 ball)."""
    all_words = list(itertools.product(range(q), repeat=n))
    word_idx = {w: i for i, w in enumerate(all_words)}
    N = q ** n
    nbr = np.empty((N, 1 + 2 * q), dtype=np.int32)
    for i, w in enumerate(all_words):
        nbr[i, 0] = i
        for j in range(n):
            for d in (1, 2):
                b = list(w)
                b[j] = (b[j] + d) % q
                nbr[i, 1 + 2 * j + (d - 1)] = word_idx[tuple(b)]
    return all_words, word_idx, nbr


def _greedy_cover(q, n, nbr):
    """Fast vectorized greedy set cover over the radius-1 balls. Returns selected
    codeword indices (a valid cover, since it only stops when nothing is uncovered)."""
    N = nbr.shape[0]
    uncovered = np.ones(N, dtype=bool)
    cov = np.zeros(N, dtype=np.int32)
    sel = []
    deadline_g = _HARD - 10.0
    while uncovered.any():
        gain = (~cov[nbr[uncovered]]).sum(axis=1)  # uncovered balls covered by each word
        best = int(np.argmax(gain))
        if gain[best] <= 0:
            break  # safety (should not happen)
        w = int(np.nonzero(uncovered)[0][best])
        sel.append(w)
        cov[nbr[w]] = 1
        uncovered[nbr[w]] = False
        if not _deadline(8.0):
            break
    return sel


def _remove_repair(q, n, nbr, sel):
    """Fast vectorized remove-then-repair: repeatedly drop a codeword whose removal
    uncovers <=2 words, then re-cover those words with the best-gain candidate."""
    N = nbr.shape[0]
    sel_set = set(sel)
    cov = np.zeros(N, dtype=np.int32)
    for ci in sel:
        cov[nbr[ci]] += 1

    deadline_guard = _HARD - 8.0
    improved = True
    while improved and time.time() < deadline_guard:
        improved = False
        for ci in sel:
            ball = nbr[ci]
            if cov[ball].min() > 1:
                continue  # removing is safe (no word drops to 0)
            uncovered = ball[cov[ball] == 1]
            if len(uncovered) > 2:
                continue
            # remove ci
            cov[ball] -= 1
            sel.remove(ci)
            sel_set.discard(ci)
            improved = True
            # repair each uncovered word
            for uc in uncovered:
                if cov[uc] == 0:
                    cand = nbr[uc]  # candidates (ball of uc)
                    gain = (~cov[nbr[cand]]).sum(axis=1)
                    best = int(np.argmax(gain))
                    bw = int(cand[best])
                    if bw not in sel_set:
                        sel.append(bw)
                        sel_set.add(bw)
                        cov[nbr[bw]] += 1
            if not _deadline(8.0):
                break
    return sel


def construct(q, n, radius):
    global _HARD
    _HARD = time.time() + 150.0
    if n < 0:
        return []
    if n == 0:
        return [[]]

    # --- Guaranteed-valid base ---
    base = _fallback_base(q, n, radius)
    N = q ** n

    # --- For the target (and similar) cases: fast greedy set cover ---
    if radius >= 1 and N <= 200000 and _deadline(100.0):
        all_words, word_idx, nbr = _build_neighbors(q, n)
        # dedupe base to indices (base is always a valid cover)
        base_sel = []
        seen = set()
        for cw in base:
            i = word_idx[tuple(cw)]
            if i not in seen:
                seen.add(i)
                base_sel.append(i)

        # greedy set cover from scratch (fast, vectorized)
        greedy_sel = _greedy_cover(q, n, nbr)
        # keep whichever is smaller
        if len(greedy_sel) < len(base_sel):
            base_sel = greedy_sel

        # optional: remove-repair to shrink further
        if _deadline(60.0) and len(base_sel) <= 20000:
            base_sel = _remove_repair(q, n, nbr, base_sel)

        # safety: verify coverage; if somehow invalid, add the missing words
        cov = np.zeros(N, dtype=np.int32)
        for ci in base_sel:
            cov[nbr[ci]] += 1
        missing = np.nonzero(cov == 0)[0]
        for m in missing:
            if int(m) not in seen:
                seen.add(int(m))
                base_sel.append(int(m))

        return [list(all_words[i]) for i in base_sel]

    # --- Fallback path (large space / no time): just return the guaranteed base ---
    return [list(c) for c in base]
# EVOLVE-BLOCK-END
