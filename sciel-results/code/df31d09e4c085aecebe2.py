# EVOLVE-BLOCK-START
import time
import numpy as np

_DEADLINE = 150.0
_MARGIN = 8.0


def _build_space(q, n):
    """Return (space, ball_sigs). space: (q**n, n) int8. ball_sigs: (q**n, n) int64."""
    space = np.zeros((q ** n, n), dtype=np.int8)
    for i in range(n):
        space[:, i] = (np.arange(q ** n) // (q ** (n - 1 - i))) % q
    # ball signature of a word: for each coordinate c, the set of values within radius 1
    ball_sigs = np.zeros((q ** n, n), dtype=np.int64)
    for c in range(n):
        vals = space[:, c].astype(np.int64)
        sig = np.zeros(q, dtype=np.int64)
        sig[(vals - 1) % q] = 1
        sig[vals] = 1
        sig[(vals + 1) % q] = 1
        ball_sigs[:, c] = sig
    return space, ball_sigs


def _covered_mask(space, ball_sigs, centers_idx):
    """Boolean array: which space words are within radius 1 of any center."""
    if len(centers_idx) == 0:
        return np.zeros(len(space), dtype=bool)
    cb = ball_sigs[centers_idx]                       # (m, n)
    cb = (cb == 1).astype(np.uint8)
    sig = cb[:, 0]
    for c in range(1, len(space[0])):
        sig = (sig << 1) | cb[:, c]
    sig = np.unique(sig)
    covered = np.zeros(len(space), dtype=bool)
    if len(sig):
        for s in sig:
            covered |= np.all(ball_sigs == s, axis=1)
    return covered


def _best_center(space, ball_sigs, uncovered):
    """Index of the word whose ball covers the most uncovered words (0 if none)."""
    if not np.any(uncovered):
        return 0
    cb = ball_sigs[uncovered]                         # (u, n)
    cb = (cb == 1).astype(np.uint8)
    sig = cb[:, 0]
    for c in range(1, len(space[0])):
        sig = (sig << 1) | cb[:, c]
    sig = np.unique(sig)
    if not len(sig):
        return 0
    counts = np.zeros(len(sig), dtype=np.int64)
    for i, s in enumerate(sig):
        counts[i] = np.count_nonzero(np.all(ball_sigs == s, axis=1) & uncovered)
    best = int(np.argmax(counts))
    if counts[best] <= 0:
        return 0
    return int(np.where(np.all(ball_sigs == sig[best], axis=1))[0][0])


def _greedy(space, ball_sigs, seed_idx, rng):
    covered = _covered_mask(space, ball_sigs, np.array([seed_idx]))
    centers = [seed_idx]
    while not np.all(covered):
        uncovered = ~covered
        c = _best_center(space, ball_sigs, uncovered)
        centers.append(c)
        covered = _covered_mask(space, ball_sigs, np.array(centers))
    return centers


def _reduce(space, ball_sigs, centers):
    centers = list(centers)
    improved = True
    while improved:
        improved = False
        m = len(centers)
        for i in range(m):
            without = [j for j in range(m) if j != i]
            cov = _covered_mask(space, ball_sigs, np.array(without))
            if np.all(cov):
                centers = without
                improved = True
                break
            else:
                # gap exists: find best single repair center
                gap = ~cov
                c = _best_center(space, ball_sigs, gap)
                if c in centers:
                    continue
                newc = centers + [c]
                newcov = _covered_mask(space, ball_sigs, np.array(newc))
                if np.all(newcov):
                    centers = newc
                    improved = True
                    break
    return centers


def construct(q, n, radius):
    t0 = time.time()
    space, ball_sigs = _build_space(q, n)
    N = len(space)
    rng = np.random.default_rng(12345)

    # baseline: greedy from word 0
    centers0 = _greedy(space, ball_sigs, 0, rng)
    best = centers0

    def time_ok():
        return (time.time() - t0) < (_DEADLINE - _MARGIN)

    # multiple randomized restarts
    max_restarts = 24
    for r in range(max_restarts):
        if not time_ok():
            break
        seed = int(rng.integers(0, N))
        centers = _greedy(space, ball_sigs, seed, rng)
        if len(centers) < len(best):
            best = centers
        if not time_ok():
            break

    # post-hoc reduction on the best
    if time_ok():
        best = _reduce(space, ball_sigs, best)

    return [list(map(int, space[i])) for i in best]
# EVOLVE-BLOCK-END
