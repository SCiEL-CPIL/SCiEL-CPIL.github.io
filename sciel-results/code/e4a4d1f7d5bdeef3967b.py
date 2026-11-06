"""Minimum maximal no-isosceles set: randomized greedy + local-search removal,
vectorized distance precomputation, time-guarded. Returns list of (x,y)."""
import time
import random
import numpy as np

N = 16
DEADLINE = 320.0
_PTS = np.array([(x, y) for y in range(N) for x in range(N)], dtype=np.int32)
_NP = N * N
_DX = _PTS[:, 0:1] - _PTS[:, 0:1].T
_DY = _PTS[:, 1:2] - _PTS[:, 1:2].T
D = (_DX * _DX + _DY * _DY).astype(np.int64)


def _build_blocking():
    blocking = [set() for _ in range(_NP)]
    for a in range(_NP):
        for b in range(a + 1, _NP):
            dab = D[a, b]
            cand = set()
            cand.update(np.where(D[:, a] == D[:, b])[0].tolist())
            cand.update(np.where(D[a, :] == dab)[0].tolist())
            cand.update(np.where(D[b, :] == dab)[0].tolist())
            cand.discard(a)
            cand.discard(b)
            for p in cand:
                blocking[p].add((a, b))
    return blocking


_BLOCKING = _build_blocking()


def _valid_add(p, chosen):
    for (a, b) in _BLOCKING[p]:
        if a in chosen and b in chosen:
            return False
    return True


def _blocked(q, chosen):
    for (a, b) in _BLOCKING[q]:
        if a in chosen and b in chosen:
            return True
    return False


def _greedy(order):
    chosen = set()
    for p in order:
        if _valid_add(p, chosen):
            chosen.add(p)
    return chosen


def _local_search(best, time_ok):
    while time_ok():
        improved = False
        for p in list(best):
            if not time_ok():
                break
            cand = best - {p}
            unblocked = [q for q in range(_NP) if q not in cand and not _blocked(q, cand)]
            if not unblocked:
                best = cand
                improved = True
                break
        if not improved:
            break
    return best


def construct():
    start = time.time()

    def time_ok():
        return time.time() - start < DEADLINE

    rng = random.Random(12345)
    base = _greedy(list(range(_NP)))
    best = _local_search(set(base), time_ok)
    best_len = len(best)

    while time_ok():
        order = list(range(_NP))
        rng.shuffle(order)
        s = _greedy(order)
        s = _local_search(set(s), time_ok)
        if len(s) < best_len:
            best = s
            best_len = len(s)

    return [tuple(_PTS[i]) for i in sorted(best)]
