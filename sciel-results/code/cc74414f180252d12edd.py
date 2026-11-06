"""Constraint-based backtracking solver for the 9-dim snake-in-the-box."""
import time
import numpy as np

N = 9
FULL = 1 << N

# Precompute neighbors for each vertex (list of 9 neighbors)
_NEIGH = np.empty((FULL, N), dtype=np.int64)
for _v in range(FULL):
    for _b in range(N):
        _NEIGH[_v, _b] = _v ^ (1 << _b)
del _v, _b


def _greedy_seed():
    path, used = [0], {0}
    while True:
        cur, nxt = path[-1], None
        for b in range(N):
            v = cur ^ (1 << b)
            if v in used or any((v ^ p).bit_count() == 1 for p in path[:-1]):
                continue
            nxt = v
            break
        if nxt is None:
            return path
        path.append(nxt)
        used.add(nxt)


# EVOLVE-BLOCK-START
def construct():
    start = time.time()
    deadline = start + 115.0  # safety margin under 120s

    best_path = []
    best_len = 0

    used = np.zeros(FULL, dtype=bool)
    forbidden = np.zeros(FULL, dtype=bool)

    path = [0]
    used[0] = True
    forbidden[0] = True

    def score_candidate(v, cur):
        cnt = 0
        for b in range(N):
            u = _NEIGH[v, b]
            if used[u] or forbidden[u]:
                continue
            if (u ^ cur).bit_count() == 1:
                continue
            cnt += 1
        return cnt

    def backtrack(cur):
        nonlocal best_path, best_len
        if time.time() > deadline:
            return
        if len(path) > best_len:
            best_len = len(path)
            best_path = list(path)
        candidates = []
        for b in range(N):
            v = _NEIGH[cur, b]
            if not forbidden[v]:
                candidates.append(v)
        if not candidates:
            return
        candidates.sort(key=lambda v: score_candidate(v, cur))
        for v in candidates:
            if time.time() > deadline:
                return
            old_forbidden = forbidden.copy()
            old_used = used.copy()
            used[v] = True
            forbidden[v] = True
            for b in range(N):
                forbidden[_NEIGH[cur, b]] = True
            path.append(v)
            backtrack(v)
            path.pop()
            forbidden[:] = old_forbidden
            used[:] = old_used
            if time.time() > deadline:
                return

    backtrack(0)

    if best_len > 0:
        return best_path
    return _greedy_seed()
# EVOLVE-BLOCK-END
