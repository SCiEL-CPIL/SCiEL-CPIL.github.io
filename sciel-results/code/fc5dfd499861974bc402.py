import time
import random
import sys
from collections import deque

import numpy as np

N = 12
FULL_MASK = (1 << N) - 1
SIZE = 1 << N
DEADLINE_SECONDS = 82.0
_SEED = 123456789


def compute_threshold(trans_a, trans_b):
    """Exact shortest reset word length via BFS on the subset automaton.
    trans_a/trans_b: length-N iterables of ints in 0..N-1. Returns 0 if not synchronizing."""
    a = np.fromiter(trans_a, dtype=np.uint8, count=N).reshape(-1, 1)
    b = np.fromiter(trans_b, dtype=np.uint8, count=N).reshape(-1, 1)
    idx = np.arange(N, dtype=np.uint8).reshape(-1, 1)
    bits = 1 << idx
    na = np.zeros(SIZE, dtype=np.int64)
    nb = np.zeros(SIZE, dtype=np.int64)
    for m in range(1, SIZE):
        lsb = m & (-m)
        i = lsb.bit_length() - 1
        r = m ^ lsb
        na[m] = na[r] | (1 << int(a[i, 0]))
        nb[m] = nb[r] | (1 << int(b[i, 0]))
    dist = np.full(SIZE, -1, dtype=np.int32)
    dist[FULL_MASK] = 0
    q = deque([FULL_MASK])
    while q:
        cur = q.popleft()
        if cur & (cur - 1) == 0:
            return int(dist[cur])
        d = int(dist[cur]) + 1
        for nxt in (int(na[cur]), int(nb[cur])):
            if dist[nxt] == -1:
                dist[nxt] = d
                q.append(nxt)
    return 0


def _is_singleton(m):
    return m & (m - 1) == 0


def _eval_multi(cands, deadline):
    """Vectorized threshold evaluation for many candidate automata at once."""
    if not cands:
        return []
    arr = np.asarray(cands, dtype=np.uint8)
    K = arr.shape[0]
    A = arr[:, 0, :]
    B = arr[:, 1, :]
    bits = (1 << np.arange(N, dtype=np.uint8))
    Aexp = A[:, :, None] * bits[None, None, :]
    Bexp = B[:, :, None] * bits[None, None, :]
    na = np.zeros((K, SIZE), dtype=np.int64)
    nb = np.zeros((K, SIZE), dtype=np.int64)
    for m in range(1, SIZE):
        lsb = m & (-m)
        i = lsb.bit_length() - 1
        r = m ^ lsb
        na[:, m] = na[:, r] | Aexp[:, i, 0]
        nb[:, m] = nb[:, r] | Bexp[:, i, 0]
    dist = np.full((K, SIZE), -1, dtype=np.int32)
    dist[:, FULL_MASK] = 0
    q = deque([FULL_MASK] * K)
    res = [0] * K
    while q:
        cur = q.popleft()
        if _is_singleton(cur):
            continue
        d = dist[:, cur]
        active = d == -1
        if not active.any():
            continue
        newd = (d + 1)[active]
        na_next = na[active, cur]
        nb_next = nb[active, cur]
        dist[active, cur] = newd
        q.extend(cur for _ in range(int(active.sum())))
        for arr_next in (na_next, nb_next):
            hit = np.nonzero(arr_next)[0]
            for h in hit:
                if dist[h, int(arr_next[h])] == -1:
                    dist[h, int(arr_next[h])] = int(newd[h])
                    q.append(int(arr_next[h]))
    for k in range(K):
        best = 0
        for s in range(SIZE):
            if _is_singleton(s):
                dd = int(dist[k, s])
                if dd >= 0 and dd < best:
                    best = dd
        res[k] = best
    return res


def _cycle_with_defects(base, defects, base_idx):
    out = [base] * N
    for s, d in defects:
        out[s] = d
    return out


def _defect_candidates():
    cands = []
    for ca in range(1, N):
        for cb in range(1, N):
            cands.append([_cycle_with_defects((lambda i: (i + ca) % N), [(0, 0)], 0),
                          _cycle_with_defects((lambda i: (i + cb) % N), [(0, 0)], 0)])
    for ca in range(1, N):
        for cb in range(1, N):
            for da in range(N):
                for db in range(N):
                    if da == 0 and db == 0:
                        continue
                    A = _cycle_with_defects((lambda i: (i + ca) % N), [(0, da)], 0)
                    B = _cycle_with_defects((lambda i: (i + cb) % N), [(0, db)], 0)
                    cands.append([A, B])
    for ca in range(1, N):
        for cb in range(1, N):
            for s in range(1, N):
                for da in range(N):
                    for db in range(N):
                        A = _cycle_with_defects((lambda i: (i + ca) % N), [(s, da)], s)
                        B = _cycle_with_defects((lambda i: (i + cb) % N), [(s, db)], s)
                        cands.append([A, B])
    return cands


def _hill_climb(cur, deadline, batch=24):
    cur_a = cur[0].copy()
    cur_b = cur[1].copy()
    cur_score = compute_threshold(cur_a, cur_b)
    if cur_score == 0:
        return None, None, 0
    best_a = cur_a.copy()
    best_b = cur_b.copy()
    best_score = cur_score
    st = np.random.RandomState(_SEED)
    while time.time() < deadline:
        pool = []
        for _ in range(batch):
            aa = cur_a.copy()
            bb = cur_b.copy()
            if st.rand() < 0.5:
                s = st.randint(0, N)
                nd = st.randint(0, N)
                if nd != aa[s]:
                    aa[s] = nd
            else:
                s = st.randint(0, N)
                nd = st.randint(0, N)
                if nd != bb[s]:
                    bb[s] = nd
            pool.append([aa, bb])
        scores = _eval_multi(pool, deadline)
        bi = int(np.argmax(scores))
        bs = scores[bi]
        if bs > best_score:
            best_score = bs
            best_a = pool[bi][0].copy()
            best_b = pool[bi][1].copy()
            cur_a = best_a
            cur_b = best_b
    return best_a, best_b, best_score


def construct():
    t0 = time.time()
    deadline = t0 + DEADLINE_SECONDS
    random.seed(_SEED)

    cands = _defect_candidates()
    random.shuffle(cands)

    best_a = None
    best_b = None
    best_score = 0
    batch = 48
    i = 0
    while i < len(cands) and time.time() < deadline - 15.0:
        chunk = cands[i:i + batch]
        scores = _eval_multi(chunk, deadline)
        for j, s in enumerate(scores):
            if s > best_score:
                best_score = s
                best_a = chunk[j][0].copy()
                best_b = chunk[j][1].copy()
        i += batch

    if best_a is None:
        best_a = [i for i in range(N)]
        best_a = [(x + 1) % N for x in best_a]
        best_b = [0] * N
        best_score = compute_threshold(best_a, best_b)

    # Annealed hill climbing with periodic restarts near the best solution
    remaining = deadline - time.time()
    n_rounds = max(1, int(remaining / 6.0))
    for r in range(n_rounds):
        if time.time() > deadline - 2.0:
            break
        rnd_deadline = min(deadline - 2.0, t0 + (DEADLINE_SECONDS - 2.0) * (r + 1) / n_rounds)
        if r > 0 and random.random() < 0.35:
            start_a = best_a.copy()
            start_b = best_b.copy()
            for _ in range(1):
                if random.random() < 0.5:
                    s = random.randint(0, N - 1)
                    start_a[s] = random.randint(0, N - 1)
                else:
                    s = random.randint(0, N - 1)
                    start_b[s] = random.randint(0, N - 1)
            start_score = compute_threshold(start_a, start_b)
            if start_score == 0:
                start_a = best_a.copy()
                start_b = best_b.copy()
            else:
                best_a = start_a
                best_b = start_b
                best_score = start_score
        else:
            start_a = best_a.copy()
            start_b = best_b.copy()
        ra, rb, rs = _hill_climb([start_a, start_b], rnd_deadline)
        if rs > best_score:
            best_score = rs
            best_a = ra
            best_b = rb

    if best_score == 0 or best_a is None:
        a = [(i + 1) % N for i in range(N)]
        b = [0] * N
        return [a, b]

    return [list(best_a), list(best_b)]


if __name__ == "__main__":
    a, b = construct()
    print(f"Best score found: {compute_threshold(a, b)}")
    print(f"A: {a}")
    print(f"B: {b}")
