import time
import random
import numpy as np

N = 40

def _build_S():
    rng = random.Random(987654)
    all_masks = [m for m in range(1, 64)]
    rng.shuffle(all_masks)
    from itertools import combinations
    def no_bad(s, m):
        for c in combinations(s, 5):
            x = 0
            for e in c:
                x ^= e
            if x == m:
                return False
        return True
    S = []
    for m in all_masks:
        if len(S) >= 40:
            break
        if no_bad(S, m):
            S.append(m)
    return S[:40]

S = _build_S()
Sset = set(S)

def _has_c6(adj):
    A = adj.astype(np.int8)
    A2 = A @ A
    B = (A2 > 0).astype(np.int8)
    C = B @ B
    D = C @ A
    return bool((D.diagonal() > 0).any())

def _try_add(adj, i, j):
    if i == j or adj[i, j]:
        return False
    adj[i, j] = adj[j, i] = 1
    ok = not _has_c6(adj)
    if not ok:
        adj[i, j] = adj[j, i] = 0
    return ok

def construct():
    t0 = time.time()
    DEADLINE = 140.0
    edges = []
    adj = np.zeros((N, N), dtype=np.int8)
    # algebraic seed
    for i in range(N):
        for j in range(i + 1, N):
            if (i ^ j) in Sset:
                adj[i, j] = adj[j, i] = 1
                edges.append([i, j])
    # greedy fill
    changed = True
    while changed and time.time() < DEADLINE - 20:
        changed = False
        for i in range(N):
            for j in range(i + 1, N):
                if not adj[i, j] and _try_add(adj, i, j):
                    edges.append([i, j])
                    changed = True
    result = []
    for i in range(N):
        for j in range(i + 1, N):
            if adj[i, j]:
                result.append([i, j])
    return result
