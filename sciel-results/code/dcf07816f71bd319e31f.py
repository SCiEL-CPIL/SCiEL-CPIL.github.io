import time
import random
import numpy as np

N = 40
DEADLINE = 150.0
SAFETY = 8.0

def _build_A(edge_set):
    A = np.zeros((N, N), dtype=np.int8)
    for (u, v) in edge_set:
        A[u, v] = 1
        A[v, u] = 1
    return A

def _simple_path5(A, u, v):
    Au = A[u]
    for a in range(N):
        if a == u or a == v or not Au[a]:
            continue
        Ab = A[a]
        for b in range(N):
            if b == u or b == v or b == a or not Ab[b]:
                continue
            Ac = A[b]
            for c in range(N):
                if c == u or c == v or c == a or c == b or not Ac[c]:
                    continue
                Ad = A[c]
                for d in range(N):
                    if d == u or d == v or d == a or d == b or d == c:
                        continue
                    if Ad[d] and A[d, v]:
                        return True
    return False

def _has_c6_fast(A):
    A5 = A
    for _ in range(4):
        A5 = A5 @ A
    cand = np.argwhere((A5 > 0) & (A > 0))
    if len(cand) == 0:
        return False
    for u, v in cand:
        if _simple_path5(A, int(u), int(v)):
            return True
    return False

def _would_create_c6(A, u, v):
    A[u, v] = 1
    A[v, u] = 1
    res = _has_c6_fast(A)
    A[u, v] = 0
    A[v, u] = 0
    return res

def _local_search(edge_set, deadline, rng):
    A = _build_A(edge_set)
    stall = 0
    while time.time() < deadline:
        improved = False
        for u in range(N):
            for v in range(u + 1, N):
                if A[u, v]:
                    continue
                if _would_create_c6(A, u, v):
                    continue
                A[u, v] = 1
                A[v, u] = 1
                edge_set.add((u, v))
                improved = True
                break
            if improved:
                break
        if not improved:
            swapped = False
            elist = list(edge_set)
            rng.shuffle(elist)
            for rem in elist:
                ru, rv = rem
                A[ru, rv] = 0
                A[rv, ru] = 0
                edge_set.discard(rem)
                for u in range(N):
                    for v in range(u + 1, N):
                        if A[u, v]:
                            continue
                        if _would_create_c6(A, u, v):
                            continue
                        A[u, v] = 1
                        A[v, u] = 1
                        edge_set.add((u, v))
                        swapped = True
                        break
                    if swapped:
                        break
                if swapped:
                    break
                edge_set.add(rem)
                A[ru, rv] = 1
                A[rv, ru] = 1
            if swapped:
                stall = 0
                continue
            else:
                stall += 1
                if stall > 40:
                    break
    return edge_set

def construct():
    start = time.time()
    deadline = start + DEADLINE - SAFETY
    rng = random.Random(12345)

    # Seed: 8 disjoint K5
    seed = set()
    for k in range(8):
        for a in range(5):
            for b in range(a + 1, 5):
                seed.add((5 * k + a, 5 * k + b))
    best_set = _local_search(set(seed), deadline, rng)
    best_count = len(best_set)

    # Random restarts
    while time.time() < deadline - 2:
        p = rng.uniform(0.25, 0.45)
        es = set()
        for u in range(N):
            for v in range(u + 1, N):
                if rng.random() < p:
                    es.add((u, v))
        # clean C6s
        A = _build_A(es)
        guard = 0
        while _has_c6_fast(A) and guard < 500 and time.time() < deadline - 2:
            guard += 1
            A5 = A
            for _ in range(4):
                A5 = A5 @ A
            cand = np.argwhere((A5 > 0) & (A > 0))
            cand = cand.tolist()
            rng.shuffle(cand)
            done = False
            for u, v in cand:
                if _simple_path5(A, u, v):
                    es.discard((min(u, v), max(u, v)))
                    A[u, v] = 0
                    A[v, u] = 0
                    done = True
                    break
            if not done:
                break
        es = _local_search(es, deadline, rng)
        if len(es) > best_count:
            best_count = len(es)
            best_set = es

    return [[u, v] for (u, v) in best_set]
