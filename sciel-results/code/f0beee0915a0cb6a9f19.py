import time
import random
import numpy as np

N = 40
DEADLINE = time.time() + 110.0  # explicit safety margin vs 120s wall-clock


def _has_c6(A, u, v):
    """Exact: does adding edge (u,v) create a C6?
    A C6 through the new edge <=> a length-3 path u-x-y-v in G-{u,v},
    x,y distinct, not in {u,v}. Count via one matrix product without
    copying: mask out u and v by zeroing their rows/cols on a scratch copy
    (40x40 int64 copy is ~13KB, fast)."""
    B = A.copy()
    B[u, :] = 0
    B[:, u] = 0
    B[v, :] = 0
    B[:, v] = 0
    B3 = (B @ B) @ B
    # x==y terms vanish since B[x,x]=0, so B3[u,v] counts only x!=y paths.
    return B3[u, v] > 0


def _try_greedy_addition(A, order):
    B = A.copy()
    m = int(B.sum() // 2)
    for (u, v) in order:
        if B[u, v]:
            continue
        if not _has_c6(B, u, v):
            B[u, v] = 1
            B[v, u] = 1
            m += 1
    return B, m


def construct():
    start = time.time()

    all_pairs = [(u, v) for u in range(N) for v in range(u + 1, N)]

    best_A = None
    best_m = 0

    def greedy_random(seed):
        r = random.Random(seed)
        order = all_pairs[:]
        r.shuffle(order)
        return _try_greedy_addition(np.zeros((N, N), dtype=np.int64), order)

    def local_search(A0, max_add_rounds=60, try_2opt=False):
        B = A0.copy()
        m = int(B.sum() // 2)
        for _ in range(max_add_rounds):
            added = False
            for (u, v) in all_pairs:
                if B[u, v]:
                    continue
                if not _has_c6(B, u, v):
                    B[u, v] = 1
                    B[v, u] = 1
                    m += 1
                    added = True
            if not added:
                break
            if try_2opt:
                # attempt a single 1-remove / refill move
                moved = False
                for (ru, rv) in all_pairs:
                    if not B[ru, rv]:
                        continue
                    B2 = B.copy()
                    B2[ru, rv] = 0
                    B2[rv, ru] = 0
                    m2 = m - 1
                    for (u, v) in all_pairs:
                        if B2[u, v]:
                            continue
                        if not _has_c6(B2, u, v):
                            B2[u, v] = 1
                            B2[v, u] = 1
                            m2 += 1
                    if m2 > m:
                        B = B2
                        m = m2
                        moved = True
                        break
                if not moved:
                    break
        return B, m

    # Phase 1: several randomized greedy passes.
    for s in range(30):
        A, m = greedy_random(1000 + s)
        if m > best_m:
            best_m = m
            best_A = A.copy()
        if time.time() > DEADLINE - 30:
            break

    # Phase 2: local search on the best.
    A, m = local_search(best_A, max_add_rounds=40, try_2opt=True)
    if m > best_m:
        best_m = m
        best_A = A.copy()

    # Phase 3: randomized restarts + light local search while time remains.
    seed = 5000
    while time.time() < DEADLINE - 25:
        A, m = greedy_random(seed)
        seed += 1
        A, m = local_search(A, max_add_rounds=12, try_2opt=False)
        if m > best_m:
            best_m = m
            best_A = A.copy()

    # Phase 4: final polish with 2-opt on the global best.
    if time.time() < DEADLINE - 12:
        A, m = local_search(best_A, max_add_rounds=80, try_2opt=True)
        if m > best_m:
            best_m = m
            best_A = A.copy()

    edges = [[u, v] for u in range(N) for v in range(u + 1, N) if best_A[u, v]]
    return edges
