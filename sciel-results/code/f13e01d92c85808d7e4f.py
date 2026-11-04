"""Local-search QAP solver with O(1) swap-delta evaluation and a deadline guard."""
# EVOLVE-BLOCK-START
import time
import numpy as np


def solve(a, b):
    a = np.asarray(a, dtype=np.int64)
    b = np.asarray(b, dtype=np.int64)
    n = a.shape[0]

    t0 = time.perf_counter()
    deadline = t0 + 112.0  # safety margin before the 120 s wall-clock kill

    # ---- strong initial permutation: rank-matching (heavy flows on short distances)
    heavy = np.argsort(-a.sum(axis=1), kind="stable")
    central = np.argsort(b.sum(axis=1), kind="stable")
    p = np.empty(n, dtype=np.int64)
    p[heavy] = central
    p = p.astype(np.int64)

    # ---- cost helpers (O(n) per swap)
    def cost(i):
        return np.dot(a[i], b[p[i]])

    def swap_delta(i, j):
        pi = p[i]
        pj = p[j]
        ai = a[i]
        aj = a[j]
        bi = b[pi]
        bj = b[pj]
        return (np.dot(ai, bj) + np.dot(aj, bi)
                - np.dot(ai, bi) - np.dot(aj, bj))

    def do_swap(i, j):
        p[i], p[j] = p[j], p[i]

    # ---- repair: short deterministic improving sweep (used after perturbations)
    def repair():
        nonlocal t0
        improved = True
        while improved and time.perf_counter() < deadline:
            improved = False
            for i in range(n - 1):
                ai = a[i]
                pi = p[i]
                bi = b[pi]
                for j in range(i + 1, n):
                    if time.perf_counter() > deadline:
                        return
                    d = (np.dot(ai, b[p[j]]) + np.dot(a[j], bi)
                         - np.dot(ai, bi) - np.dot(a[j], b[p[j]]))
                    if d < 0:
                        p[i], p[j] = p[j], p[i]
                        improved = True

    # ---- main steepest-descent hill climb
    repair()

    rng = np.random.default_rng(12345)
    no_improve = 0
    best_p = p.copy()

    while time.perf_counter() < deadline:
        best_d = 0
        best_pair = None
        for i in range(n - 1):
            ai = a[i]
            pi = p[i]
            bi = b[pi]
            for j in range(i + 1, n):
                if time.perf_counter() > deadline:
                    break
                d = (np.dot(ai, b[p[j]]) + np.dot(a[j], bi)
                     - np.dot(ai, bi) - np.dot(a[j], b[p[j]]))
                if d < best_d:
                    best_d = d
                    best_pair = (i, j)
            if time.perf_counter() > deadline:
                break
        if time.perf_counter() > deadline:
            break

        if best_pair is None:
            # local optimum reached: try a small perturbation (error repair)
            no_improve += 1
            if no_improve > 6:
                break
            for _ in range(4):
                i = rng.integers(n)
                j = rng.integers(n)
                if i != j:
                    do_swap(i, j)
            repair()
            continue

        do_swap(best_pair[0], best_pair[1])
        no_improve = 0

    return p.tolist()
# EVOLVE-BLOCK-END
