"""New design: spectral initialization + fast 2-swap local search + tabu escape."""
# EVOLVE-BLOCK-START
import time
import numpy as np


def solve(a, b):
    n = len(a)
    deadline = time.time() + 115.0  # safety margin under 120s wall-clock

    # ---------- cost helper (exact, O(n^2)) ----------
    def cost(p):
        # p is a permutation of locations for facilities
        return int(np.sum(a * b[np.ix_(p, p)]))

    # ---------- spectral / rank initialization ----------
    def init_spectral():
        # Use leading eigenvectors of a and b to align structure.
        try:
            w_a, V_a = np.linalg.eigh(a)
            w_b, V_b = np.linalg.eigh(b)
        except Exception:
            return init_rank()
        # take top few eigenvectors (largest magnitude)
        k = min(8, n)
        ia = np.argsort(-np.abs(w_a))[:k]
        ib = np.argsort(-np.abs(w_b))[:k]
        Ea = V_a[:, ia]
        Eb = V_b[:, ib]
        # assign facilities to locations by projecting onto shared spectral frame
        # facility i -> location with max dot product of its row in Ea vs Eb
        # build assignment greedily
        used = np.zeros(n, dtype=bool)
        p = np.empty(n, dtype=np.int64)
        order = np.argsort(-np.abs(Ea).sum(axis=1), kind="stable")
        for i in order:
            scores = Eb.T @ Ea[:, i]
            scores[used] = -np.inf
            j = int(np.argmax(scores))
            p[i] = j
            used[j] = True
        return p

    def init_rank():
        heavy = np.argsort(-a.sum(axis=1), kind="stable")
        central = np.argsort(b.sum(axis=1), kind="stable")
        p = np.empty(n, dtype=np.int64)
        p[heavy] = central
        return p

    # ---------- fast swap delta (O(n)) ----------
    # delta of swapping positions of facilities i and j
    def swap_delta(p, i, j):
        li = p[i]
        lj = p[j]
        # cost change: only rows/cols i and j in b change
        # new contribution uses b with li<->lj
        ai = a[i]
        aj = a[j]
        # old: a[i,j]*b[li,lj] + a[j,i]*b[lj,li] + sum_k a[i,k]b[li,pk] + sum_k a[j,k]b[lj,pk]
        # we compute delta by considering the 2x2 plus cross terms
        # cross: for each k != i,j: a[i,k](b[li,pk]-b[lj,pk]) + a[j,k](b[lj,pk]-b[li,pk])
        # plus the i-j pair term
        b_li = b[li]
        b_lj = b[lj]
        # cross sum over k
        diff = b_li - b_lj  # shape (n,)
        # sum_k a[i,k]*diff[k] - sum_k a[j,k]*diff[k]
        cross = ai @ diff - aj @ diff
        # pair term: old a[i,j]*b[li,lj] + a[j,i]*b[lj,li]; new a[i,j]*b[lj,li] + a[j,i]*b[li,lj]
        pair_old = a[i, j] * b_li[lj] + a[j, i] * b_lj[li]
        pair_new = a[i, j] * b_lj[li] + a[j, i] * b_li[lj]
        # diagonal terms a[i,i]*b[li,li] unchanged; a[j,j]*b[lj,lj] unchanged
        return int(cross + (pair_new - pair_old))

    # ---------- 2-swap local search ----------
    def local_search(p, tabu_list=None, tabu_len=0):
        cur = cost(p)
        best_p = p.copy()
        best_c = cur
        improved = True
        while improved:
            if time.time() > deadline:
                break
            improved = False
            best_d = 0
            best_i = -1
            best_j = -1
            # single pass over pairs
            for i in range(n):
                if time.time() > deadline:
                    break
                pi = p[i]
                ai = a[i]
                for j in range(i + 1, n):
                    if tabu_list and i in tabu_list and j in tabu_list:
                        continue
                    d = swap_delta(p, i, j)
                    if d < best_d:
                        best_d = d
                        best_i = i
                        best_j = j
            if best_i >= 0 and best_d < 0:
                # apply
                p[best_i], p[best_j] = p[best_j], p[best_i]
                cur += best_d
                if cur < best_c:
                    best_c = cur
                    best_p = p.copy()
                improved = True
                if tabu_list is not None:
                    tabu_list.add(best_i)
                    tabu_list.add(best_j)
                    if len(tabu_list) > tabu_len:
                        # pop oldest (approx)
                        to_remove = list(tabu_list)[:len(tabu_list) - tabu_len]
                        for x in to_remove:
                            tabu_list.discard(x)
        return best_p, best_c

    # ---------- main ----------
    p = init_spectral()
    p, c = local_search(p)
    best_p = p
    best_c = c

    # tabu-based escape / restarts
    while time.time() < deadline - 2.0:
        # perturb
        q = best_p.copy()
        rng = np.random.default_rng()
        for _ in range(6):
            i, j = rng.integers(0, n, size=2)
            if i != j:
                q[i], q[j] = q[j], q[i]
        tabu = set()
        q, cq = local_search(q, tabu_list=tabu, tabu_len=30)
        if cq < best_c:
            best_c = cq
            best_p = q

    return best_p.tolist()
# EVOLVE-BLOCK-END
