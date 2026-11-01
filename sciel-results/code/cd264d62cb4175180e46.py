"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 13]

import time
import numpy as np
import random

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140  # safety margin before 150s

    n, m, p = DIMENSIONS
    nm, mp, pn = n * m, m * p, p * n

    # Target triples (a,b,c) that must evaluate to 1
    target = [(i * m + j, j * p + k, k * n + i)
              for i in range(n) for j in range(m) for k in range(p)]
    ta = np.array([t[0] for t in target], dtype=np.int64)
    tb = np.array([t[1] for t in target], dtype=np.int64)
    tc = np.array([t[2] for t in target], dtype=np.int64)

    def make_schoolbook():
        R = n * m * p
        U = np.zeros((nm, R), dtype=int)
        V = np.zeros((mp, R), dtype=int)
        W = np.zeros((pn, R), dtype=int)
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
        return U, V, W

    def verify(U, V, W, R):
        # (a,b,c) -> sum_r U[a,r]*V[b,r]*W[c,r] must equal 1 for all target triples
        vals = (U[ta, :R] * V[tb, :R] * W[tc, :R]).sum(axis=1)
        return bool(np.all(vals == 1))

    U, V, W = make_schoolbook()
    R = U.shape[1]
    best_U, best_V, best_W, best_R = U, V, W, R

    def supports(U, V, col):
        us = tuple(np.nonzero(U[:, col])[0])
        vs = tuple(np.nonzero(V[:, col])[0])
        return us, vs

    def remove_column(U, V, W, R, c):
        idx = np.arange(R)
        mask = idx != c
        return U[:, mask].copy(), V[:, mask].copy(), W[:, mask].copy(), R - 1

    def merge_columns(U, V, W, R, c1, c2):
        # combine two columns sharing U,V support into one; drop the other
        new = W[:, c1].copy()
        new[c1] = W[c1, c1] + W[c1, c2]
        idx = np.arange(R)
        mask = idx != c2
        Wn = W[:, mask].copy()
        Wn[:, c1 if c1 < c2 else c1] = new
        # careful: after dropping c2, index c1 stays if c1<c2, else shifts
        # Rebuild cleanly:
        cols = [c for c in range(R) if c != c2]
        Wnew = np.zeros((pn, R - 1), dtype=int)
        for new_c, old_c in enumerate(cols):
            Wnew[:, new_c] = W[:, old_c]
        # set merged column at position of c1 within cols
        pos = cols.index(c1)
        Wnew[:, pos] = new
        return U[:, mask].copy(), V[:, mask].copy(), Wnew, R - 1

    # Greedy search: repeatedly try removals and merges, keep best valid.
    # Randomize column order to escape local minima; keep a time guard.
    rng = random.Random(12345)
    improved = True
    while improved and (time.time() < deadline):
        improved = False
        cols = list(range(R))
        rng.shuffle(cols)
        for c in cols:
            if time.time() > deadline:
                break
            # Try removal
            Ut, Vt, Wt, Rt = remove_column(U, V, W, R, c)
            if verify(Ut, Vt, Wt, Rt):
                U, V, W, R = Ut, Vt, Wt, Rt
                best_U, best_V, best_W, best_R = U.copy(), V.copy(), W.copy(), R
                improved = True
                break

        if not improved:
            # Try merges: find pairs with matching (U,V) support
            sup = {c: supports(U, V, c) for c in range(R)}
            # group by support
            groups = {}
            for c in range(R):
                groups.setdefault(sup[c], []).append(c)
            merged_any = False
            for key, members in groups.items():
                if time.time() > deadline:
                    break
                if len(members) < 2:
                    continue
                c1, c2 = members[0], members[1]
                Ut, Vt, Wt, Rt = merge_columns(U, V, W, R, c1, c2)
                if verify(Ut, Vt, Wt, Rt):
                    U, V, W, R = Ut, Vt, Wt, Rt
                    best_U, best_V, best_W, best_R = U.copy(), V.copy(), W.copy(), R
                    merged_any = True
                    break
            if merged_any:
                improved = True

    # Convert to lists of lists (integer entries)
    def to_list(A):
        return A.tolist()

    return {"U": to_list(best_U[:, :best_R]),
            "V": to_list(best_V[:, :best_R]),
            "W": to_list(best_W[:, :best_R])}
# EVOLVE-BLOCK-END
