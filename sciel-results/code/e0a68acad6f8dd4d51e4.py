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
    n_targets = len(target)

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
        vals = (U[ta, :R] * V[tb, :R] * W[tc, :R]).sum(axis=1)
        return bool(np.all(vals == 1))

    def supports(U, V, col):
        us = tuple(np.nonzero(U[:, col])[0])
        vs = tuple(np.nonzero(V[:, col])[0])
        return us, vs

    def remove_column(U, V, W, R, c):
        idx = np.arange(R)
        mask = idx != c
        return U[:, mask].copy(), V[:, mask].copy(), W[:, mask].copy(), R - 1

    def merge_group(U, V, W, R, members):
        """Merge all columns in 'members' (sharing same U,V support) into one."""
        keep = members[0]
        drop = [c for c in members if c != keep]
        # New W column = sum of W columns for all members
        new_w = sum(W[:, c] for c in members)
        # Build new matrices without dropped columns
        cols_to_keep = [c for c in range(R) if c not in drop]
        Wnew = np.zeros((pn, R - len(drop)), dtype=int)
        Unew = U[:, cols_to_keep].copy()
        Vnew = V[:, cols_to_keep].copy()
        for new_c, old_c in enumerate(cols_to_keep):
            Wnew[:, new_c] = W[:, old_c]
        pos = cols_to_keep.index(keep)
        Wnew[:, pos] = new_w
        return Unew, Vnew, Wnew, R - len(drop)

    def split_column(U, V, W, R, c, w1, w2):
        """Split column c into two columns with W values w1 and w2 (w1+w2 = W[:,c])."""
        # Add a new column after c
        Unew = np.zeros((nm, R + 1), dtype=int)
        Vnew = np.zeros((mp, R + 1), dtype=int)
        Wnew = np.zeros((pn, R + 1), dtype=int)
        for i in range(R):
            if i < c:
                Unew[:, i] = U[:, i]
                Vnew[:, i] = V[:, i]
                Wnew[:, i] = W[:, i]
            elif i == c:
                Unew[:, i] = U[:, i]
                Vnew[:, i] = V[:, i]
                Wnew[:, i] = w1
            else:
                Unew[:, i + 1] = U[:, i]
                Vnew[:, i + 1] = V[:, i]
                Wnew[:, i + 1] = W[:, i]
        # Insert new column at c+1
        Unew[:, c + 1] = U[:, c]
        Vnew[:, c + 1] = V[:, c]
        Wnew[:, c + 1] = w2
        return Unew, Vnew, Wnew, R + 1

    def greedy_search(U, V, W, R, rng, deadline):
        """Iteratively apply removals and merges until stuck."""
        improved = True
        while improved and (time.time() < deadline):
            improved = False
            # Try removals
            cols = list(range(R))
            rng.shuffle(cols)
            for c in cols:
                if time.time() > deadline:
                    return U, V, W, R
                Ut, Vt, Wt, Rt = remove_column(U, V, W, R, c)
                if verify(Ut, Vt, Wt, Rt):
                    U, V, W, R = Ut, Vt, Wt, Rt
                    improved = True
                    break

            if not improved:
                # Try group merges
                sup = {}
                groups = {}
                for c in range(R):
                    s = supports(U, V, c)
                    sup[c] = s
                    groups.setdefault(s, []).append(c)
                merged_any = False
                for key, members in groups.items():
                    if time.time() > deadline:
                        return U, V, W, R
                    if len(members) < 2:
                        continue
                    Ut, Vt, Wt, Rt = merge_group(U, V, W, R, members)
                    if verify(Ut, Vt, Wt, Rt):
                        U, V, W, R = Ut, Vt, Wt, Rt
                        merged_any = True
                        break
                if merged_any:
                    improved = True

            if not improved and R > 1:
                # Try splits to escape local minima
                cols = list(range(R))
                rng.shuffle(cols)
                for c in cols:
                    if time.time() > deadline:
                        return U, V, W, R
                    wc = W[:, c]
                    nz = np.nonzero(wc)[0]
                    if len(nz) < 2:
                        continue
                    # Try splitting: assign one nonzero element to w2
                    for idx in nz:
                        if time.time() > deadline:
                            return U, V, W, R
                        w2 = np.zeros(pn, dtype=int)
                        w2[idx] = wc[idx]
                        w1 = wc - w2
                        Ut, Vt, Wt, Rt = split_column(U, V, W, R, c, w1, w2)
                        if verify(Ut, Vt, Wt, Rt):
                            U, V, W, R = Ut, Vt, Wt, Rt
                            improved = True
                            break
                    if improved:
                        break

        return U, V, W, R

    U, V, W = make_schoolbook()
    R = U.shape[1]
    best_U, best_V, best_W, best_R = U.copy(), V.copy(), W.copy(), R

    # Multiple restarts with different random seeds
    num_restarts = 10
    for restart in range(num_restarts):
        if time.time() > deadline:
            break
        seed = 12345 + restart * 7919
        rng = random.Random(seed)
        U, V, W = make_schoolbook()
        R = U.shape[1]
        U, V, W, R = greedy_search(U, V, W, R, rng, deadline)
        if R < best_R:
            best_U, best_V, best_W, best_R = U.copy(), V.copy(), W.copy(), R

    # Final polish: one more pass on the best
    if time.time() < deadline:
        rng = random.Random(99999)
        U, V, W, R = greedy_search(best_U, best_V, best_W, best_R, rng, deadline)
        if R < best_R:
            best_U, best_V, best_W, best_R = U.copy(), V.copy(), W.copy(), R

    # Convert to lists of lists (integer entries)
    def to_list(A):
        return A.tolist()

    return {"U": to_list(best_U[:, :best_R]),
            "V": to_list(best_V[:, :best_R]),
            "W": to_list(best_W[:, :best_R])}
# EVOLVE-BLOCK-END
