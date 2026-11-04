"""Rewritten program: structured sparse linear algebra / binary IP for exact tensor decomposition."""
DIMENSIONS = [3, 3, 8]
import random
random.seed(1)


# EVOLVE-BLOCK-START
def construct():
    n, m, p = 3, 3, 8
    nm, mp, pn = n * m, m * p, p * n  # 9, 24, 24

    import numpy as np
    from itertools import product

    # Build the target tensor: T[a,b,c] = 1 if a=i*m+j, b=j*p+k, c=k*n+i for some i,j,k
    # Otherwise T[a,b,c] = 0
    target = np.zeros((nm, mp, pn), dtype=int)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                target[a, b, c] = 1

    # Generate candidate rank-1 patterns.
    # Each pattern is a triple (u, v, w) of sparse vectors with entries in {-1, 0, 1}.
    # We'll generate patterns that are "structured": each has small support.
    # For each candidate, compute the tensor contribution: contrib[a,b,c] = u[a]*v[b]*w[c]
    # We want to select a subset such that sum of contributions equals target exactly.

    # Strategy: Generate a dictionary of candidate patterns, then use a greedy ILP approach.
    # For efficiency, we'll focus on patterns with support size <= 2 in each vector.

    candidates = []

    # Add schoolbook patterns (each has support 1 in each vector)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u = [0] * nm
                v = [0] * mp
                w = [0] * pn
                u[i * m + j] = 1
                v[j * p + k] = 1
                w[k * n + i] = 1
                candidates.append((u, v, w))

    # Add Strassen-like patterns for 3x3x3 sub-blocks.
    # For a 3x3x3 block (i,j,k ranges over 0..2), we can use the 7-term Strassen algorithm.
    # The Strassen algorithm for C = A*B where A,B are 3x3:
    # M1 = (A00 + A22) * (B00 + B22)
    # M2 = (A20 + A22) * B20
    # M3 = A00 * (B01 - B21)
    # M4 = A22 * (B21 - B01)
    # M5 = (A00 + A10) * B00
    # M6 = (A12 + A22) * B22
    # M7 = (A00 + A01) * (B11 - B21)
    # M8 = (A10 + A20) * (B00 - B10)
    # M9 = (A11 + A21) * (B10 - B20)
    # M10 = (A11 + A12) * (B22 - B02)
    # M11 = (A01 + A11) * (B00 - B01)
    # M12 = (A02 + A12) * (B11 - B12)
    # Actually, standard Strassen is for 2x2. For 3x3, it's more complex.
    # Let me use a different approach: generate random sparse patterns and verify.

    # Add random sparse patterns with support size 2
    np.random.seed(42)
    for trial in range(2000):
        u = np.zeros(nm, dtype=int)
        v = np.zeros(mp, dtype=int)
        w = np.zeros(pn, dtype=int)
        # Pick 2 random positions for each
        u_idx = np.random.choice(nm, size=2, replace=False)
        v_idx = np.random.choice(mp, size=2, replace=False)
        w_idx = np.random.choice(pn, size=2, replace=False)
        for idx in u_idx:
            u[idx] = np.random.choice([-1, 1])
        for idx in v_idx:
            v[idx] = np.random.choice([-1, 1])
        for idx in w_idx:
            w[idx] = np.random.choice([-1, 1])
        candidates.append((u.tolist(), v.tolist(), w.tolist()))

    # Add random sparse patterns with support size 3
    for trial in range(1000):
        u = np.zeros(nm, dtype=int)
        v = np.zeros(mp, dtype=int)
        w = np.zeros(pn, dtype=int)
        u_idx = np.random.choice(nm, size=3, replace=False)
        v_idx = np.random.choice(mp, size=3, replace=False)
        w_idx = np.random.choice(pn, size=3, replace=False)
        for idx in u_idx:
            u[idx] = np.random.choice([-1, 1])
        for idx in v_idx:
            v[idx] = np.random.choice([-1, 1])
        for idx in w_idx:
            w[idx] = np.random.choice([-1, 1])
        candidates.append((u.tolist(), v.tolist(), w.tolist()))

    # For each candidate, compute its tensor contribution as a flattened vector
    # and check if it's useful (non-zero contribution)
    useful_candidates = []
    for (u, v, w) in candidates:
        u_arr = np.array(u)
        v_arr = np.array(v)
        w_arr = np.array(w)
        contrib = np.outer(np.outer(u_arr, v_arr), w_arr)  # shape (nm, mp, pn)
        if np.any(contrib != 0):
            useful_candidates.append((u, v, w, contrib.flatten().astype(int)))

    # Now use a greedy approach to select a minimal subset.
    # We need: sum of selected contributions = target (flattened)
    target_flat = target.flatten().astype(int)

    # Greedy: repeatedly pick the candidate that best reduces the residual.
    # "Best" = maximizes inner product with residual (to match target) while minimizing
    # the number of new non-zero positions it introduces where target is 0.

    residual = target_flat.copy()
    selected = []
    used_indices = set()

    max_iter = 100
    for it in range(max_iter):
        if np.all(residual == 0):
            break
        best_idx = -1
        best_score = -1e18
        for idx, (u, v, w, contrib) in enumerate(useful_candidates):
            if idx in used_indices:
                continue
            # Score: how well this candidate helps match the residual
            # We want to reduce ||residual - contrib||^2 or similar
            # Simpler: maximize dot product with residual where residual is non-zero,
            # and minimize dot product with -residual where residual is non-zero
            # Actually, let's just try adding it and see if it reduces the L1 error
            new_residual = residual - contrib
            # Penalty: number of positions where new_residual is non-zero but target is zero
            # and the magnitude of those
            error_reduction = np.sum(np.abs(residual)) - np.sum(np.abs(new_residual))
            # Also penalize introducing new errors
            new_errors = np.sum((new_residual != 0) & (target_flat == 0))
            score = error_reduction - 0.1 * new_errors
            if score > best_score:
                best_score = score
                best_idx = idx

        if best_idx == -1 or best_score <= 0:
            # No improvement, break
            break

        u, v, w, contrib = useful_candidates[best_idx]
        selected.append((u, v, w))
        residual = residual - contrib
        used_indices.add(best_idx)

    # Check if we achieved exact decomposition
    if not np.all(residual == 0):
        # Fallback: use schoolbook
        rank = n * m * p
        U = [[0] * rank for _ in range(nm)]
        V = [[0] * rank for _ in range(mp)]
        W = [[0] * rank for _ in range(pn)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}

    # Build the result matrices
    R = len(selected)
    U = [[0] * R for _ in range(nm)]
    V = [[0] * R for _ in range(mp)]
    W = [[0] * R for _ in range(pn)]
    for r, (u, v, w) in enumerate(selected):
        for a in range(nm):
            U[a][r] = u[a]
        for b in range(mp):
            V[b][r] = v[b]
        for c in range(pn):
            W[c][r] = w[c]

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
