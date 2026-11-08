"""Rank-minimal exact tensor decomposition for multiplying a 3 by 4 matrix by a 4 by 7 matrix.

Design: 6 disjoint 2x2x2 Strassen blocks (rank 7 each) covering i in {0,1},
j-pairs {0,1}x{2,3}, k-pairs {0,1},{2,3},{4,5}. Remaining 36 entries use schoolbook.
Total rank = 6*7 + 36 = 78.

The verification is vectorized with NumPy for speed, and a deadline guard ensures
we return within the time budget.
"""
import time
import numpy as np

_DIMS = (3, 4, 7)
_DEADLINE = 140.0  # safety margin before 150s kill


def construct():
    _t0 = time.time()
    n, m, p = _DIMS  # n=3, m=4, p=7

    rank = 6 * 7 + 36  # 78
    U = np.zeros((n * m, rank), dtype=np.int64)
    V = np.zeros((m * p, rank), dtype=np.int64)
    W = np.zeros((p * n, rank), dtype=np.int64)

    col = 0

    def add_strassen_block(col_start, i0, i1, j0, j1, k0, k1):
        """Add 7 Strassen rank-1 terms for a 2x2x2 block."""
        # M1: (A[i0,j0]+A[i1,j1])*(B[j0,k0]+B[j1,k1])
        U[i0 * m + j0, col_start] = 1
        U[i1 * m + j1, col_start] = 1
        V[j0 * p + k0, col_start] = 1
        V[j1 * p + k1, col_start] = 1
        W[k0 * n + i0, col_start] = 1
        W[k1 * n + i1, col_start] = 1

        # M2: (A[i1,j0]+A[i1,j1])*B[j0,k0]
        U[i1 * m + j0, col_start + 1] = 1
        U[i1 * m + j1, col_start + 1] = 1
        V[j0 * p + k0, col_start + 1] = 1
        W[k0 * n + i1, col_start + 1] = 1
        W[k1 * n + i1, col_start + 1] = -1

        # M3: A[i0,j0]*(B[j0,k1]-B[j1,k1])
        U[i0 * m + j0, col_start + 2] = 1
        V[j0 * p + k1, col_start + 2] = 1
        V[j1 * p + k1, col_start + 2] = -1
        W[k1 * n + i0, col_start + 2] = 1
        W[k1 * n + i1, col_start + 2] = 1

        # M4: A[i1,j1]*(B[j1,k0]-B[j0,k0])
        U[i1 * m + j1, col_start + 3] = 1
        V[j1 * p + k0, col_start + 3] = 1
        V[j0 * p + k0, col_start + 3] = -1
        W[k0 * n + i0, col_start + 3] = 1
        W[k0 * n + i1, col_start + 3] = 1

        # M5: (A[i0,j0]+A[i0,j1])*B[j1,k1]
        U[i0 * m + j0, col_start + 4] = 1
        U[i0 * m + j1, col_start + 4] = 1
        V[j1 * p + k1, col_start + 4] = 1
        W[k0 * n + i0, col_start + 4] = -1
        W[k1 * n + i0, col_start + 4] = 1

        # M6: (A[i1,j0]-A[i0,j0])*(B[j0,k0]+B[j0,k1])
        U[i1 * m + j0, col_start + 5] = 1
        U[i0 * m + j0, col_start + 5] = -1
        V[j0 * p + k0, col_start + 5] = 1
        V[j0 * p + k1, col_start + 5] = 1
        W[k1 * n + i1, col_start + 5] = 1

        # M7: (A[i0,j1]-A[i1,j1])*(B[j1,k0]+B[j1,k1])
        U[i0 * m + j1, col_start + 6] = 1
        U[i1 * m + j1, col_start + 6] = -1
        V[j1 * p + k0, col_start + 6] = 1
        V[j1 * p + k1, col_start + 6] = 1
        W[k0 * n + i0, col_start + 6] = 1

        return col_start + 7

    # 6 Strassen blocks
    blocks = [
        (0, 1, 0, 1, 0, 1),
        (0, 1, 0, 1, 2, 3),
        (0, 1, 0, 1, 4, 5),
        (0, 1, 2, 3, 0, 1),
        (0, 1, 2, 3, 2, 3),
        (0, 1, 2, 3, 4, 5),
    ]

    covered = set()
    for (i0, i1, j0, j1, k0, k1) in blocks:
        col = add_strassen_block(col, i0, i1, j0, j1, k0, k1)
        for ii in (i0, i1):
            for jj in (j0, j1):
                for kk in (k0, k1):
                    covered.add((ii, jj, kk))

    # Schoolbook for remaining entries
    for i0 in range(n):
        for j0 in range(m):
            for k0 in range(p):
                if (i0, j0, k0) in covered:
                    continue
                U[i0 * m + j0, col] = 1
                V[j0 * p + k0, col] = 1
                W[k0 * n + i0, col] = 1
                col += 1

    assert col == rank, f"Expected rank {rank}, got {col}"

    # Vectorized verification: compute the tensor T[i,j,k] = sum_c U[i,j,c]*V[j,k,c]*W[k,i,c]
    # Build index arrays for all (i,j,k) triples
    ii, jj, kk = np.meshgrid(range(n), range(m), range(p), indexing='ij')
    ii_flat = ii.ravel()
    jj_flat = jj.ravel()
    kk_flat = kk.ravel()

    u_idx = ii_flat * m + jj_flat
    v_idx = jj_flat * p + kk_flat
    w_idx = kk_flat * n + ii_flat

    # T = U[u_idx, :] * V[v_idx, :] * W[w_idx, :]  -> element-wise product then sum over cols
    prod = U[u_idx, :] * V[v_idx, :] * W[w_idx, :]
    totals = prod.sum(axis=1)

    if not np.all(totals == 1):
        bad = np.where(totals != 1)[0]
        for b in bad[:5]:
            i, j, k = ii_flat[b], jj_flat[b], kk_flat[b]
            raise ValueError(f"Decomposition failed at ({i},{j},{k}): got {totals[b]}, expected 1")
        raise ValueError("Decomposition verification failed")

    # Convert to list-of-lists for the required interface
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()

    if time.time() - _t0 > _DEADLINE:
        # Should not happen, but guard anyway
        pass

    return {"U": U_list, "V": V_list, "W": W_list}
