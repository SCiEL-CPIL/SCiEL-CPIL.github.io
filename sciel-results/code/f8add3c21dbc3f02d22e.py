"""Deterministic multi-Strassen rank-78 decomposition for 3x4*4x7 matrix multiplication.

Uses 6 disjoint 2x2x2 Strassen blocks with i in {0,1}, partitioning j into {0,1}x{2,3}
and k into {0,1}x{2,3}x{4,5}. Remaining 36 entries use schoolbook.
Total rank = 6*7 + 36 = 78.

Optimization: use NumPy for verification to reduce wall-clock time, ensuring we stay
well within the 150s limit while maintaining correctness.
"""
import time
import numpy as np

DIMENSIONS = [3, 4, 7]
_DEADLINE = 140.0  # safety margin before 150s kill


def construct():
    _t0 = time.time()
    n, m, p = DIMENSIONS  # n=3, m=4, p=7

    # 6 Strassen blocks (i in {0,1}), 36 schoolbook terms
    rank = 6 * 7 + 36  # 78
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    col = 0

    def add_strassen_block(col_start, i0, i1, j0, j1, k0, k1):
        """Add 7 Strassen rank-1 terms for a 2x2x2 block with explicit indices."""
        # M1: (A[i0,j0]+A[i1,j1])*(B[j0,k0]+B[j1,k1]) -> C[i0,k0]+, C[i1,k1]+=
        U[i0 * m + j0][col_start] = 1
        U[i1 * m + j1][col_start] = 1
        V[j0 * p + k0][col_start] = 1
        V[j1 * p + k1][col_start] = 1
        W[k0 * n + i0][col_start] = 1
        W[k1 * n + i1][col_start] = 1

        # M2: (A[i1,j0]+A[i1,j1])*B[j0,k0] -> C[i1,k0]+, C[i1,k1]-
        U[i1 * m + j0][col_start + 1] = 1
        U[i1 * m + j1][col_start + 1] = 1
        V[j0 * p + k0][col_start + 1] = 1
        W[k0 * n + i1][col_start + 1] = 1
        W[k1 * n + i1][col_start + 1] = -1

        # M3: A[i0,j0]*(B[j0,k1]-B[j1,k1]) -> C[i0,k1]+, C[i1,k1]+=
        U[i0 * m + j0][col_start + 2] = 1
        V[j0 * p + k1][col_start + 2] = 1
        V[j1 * p + k1][col_start + 2] = -1
        W[k1 * n + i0][col_start + 2] = 1
        W[k1 * n + i1][col_start + 2] = 1

        # M4: A[i1,j1]*(B[j1,k0]-B[j0,k0]) -> C[i0,k0]+, C[i1,k0]+=
        U[i1 * m + j1][col_start + 3] = 1
        V[j1 * p + k0][col_start + 3] = 1
        V[j0 * p + k0][col_start + 3] = -1
        W[k0 * n + i0][col_start + 3] = 1
        W[k0 * n + i1][col_start + 3] = 1

        # M5: (A[i0,j0]+A[i0,j1])*B[j1,k1] -> C[i0,k0]-, C[i0,k1]+=
        U[i0 * m + j0][col_start + 4] = 1
        U[i0 * m + j1][col_start + 4] = 1
        V[j1 * p + k1][col_start + 4] = 1
        W[k0 * n + i0][col_start + 4] = -1
        W[k1 * n + i0][col_start + 4] = 1

        # M6: (A[i1,j0]-A[i0,j0])*(B[j0,k0]+B[j0,k1]) -> C[i1,k1]+=
        U[i1 * m + j0][col_start + 5] = 1
        U[i0 * m + j0][col_start + 5] = -1
        V[j0 * p + k0][col_start + 5] = 1
        V[j0 * p + k1][col_start + 5] = 1
        W[k1 * n + i1][col_start + 5] = 1

        # M7: (A[i0,j1]-A[i1,j1])*(B[j1,k0]+B[j1,k1]) -> C[i0,k0]+=
        U[i0 * m + j1][col_start + 6] = 1
        U[i1 * m + j1][col_start + 6] = -1
        V[j1 * p + k0][col_start + 6] = 1
        V[j1 * p + k1][col_start + 6] = 1
        W[k0 * n + i0][col_start + 6] = 1

        return col_start + 7

    # 6 Strassen blocks: i in {0,1}, j-pairs {0,1} and {2,3}, k-pairs {0,1},{2,3},{4,5}
    blocks = [
        (0, 1, 0, 1, 0, 1),
        (0, 1, 0, 1, 2, 3),
        (0, 1, 0, 1, 4, 5),
        (0, 1, 2, 3, 0, 1),
        (0, 1, 2, 3, 2, 3),
        (0, 1, 2, 3, 4, 5),
    ]

    # Track which (i,j,k) tuples are covered by Strassen blocks
    covered = set()
    for (i0, i1, j0, j1, k0, k1) in blocks:
        col = add_strassen_block(col, i0, i1, j0, j1, k0, k1)
        for ii in (i0, i1):
            for jj in (j0, j1):
                for kk in (k0, k1):
                    covered.add((ii, jj, kk))

    # Schoolbook for all remaining (i,j,k)
    for i0 in range(n):
        for j0 in range(m):
            for k0 in range(p):
                if (i0, j0, k0) in covered:
                    continue
                U[i0 * m + j0][col] = 1
                V[j0 * p + k0][col] = 1
                W[k0 * n + i0][col] = 1
                col += 1

    assert col == rank, f"Expected rank {rank}, got {col}"

    # Verify the decomposition using NumPy for speed
    U_np = np.array(U, dtype=np.int8)
    V_np = np.array(V, dtype=np.int8)
    W_np = np.array(W, dtype=np.int8)

    # Fully vectorized verification:
    # For each (i,j,k), check sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] == 1
    # We can construct this as a batched operation.
    
    # Create index arrays
    i_idx = np.repeat(np.arange(n), m * p)
    j_idx = np.repeat(np.arange(m), p)
    j_idx = np.tile(j_idx, n)
    k_idx = np.tile(np.arange(p), n * m)
    
    u_rows = U_np[i_idx * m + j_idx]  # shape (n*m*p, rank)
    v_rows = V_np[j_idx * p + k_idx]  # shape (n*m*p, rank)
    w_rows = W_np[k_idx * n + i_idx]  # shape (n*m*p, rank)
    
    # Compute all (i,j,k) checks at once
    results = np.sum(u_rows * v_rows * w_rows, axis=1)
    
    if not np.all(results == 1):
        bad = np.where(results != 1)[0]
        for idx in bad[:5]:  # Show first few failures
            ii = idx // (m * p)
            jj = (idx % (m * p)) // p
            kk = idx % p
            raise ValueError(f"Decomposition failed at ({ii},{jj},{kk}): got {results[idx]}, expected 1")

    return {"U": U, "V": V, "W": W}
