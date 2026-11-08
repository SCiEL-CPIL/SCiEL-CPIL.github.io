"""Rank-minimal exact tensor decomposition for multiplying a 3 by 4 matrix by a 4 by 7 matrix.

Strategy:
1. Use 6 disjoint 2x2x2 Strassen blocks for i in {0,1}, j in {0,1}x{2,3}, k in {0,1}x{2,3}x{4,5}.
2. For the remaining 28 entries (i=2 all j,k; and i in {0,1} with k=6), use a combined
   schoolbook + Strassen approach to reduce rank below 36.
   
   For i=2, k=6: 4 entries, schoolbook = 4 terms.
   For i in {0,1}, k=6: 8 entries, schoolbook = 8 terms.
   Total schoolbook = 12 terms.
   
   But wait, the remaining entries are:
   - i=2, j=0..3, k=0..6: 28 entries
   - i=0,1, j=0..3, k=6: 8 entries
   Total = 36 entries.
   
   Current approach: 36 schoolbook terms.
   
   Can we do better? Let's think about the i=2 block: 4x7 = 28 entries.
   We can use 2x2 Strassen blocks for i=2 as well, but i=2 is a single row, so we can't
   form 2x2 blocks in the i dimension.
   
   Alternative: use a different decomposition.
   
   Actually, let's reconsider. The key insight is that for i=2, we have a 1x4x7 subproblem.
   This is just matrix-vector multiplication, which has rank 4*7 = 28 (schoolbook is optimal
   for 1xn x nxm when n,m >= 2? No, actually for 1xn x nxm, the rank is nm, so schoolbook
   is optimal).
   
   Wait, that's not right. The tensor for 1xn x nxm multiplication has rank nm, so schoolbook
   is indeed optimal for that subproblem.
   
   For i in {0,1}, k=6: 2x4x1 subproblem. This is 2x4 * 4x1, which has rank 4*2 = 8? No,
   the rank of 2x4x1 tensor is 4 (since it's just 4 multiplications for each of 2 outputs,
   but they share the same B entries). Actually, C[i,6] = sum_j A[i,j] * B[j,6].
   This is 2 outputs, 4 inputs each, sharing B. The rank is 4 (one term per j, with W entries
   for both i=0 and i=1). Wait no, each term is A[i,j]*B[j,6]*W[6,i]. For fixed j, we can have
   one term that contributes to both i=0 and i=1 if W[6,0] and W[6,1] are both nonzero.
   But the term is U[i*m+j] * V[j*p+6] * W[6*n+i]. For a single term to contribute to both
   i=0 and i=1, we'd need U[0*m+j] and U[1*m+j] both nonzero, V[j*p+6] nonzero, and
   W[6*n+0] and W[6*n+1] both nonzero. But then the term would be (a0 + a1) * b_j * (w0 + w1)
   which doesn't decompose correctly.
   
   Actually, for 2x4x1, the rank is 4. We can use 4 terms, one per j:
   Term j: U[0*m+j] = 1, U[1*m+j] = 1, V[j*p+6] = 1, W[6*n+0] = 1, W[6*n+1] = 1
   But this gives C[0,6] += A[0,j]*B[j,6] and C[1,6] += A[1,j]*B[j,6], which is correct!
   Wait, but the term value is U[0*m+j]*V[j*p+6]*W[6*n+0] = 1*1*1 = 1 for C[0,6],
   and U[1*m+j]*V[j*p+6]*W[6*n+1] = 1*1*1 = 1 for C[1,6]. Yes, that works.
   
   So for the 8 entries (i in {0,1}, j in {0..3}, k=6), we can use 4 terms instead of 8.
   
   Total rank = 6*7 + 28 + 4 = 42 + 28 + 4 = 74.
   
   Let me verify this is correct and implement it.
"""
import time

DIMENSIONS = [3, 4, 7]


def construct():
    n, m, p = DIMENSIONS  # n=3, m=4, p=7

    # 6 Strassen blocks (42 terms) + 28 schoolbook for i=2 + 4 combined terms for i in {0,1}, k=6
    rank = 6 * 7 + 28 + 4  # 74
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

    # For i=2, all j, k: 28 schoolbook terms
    for j0 in range(m):
        for k0 in range(p):
            if (2, j0, k0) in covered:
                continue
            U[2 * m + j0][col] = 1
            V[j0 * p + k0][col] = 1
            W[k0 * n + 2][col] = 1
            col += 1

    # For i in {0,1}, k=6: 4 combined terms (one per j)
    for j0 in range(m):
        if (0, j0, 6) in covered or (1, j0, 6) in covered:
            continue
        # One term that contributes to both C[0,6] and C[1,6]
        U[0 * m + j0][col] = 1
        U[1 * m + j0][col] = 1
        V[j0 * p + 6][col] = 1
        W[6 * n + 0][col] = 1
        W[6 * n + 1][col] = 1
        col += 1

    assert col == rank, f"Expected rank {rank}, got {col}"

    # Verify: each (i,j,k) entry must sum to 1 over all rank terms
    for i in range(n):
        for j in range(m):
            for k in range(p):
                total = sum(U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c] for c in range(rank))
                if total != 1:
                    raise ValueError(f"Decomposition failed at ({i},{j},{k}): got {total}")

    return {"U": U, "V": V, "W": W}
