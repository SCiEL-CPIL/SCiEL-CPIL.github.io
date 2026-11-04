"""Rewritten program for matrix multiplication 3x4x5 tensor decomposition.
Uses Strassen block recursion with a more aggressive search for rank reduction.
"""
import time
import numpy as np

DIMENSIONS = [3, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 3, 4, 5
    deadline = time.time() + 115.0

    def strassen_2x2x2():
        """Return U, V, W for 2x2x2 Strassen decomposition (rank 7)."""
        U = [
            [1, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 1, 0],
            [0, 0, 1, 0, 0, 1, 0],
            [1, 1, 0, 0, 0, 0, 1],
        ]
        V = [
            [1, 0, 0, 0, 0, 1, 1],
            [0, 1, -1, 1, 0, 0, 1],
            [1, 0, 1, 0, 0, 1, 0],
            [1, 1, 0, -1, 1, 0, 0],
        ]
        W = [
            [1, 0, 1, 0, 0, 0, 0],
            [0, 1, 0, 0, 1, 0, 1],
            [0, 0, 0, 1, 0, 1, 0],
            [1, 0, 0, 1, -1, 0, 0],
        ]
        return U, V, W

    def verify_fast(U, V, W, n, m, p, tol=1e-9):
        U = np.array(U, dtype=float)
        V = np.array(V, dtype=float)
        W = np.array(W, dtype=float)
        rank = U.shape[1]
        U_r = U.reshape(n, m, rank)
        V_r = V.reshape(m, p, rank)
        W_r = W.reshape(p, n, rank)
        T = np.einsum('ijc,jkc,kic->ijk', U_r, V_r, W_r, optimize=True)
        return np.allclose(T, np.ones((n, m, p)), atol=tol, rtol=0)

    # Strategy: Build the 3x4x5 tensor using Strassen on 2x2x2 blocks.
    # We partition the 4 middle dimension into two 2-blocks: {0,1} and {2,3},
    # and handle k=4 separately.
    # For each pair (i in 0..2, j in 0..3), we need to handle k in 0..4.
    # 
    # A better approach: Think of this as multiplying a 3x4 matrix by a 4x5 matrix.
    # We can use the following block structure:
    # - Split rows of first matrix into {0,1} and {2}
    # - Split cols of first matrix / rows of second into {0,1}, {2,3}, and {4} is separate
    #
    # Let's use a more systematic approach:
    # For i in {0,1}:
    #   For j in {0,1}: handle (i,j) with k in {0,1} using Strassen (7 terms)
    #   For j in {2,3}: handle (i,j) with k in {2,3} using Strassen (7 terms)
    #   For all j: handle k=4 using schoolbook (4 terms per i)
    # For i=2:
    #   Schoolbook for all j,k (20 terms)
    #
    # Total: 2 * 2 * 7 + 2 * 4 + 20 = 28 + 8 + 20 = 56 terms
    #
    # Alternative: For i in {0,1}, treat j in {0,1,2,3} as a 2x4 block.
    # Split j into {0,1} and {2,3}, and k into {0,1}, {2,3}, and {4}.
    # For each (i, j_block, k_block) combination:
    #   - (i in {0,1}, j in {0,1}, k in {0,1}): 2x2x2 Strassen = 7 terms
    #   - (i in {0,1}, j in {2,3}, k in {0,1}): 2x2x2 Strassen = 7 terms
    #   - (i in {0,1}, j in {0,1}, k in {2,3}): 2x2x2 Strassen = 7 terms
    #   - (i in {0,1}, j in {2,3}, k in {2,3}): 2x2x2 Strassen = 7 terms
    #   - k=4: schoolbook for i in {0,1}, j in {0,1,2,3} = 8 terms
    #   - i=2: schoolbook for all j,k = 20 terms
    # Total: 4*7 + 8 + 20 = 28 + 8 + 20 = 56 terms
    #
    # Can we do better? Let's try a different partition.
    # What if we group i in {0,1} and handle the full 2x4x5 subproblem?
    # 2x4x5 can be seen as two 2x2x5 problems (j in {0,1} and j in {2,3}).
    # Each 2x2x5: split k into {0,1}, {2,3}, {4}
    #   - k in {0,1}: Strassen 2x2x2 = 7
    #   - k in {2,3}: Strassen 2x2x2 = 7
    #   - k in {4}: schoolbook 2x2x1 = 4
    #   Total per 2x2x5: 18
    # Two such subproblems: 36
    # Plus i=2: 1x4x5 schoolbook = 20
    # Total: 36 + 20 = 56
    #
    # Another approach: Use the fact that 3x4x5 can be embedded in a larger
    # structured way. Let's try to find a decomposition with fewer terms.
    #
    # Known lower bound for n x m x p matrix multiplication is n*m*p (schoolbook).
    # Strassen reduces 2x2x2 from 8 to 7. For 3x4x5, we can potentially do better
    # than 60 (schoolbook) by using Strassen on sub-blocks.
    #
    # Let's try a more creative approach: use a 3x3x3 Strassen-like decomposition
    # for the main block and handle the rest.
    #
    # Actually, let's just implement the 56-term solution first and verify it works.
    # Then we can try to optimize further.

    U_cols = []
    V_cols = []
    W_cols = []

    U_s, V_s, W_s = strassen_2x2x2()

    # For i in {0,1}, j in {0,1}, k in {0,1}: Strassen
    for c in range(7):
        U_col = [0.0] * (n * m)
        V_col = [0.0] * (m * p)
        W_col = [0.0] * (p * n)
        for i in range(2):
            for j in range(2):
                U_col[i*m + j] = U_s[i*2+j][c]
        for j in range(2):
            for k in range(2):
                V_col[j*p + k] = V_s[j*2+k][c]
        for k in range(2):
            for i in range(2):
                W_col[k*n + i] = W_s[k*2+i][c]
        U_cols.append(U_col)
        V_cols.append(V_col)
        W_cols.append(W_col)

    # For i in {0,1}, j in {2,3}, k in {0,1}: Strassen
    for c in range(7):
        U_col = [0.0] * (n * m)
        V_col = [0.0] * (m * p)
        W_col = [0.0] * (p * n)
        for i in range(2):
            for j in range(2):
                U_col[i*m + (2+j)] = U_s[i*2+j][c]
        for j in range(2):
            for k in range(2):
                V_col[(2+j)*p + k] = V_s[j*2+k][c]
        for k in range(2):
            for i in range(2):
                W_col[k*n + i] = W_s[k*2+i][c]
        U_cols.append(U_col)
        V_cols.append(V_col)
        W_cols.append(W_col)

    # For i in {0,1}, j in {0,1}, k in {2,3}: Strassen
    for c in range(7):
        U_col = [0.0] * (n * m)
        V_col = [0.0] * (m * p)
        W_col = [0.0] * (p * n)
        for i in range(2):
            for j in range(2):
                U_col[i*m + j] = U_s[i*2+j][c]
        for j in range(2):
            for k in range(2):
                V_col[j*p + (2+k)] = V_s[j*2+k][c]
        for k in range(2):
            for i in range(2):
                W_col[(2+k)*n + i] = W_s[k*2+i][c]
        U_cols.append(U_col)
        V_cols.append(V_col)
        W_cols.append(W_col)

    # For i in {0,1}, j in {2,3}, k in {2,3}: Strassen
    for c in range(7):
        U_col = [0.0] * (n * m)
        V_col = [0.0] * (m * p)
        W_col = [0.0] * (p * n)
        for i in range(2):
            for j in range(2):
                U_col[i*m + (2+j)] = U_s[i*2+j][c]
        for j in range(2):
            for k in range(2):
                V_col[(2+j)*p + (2+k)] = V_s[j*2+k][c]
        for k in range(2):
            for i in range(2):
                W_col[(2+k)*n + i] = W_s[k*2+i][c]
        U_cols.append(U_col)
        V_cols.append(V_col)
        W_cols.append(W_col)

    # For i in {0,1}, all j, k=4: schoolbook
    for i in range(2):
        for j in range(m):
            U_col = [0.0] * (n * m)
            V_col = [0.0] * (m * p)
            W_col = [0.0] * (p * n)
            U_col[i*m + j] = 1.0
            V_col[j*p + 4] = 1.0
            W_col[4*n + i] = 1.0
            U_cols.append(U_col)
            V_cols.append(V_col)
            W_cols.append(W_col)

    # For i=2, all j, k: schoolbook
    for j in range(m):
        for k in range(p):
            U_col = [0.0] * (n * m)
            V_col = [0.0] * (m * p)
            W_col = [0.0] * (p * n)
            U_col[2*m + j] = 1.0
            V_col[j*p + k] = 1.0
            W_col[k*n + 2] = 1.0
            U_cols.append(U_col)
            V_cols.append(V_col)
            W_cols.append(W_col)

    rank = len(U_cols)
    U_mat = [[U_cols[c][r] for c in range(rank)] for r in range(n*m)]
    V_mat = [[V_cols[c][r] for c in range(rank)] for r in range(m*p)]
    W_mat = [[W_cols[c][r] for c in range(rank)] for r in range(p*n)]

    best_result = {"U": U_mat, "V": V_mat, "W": W_mat}
    best_rank = rank

    if verify_fast(U_mat, V_mat, W_mat, n, m, p):
        pass
    else:
        # Fallback to schoolbook if verification fails
        schoolbook_rank = n * m * p
        U_sb = [[0] * schoolbook_rank for _ in range(n * m)]
        V_sb = [[0] * schoolbook_rank for _ in range(m * p)]
        W_sb = [[0] * schoolbook_rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_sb[i*m+j][col] = 1
                    V_sb[j*p+k][col] = 1
                    W_sb[k*n+i][col] = 1
                    col += 1
        best_result = {"U": U_sb, "V": V_sb, "W": W_sb}
        best_rank = schoolbook_rank

    return best_result
# EVOLVE-BLOCK-END
