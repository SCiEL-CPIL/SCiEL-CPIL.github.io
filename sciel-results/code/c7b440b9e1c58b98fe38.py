"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [3, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 3, 4, 5
    deadline = time.time() + 115.0  # safety margin before 150s kill

    def schoolbook():
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j][col] = 1
                    V[j*p+k][col] = 1
                    W[k*n+i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}

    def strassen_2x2x2():
        """Return U, V, W for 2x2x2 Strassen decomposition."""
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
        """Faster verification using einsum."""
        U = np.array(U, dtype=float)
        V = np.array(V, dtype=float)
        W = np.array(W, dtype=float)
        rank = U.shape[1]
        # Reshape U to (n, m, rank), V to (m, p, rank), W to (p, n, rank)
        U_r = U.reshape(n, m, rank)
        V_r = V.reshape(m, p, rank)
        W_r = W.reshape(p, n, rank)
        # T[i,j,k] = sum_c U_r[i,j,c] * V_r[j,k,c] * W_r[k,i,c]
        T = np.einsum('ijc,jkc,kic->ijk', U_r, V_r, W_r, optimize=True)
        return np.allclose(T, np.ones((n, m, p)), atol=tol, rtol=0)

    # Start with schoolbook baseline
    best_result = schoolbook()
    best_rank = n * m * p

    # Strategy: Use Strassen on 2x2x2 blocks for i in {0,1}, and for i=2 use
    # a rank-4 decomposition of the 1x4x5 tensor (using the 4 independent k-rows
    # shared with the V factor from the i=0,1 blocks).
    #
    # Key insight: For i=2, we need sum_c U[2,j,c]*V[j,k,c]*W[k,2,c] = 1 for all j,k.
    # If we set W[k,2,c] = delta_{k,c} for c in {0,1,2,3} and U[2,j,c] = delta_{j,c},
    # then we need V[j,c,c] = 1 for j=c. But V is shared!
    #
    # Alternative: Use rank 50. For i in {0,1}: 32 terms (as before).
    # For i=2: use 18 terms instead of 20 by sharing 2 terms with the k=4 schoolbook.
    # Actually, let's try a different approach: use the same V and W factors for i=2.
    #
    # Let's try: For i=2, use rank-4 terms where U[2,j,c]=delta_{j,c}, W[k,2,c]=delta_{k,c},
    # and we need V[j,c,c]=1. But V[j,c,c] is already set by the Strassen blocks.
    # This won't work directly.
    #
    # Instead, let's try to reduce from 52 to 50 by finding a better decomposition.
    # Use the following: for the 2x4x5 subproblem (i in {0,1}), instead of 32,
    # try to use 30 terms by a more clever Strassen application.
    #
    # Actually, let's just verify the 52-rank solution works and try small variations.

    U_cols = []
    V_cols = []
    W_cols = []

    # Helper to add a Strassen block for sub-problem at (i_offset, j_offset, k_offset)
    def add_strassen_block(i_off, j_off, k_off):
        U_s, V_s, W_s = strassen_2x2x2()
        for c in range(7):
            U_col = [0.0] * (n * m)
            V_col = [0.0] * (m * p)
            W_col = [0.0] * (p * n)
            for ii in range(2):
                for jj in range(2):
                    U_col[(i_off+ii)*m + (j_off+jj)] = U_s[ii*2+jj][c]
            for jj in range(2):
                for kk in range(2):
                    V_col[(j_off+jj)*p + (k_off+kk)] = V_s[jj*2+kk][c]
            for kk in range(2):
                for ii in range(2):
                    W_col[(k_off+kk)*n + (i_off+ii)] = W_s[kk*2+ii][c]
            U_cols.append(U_col)
            V_cols.append(V_col)
            W_cols.append(W_col)

    # Helper to add schoolbook terms for a sub-block
    def add_schoolbook_block(i_list, j_list, k_list):
        for i in i_list:
            for j in j_list:
                for k in k_list:
                    U_col = [0.0] * (n * m)
                    V_col = [0.0] * (m * p)
                    W_col = [0.0] * (p * n)
                    U_col[i*m + j] = 1.0
                    V_col[j*p + k] = 1.0
                    W_col[k*n + i] = 1.0
                    U_cols.append(U_col)
                    V_cols.append(V_col)
                    W_cols.append(W_col)

    # For i in {0,1}, j in {0,1}: Strassen on k={0,1} and k={2,3}, schoolbook k=4
    add_strassen_block(0, 0, 0)  # covers i={0,1}, j={0,1}, k={0,1}
    add_strassen_block(0, 0, 2)  # covers i={0,1}, j={0,1}, k={2,3}
    add_schoolbook_block([0,1], [0,1], [4])  # k=4 for i={0,1}, j={0,1}

    # For i in {0,1}, j in {2,3}: Strassen on k={0,1} and k={2,3}, schoolbook k=4
    add_strassen_block(0, 2, 0)  # covers i={0,1}, j={2,3}, k={0,1}
    add_strassen_block(0, 2, 2)  # covers i={0,1}, j={2,3}, k={2,3}
    add_schoolbook_block([0,1], [2,3], [4])  # k=4 for i={0,1}, j={2,3}

    # For i=2: schoolbook 1x4x5 = 20 terms
    add_schoolbook_block([2], list(range(4)), list(range(5)))

    # Convert to row-major format
    rank = len(U_cols)
    U_mat = [[U_cols[c][r] for c in range(rank)] for r in range(n*m)]
    V_mat = [[V_cols[c][r] for c in range(rank)] for r in range(m*p)]
    W_mat = [[W_cols[c][r] for c in range(rank)] for r in range(p*n)]

    if verify_fast(U_mat, V_mat, W_mat, n, m, p):
        if rank < best_rank:
            best_result = {"U": U_mat, "V": V_mat, "W": W_mat}
            best_rank = rank

    # Now try to find a rank-50 solution by solving linear system
    # For rank r, we need to find U, V, W such that the tensor equation holds.
    # This is a system of 60 nonlinear equations in 2350 variables (for r=50).
    # We use a structured ansatz: keep the Strassen structure but perturb.
    #
    # Alternative: try rank 50 with a different block structure.
    # Use Strassen for the 4x4x4 core (j,k in {0,1,2,3}) for all i,
    # plus schoolbook for k=4.
    # 4x4x4 Strassen: split into 4 2x2x2 blocks, each rank 7, total 28.
    # But this gives 28 per i-value, times 3 i-values = 84, plus 3*4=12 for k=4 = 96. Worse.
    #
    # Let's try: for i in {0,1,2}, use the same V and W structure for the k=4 part.
    # The k=4 part: for each i, we need sum_c U[i,j,c]*V[j,4,c]*W[4,i,c] = 1 for all j.
    # If we set W[4,i,c] = 1 for c in {0,1,2,3} and U[i,j,c] = delta_{j,c},
    # then we need V[j,4,c] = delta_{j,c}. This uses 4 terms per i, total 12.
    # But V[j,4,c] is shared across i! So we need V[j,4,c] to work for all i.
    # If W[4,i,c] = delta_{i,c} for c in {0,1,2}, then for i=0: need sum_c U[0,j,c]*V[j,4,c]*delta_{0,c} = V[j,4,0]*U[0,j,0] = 1.
    # This is getting complex. Let's just stick with 52 and verify.

    return best_result
# EVOLVE-BLOCK-END
