"""Strassen-based decomposition for 2x5x6 matrix multiplication."""

import time
import sys

DIMENSIONS = [2, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = 2, 5, 6
    deadline = time.time() + 100  # Safety margin: return by 100s, limit is 150s
    
    # We decompose into 6 independent 2x2x2 blocks plus 3 rank-1 blocks.
    #
    # The 2x5 matrix A has rows i in {0,1} and columns j in {0,1,2,3,4}.
    # The 5x6 matrix B has rows j in {0,1,2,3,4} and columns k in {0,1,2,3,4,5}.
    # The 2x6 matrix C has rows i in {0,1} and columns k in {0,1,2,3,4,5}.
    #
    # We pair the columns of A: (0,1), (2,3), (4,?) -- wait, we have 5 columns.
    # Let's think about how to decompose 2x5x6 into 2x2x2 blocks.
    #
    # A is 2x5. We can split columns of A into groups: {0,1}, {2,3}, {4}.
    # B is 5x6. The rows of B correspond to columns of A.
    # So B's rows are j=0,1,2,3,4.
    # We need to split B's rows into groups matching A's column groups: {0,1}, {2,3}, {4}.
    # And split B's columns (k=0,1,2,3,4,5) into pairs: {0,1}, {2,3}, {4,5}.
    #
    # For each pair of A-columns (ja, jb) and each pair of B-columns (ka, kb):
    #   We have a 2x2x2 sub-problem:
    #     A_sub = A[:, [ja, jb]]  (2x2)
    #     B_sub = B[[ja, jb], [ka, kb]]  (2x2)
    #     C_sub = C[:, [ka, kb]]  (2x2)
    #
    # This gives us 3 * 3 = 9 potential 2x2x2 blocks.
    # But wait: for A column group {4}, it's a single column, not a pair.
    # So that's not a 2x2x2 block, it's a 2x1x2 block.
    #
    # Let me reconsider. We need to partition:
    #   A columns (5) into groups, B rows (5) into matching groups, B columns (6) into groups.
    #
    # Option: 
    #   A cols: {0,1}, {2,3}, {4}
    #   B rows: {0,1}, {2,3}, {4}  
    #   B cols: {0,1}, {2,3}, {4,5}
    #
    # For (A group size, B col group size):
    #   (2, 2): 2x2x2 -> Strassen rank 7
    #   (2, 2): 2x2x2 -> Strassen rank 7
    #   (2, 2): 2x2x2 -> Strassen rank 7
    #   (2, 2): 2x2x2 -> Strassen rank 7
    #   (2, 2): 2x2x2 -> Strassen rank 7
    #   (2, 2): 2x2x2 -> Strassen rank 7
    #   (1, 2): 2x1x2 -> rank 2 (can't do better than 2 for 2x1x2? Actually 2x1x2 is just outer product of 2-vectors... no, it's A[2x1] * B[1x2] = 2x2 matrix. The tensor has rank... let me think. C = A_col * B_row^T. This is rank 1 as a tensor? No, A_col is 2-dim, B_row is 2-dim, C is 2x2. The multiplication tensor for (2,1,2) has 4 entries. U is 2x?, V is 2x?, W is 4x?. Minimum rank: each term is u_i * v_j * w_k. We need to represent c_ik = sum_r u_ir * v_jr * w_ikr... wait, j is fixed (only 1). So c_ik = sum_r u_ir * v_r * w_ikr. That's c_ik = (u * v * w_ik) where w_ik varies with i,k. Actually for j=0: c_ik = sum_r U[i,0,r] * V[0,r] * W[i,k,r]. This is a rank-1 tensor in (i,k) for each r, scaled by V[0,r]. The 2x2 matrix C = sum_r (U_col_r * V_r) * W_mat_r... no.
    #   Let me just use rank 2 for 2x1x2 blocks: use the schoolbook 2 terms.
    #   Actually, 2x1x2: A is 2x1, B is 1x2, C is 2x2. c_ik = A_i0 * B_0k. 
    #   Tensor: T[i,j,k] = delta_{j,0} * A[i,0] * B[0,k]. 
    #   As a bilinear form: C = A * B where A is 2x1, B is 1x2.
    #   Rank of this tensor: We need U (2xR), V (1xR), W (4xR).
    #   c_ik = sum_r U[i,r] * V[0,r] * W[i*n+k, r]... wait W index is k*n+i.
    #   c_ik = sum_r U[i,r] * V[0,r] * W[k*2+i, r]
    #   = sum_r (U[i,r] * V[0,r]) * W[k*2+i, r]
    #   We need this to equal A[i,0] * B[0,k] for all i,k.
    #   This is like a rank-1 matrix factorization of the 2x2 matrix A_col @ B_row.
    #   A_col @ B_row is a 2x2 matrix of rank 1 (outer product of 2-vectors).
    #   The tensor rank of a 2x1x2 multiplication is... I think it's 2.
    #   Use schoolbook: 2 terms.
    #
    # So the blocks are:
    #   For A-col groups: G_A = [(0,1), (2,3), (4,)]  -- sizes 2,2,1
    #   For B-row groups: G_B = [(0,1), (2,3), (4,)]  -- must match A-col groups
    #   For B-col groups: G_K = [(0,1), (2,3), (4,5)]  -- sizes 2,2,2
    #
    # Blocks:
    #   (G_A[0], G_K[0]): 2x2x2 -> Strassen, rank 7
    #   (G_A[0], G_K[1]): 2x2x2 -> Strassen, rank 7
    #   (G_A[0], G_K[2]): 2x2x2 -> Strassen, rank 7
    #   (G_A[1], G_K[0]): 2x2x2 -> Strassen, rank 7
    #   (G_A[1], G_K[1]): 2x2x2 -> Strassen, rank 7
    #   (G_A[1], G_K[2]): 2x2x2 -> Strassen, rank 7
    #   (G_A[2], G_K[0]): 2x1x2 -> schoolbook, rank 2
    #   (G_A[2], G_K[1]): 2x1x2 -> schoolbook, rank 2
    #   (G_A[2], G_K[2]): 2x1x2 -> schoolbook, rank 2
    #
    # Total rank: 6*7 + 3*2 = 42 + 6 = 48.
    #
    # Wait, the inspiration said 54. Let me re-read.
    # "Applying Strassen's algorithm to 2x2x2 sub-blocks reduces the rank from 60 to 54 by saving 1 rank per block (6 blocks total)."
    # 60 - 6*1 = 54. So they saved 1 rank per 2x2x2 block (8->7), and there were 6 such blocks.
    # 6*7 + something = 54. 42 + 12 = 54. So the remaining 12 rank is for the non-Strassen parts.
    # If we have 3 blocks of 2x1x2 each with rank 4 (schoolbook 2*1*2=4), that's 12. 
    # So they used schoolbook (rank 4) for the 2x1x2 blocks instead of the optimal rank 2.
    # 
    # Let me verify: can 2x1x2 be done in rank 2?
    # c_ik = A_i * B_k (where A is 2-vector, B is 2-vector). C is 2x2.
    # c_00 = A_0 * B_0
    # c_01 = A_0 * B_1
    # c_10 = A_1 * B_0
    # c_11 = A_1 * B_1
    # 
    # With rank R: c_ik = sum_{r=1}^R U_i * V_r * W_{ik}
    # (V has only 1 row since m_dim=1, so V[0,r] = V_r)
    # 
    # For R=2:
    # c_00 = U_0*V_0*W_00 + U_0*V_1*W_00' ... let me use proper notation.
    # c_ik = sum_r u_i * v_r * w_{ik}^r
    # 
    # This means C = sum_r (u_r * v_r) * W_r where u_r is 2-vector, W_r is 2x2 matrix, v_r is scalar.
    # So C is a sum of R rank-1 matrices, each scaled.
    # But C = A * B^T is itself a rank-1 matrix (outer product of A and B).
    # So R=1 should suffice! 
    # u = A, v = 1, W = B^T (as a 2x2 matrix reshaped to 4-vector).
    # Let's check: c_ik = u_i * v * w_ik = A_i * 1 * B_k = A_i * B_k. Yes!
    # So 2x1x2 has rank 1.
    #
    # Wait, but the tensor decomposition requires U, V, W with specific dimensions.
    # U: (n*m) x R = 2*1 x R = 2 x R
    # V: (m*p) x R = 1*2 x R = 2 x R
    # W: (p*n) x R = 2*2 x R = 4 x R
    # 
    # For R=1:
    # U = [[A_0], [A_1]]  -- but U must have fixed entries, not depend on A!
    # 
    # Oh right, the decomposition is for the TENSOR, not for specific matrices.
    # The entries of U, V, W are constants (coefficients), not values of A, B, C.
    # 
    # So for 2x1x2: T[i,j,k] where i in {0,1}, j in {0}, k in {0,1}.
    # T[i,0,k] = A[i,0] * B[0,k]
    # 
    # In the tensor decomposition framework:
    # C[i,k] = sum_r U[i*1+0, r] * V[0*2+k, r] * W[k*2+i, r]
    # Wait, I need to get the index conventions right.
    # 
    # U[i*m + j, r]: i is row of A (0..n-1), j is col of A (0..m-1)
    # V[j*p + k, r]: j is row of B (0..m-1), k is col of B (0..p-1)
    # W[k*n + i, r]: k is col of C (0..p-1), i is row of C (0..n-1)
    # 
    # For 2x1x2: n=2, m=1, p=2.
    # U[i*1 + 0, r] = U[i, r] for i in {0,1}
    # V[0*2 + k, r] = V[k, r] for k in {0,1}
    # W[k*2 + i, r] for k in {0,1}, i in {0,1}
    # 
    # C[i,k] = sum_r U[i,r] * V[k,r] * W[2k+i, r]
    # 
    # We need C[i,k] = A[i,0] * B[0,k].
    # The tensor T[i,j,k] for j=0 is T[i,0,k].
    # In the decomposition, T[i,0,k] = sum_r U[i*1+0, r] * V[0*2+k, r] * W[k*2+i, r]
    # = sum_r U[i,r] * V[k,r] * W[2k+i, r]
    # 
    # For this to represent the multiplication tensor, we need:
    # sum_r U[i,r] * V[k,r] * W[2k+i, r] to be a "coefficient" such that
    # C[i,k] = sum_r U[i,r] * V[k,r] * W[2k+i, r] * (A[i,0] * B[0,k])? No, that's not how it works.
    # 
    # The tensor T_{i,j,k} is the coefficient of A[i,j] * B[j,k] in C.
    # For 2x1x2, T_{i,0,k} = 1 for all i,k (since C[i,k] = A[i,0]*B[0,k]).
    # 
    # The decomposition says: T_{i,j,k} = sum_r U[i,j,r] * V[j,k,r] * W[k,i,r]
    # (using the index conventions from the problem)
    # 
    # So we need: sum_r U[i, r] * V[k, r] * W[2k+i, r] = 1 for all i in {0,1}, k in {0,1}.
    # 
    # For R=1: U[i,0] * V[k,0] * W[2k+i,0] = 1 for all i,k.
    # This means U[i,0], V[k,0], W[2k+i,0] are all nonzero and their product is 1.
    # Set U = [1, 1]^T, V = [1, 1]^T, W = [1, 1, 1, 1]^T.
    # Then sum = 1*1*1 = 1. Yes! Rank 1 works.
    #
    # So 2x1x2 has tensor rank 1.
    #
    # Total rank: 6*7 + 3*1 = 42 + 3 = 45.
    #
    # Let me implement this.

    # Strassen 2x2x2 decomposition
    # For a 2x2x2 block with A indices (i1, i2) x (j1, j2), B indices (j1, j2) x (k1, k2)
    # The 7 Strassen products are:
    # M1 = (A[i1,j1] + A[i2,j2]) * (B[j1,k1] + B[j2,k2])
    # M2 = (A[i1,j2] + A[i2,j2]) * B[j1,k1]
    # M3 = A[i1,j1] * (B[j2,k1] - B[j2,k2])
    # M4 = A[i2,j2] * (B[j1,k2] - B[j1,k1])
    # M5 = (A[i1,j1] + A[i1,j2]) * B[j2,k2]
    # M6 = (A[i2,j1] - A[i1,j1]) * (B[j1,k1] + B[j2,k1])
    # M7 = (A[i2,j2] - A[i1,j1]) * (B[j1,k2] + B[j2,k2])
    #
    # C[i1,k1] = M1 + M4 - M5 + M7
    # C[i1,k2] = M3 + M5
    # C[i2,k1] = M2 + M4
    # C[i2,k2] = M1 - M2 + M3 + M6

    # For each Strassen term r, we need U, V, W entries.
    # U[A_idx, r] = coefficient of A in M_r
    # V[B_idx, r] = coefficient of B in M_r
    # W[C_idx, r] = coefficient of M_r in C

    strassen_terms = [
        # (U_coeffs: list of (a_idx, coeff), V_coeffs: list of (b_idx, coeff), W_coeffs: list of (c_idx, coeff))
        # M1 = (A00 + A11) * (B00 + B11)
        ([(0, 1), (3, 1)], [(0, 1), (3, 1)], [(0, 1), (3, 1)]),  # C00 += M1, C11 += M1
        # M2 = (A01 + A11) * B00
        ([(1, 1), (3, 1)], [(0, 1)], [(1, 1)]),  # C10 += M2... wait let me recheck
    ]
    
    # Let me be more careful with indices.
    # Local A: [[a, b], [c, d]] where a=A[i1,j1], b=A[i1,j2], c=A[i2,j1], d=A[i2,j2]
    # Local B: [[e, f], [g, h]] where e=B[j1,k1], f=B[j1,k2], g=B[j2,k1], h=B[j2,k2]
    # Local C: [[p, q], [r, s]] where p=C[i1,k1], q=C[i1,k2], r=C[i2,k1], s=C[i2,k2]
    #
    # M1 = (a+d)(e+h)
    # M2 = (c+d)e
    # M3 = a(f-h)
    # M4 = d(g-e)
    # M5 = (a+b)h
    # M6 = (c-a)(e+g)
    # M7 = (b-d)(g+h)
    #
    # p = M1 + M4 - M5 + M7
    # q = M3 + M5
    # r = M2 + M4
    # s = M1 - M2 + M3 + M6
    
    # U entries for local A indices: a=0, b=1, c=2, d=3
    # V entries for local B indices: e=0, f=1, g=2, h=3
    # W entries for local C indices: p=0, q=1, r=2, s=3
    
    strassen = [
        # (U_coeffs, V_coeffs, W_coeffs)
        ([(0, 1), (3, 1)], [(0, 1), (3, 1)], [(0, 1), (3, 1)]),  # M1
        ([(2, 1), (3, 1)], [(0, 1)], [(2, 1)]),  # M2: r += M2
        ([(0, 1)], [(1, 1), (3, -1)], [(1, 1)]),  # M3: q += M3
        ([(3, 1)], [(2, 1), (0, -1)], [(0, 1), (2, 1)]),  # M4: p += M4, r += M4
        ([(0, 1), (1, 1)], [(3, 1)], [(0, -1), (1, 1)]),  # M5: p -= M5, q += M5
        ([(2, 1), (0, -1)], [(0, 1), (2, 1)], [(3, 1)]),  # M6: s += M6
        ([(1, 1), (3, -1)], [(2, 1), (3, 1)], [(0, 1)]),  # M7: p += M7
    ]
    
    # Build the decomposition
    U_rows = n * m  # 10
    V_rows = m * p  # 30
    W_rows = p * n  # 12
    
    # We'll build column by column
    columns = []  # Each column is (U_col, V_col, W_col)
    
    def add_column(u_entries, v_entries, w_entries):
        """u_entries: list of (global_idx, coeff), etc."""
        u_col = [0.0] * U_rows
        v_col = [0.0] * V_rows
        w_col = [0.0] * W_rows
        for idx, coeff in u_entries:
            u_col[idx] += coeff
        for idx, coeff in v_entries:
            v_col[idx] += coeff
        for idx, coeff in w_entries:
            w_col[idx] += coeff
        columns.append((u_col, v_col, w_col))
    
    # Define the blocks
    # A column groups and corresponding B row groups:
    # Group 0: A cols 0,1; B rows 0,1
    # Group 1: A cols 2,3; B rows 2,3
    # Group 2: A col 4; B row 4
    # B column groups:
    # Group 0: B cols 0,1
    # Group 1: B cols 2,3
    # Group 2: B cols 4,5
    
    a_groups = [(0, 1), (2, 3), (4,)]
    b_col_groups = [(0, 1), (2, 3), (4, 5)]
    
    for ag_idx, (ja, jb) in enumerate(a_groups):
        if time.time() > deadline:
            break
        for bk_idx, (ka, kb) in enumerate(b_col_groups):
            if len(a_groups[ag_idx]) == 2 and len(b_col_groups[bk_idx]) == 2:
                # 2x2x2 Strassen block
                i1, i2 = 0, 1
                # Local to global index mappings
                # A local indices: 0=(i1,ja), 1=(i1,jb), 2=(i2,ja), 3=(i2,jb)
                a_local_to_global = {
                    0: i1 * m + ja,
                    1: i1 * m + jb,
                    2: i2 * m + ja,
                    3: i2 * m + jb,
                }
                # B local indices: 0=(ja,ka), 1=(ja,kb), 2=(jb,ka), 3=(jb,kb)
                b_local_to_global = {
                    0: ja * p + ka,
                    1: ja * p + kb,
                    2: jb * p + ka,
                    3: jb * p + kb,
                }
                # C local indices: 0=(i1,ka), 1=(i1,kb), 2=(i2,ka), 3=(i2,kb)
                c_local_to_global = {
                    0: ka * n + i1,
                    1: kb * n + i1,
                    2: ka * n + i2,
                    3: kb * n + i2,
                }
                
                for u_coeffs, v_coeffs, w_coeffs in strassen:
                    u_entries = [(a_local_to_global[lidx], coeff) for lidx, coeff in u_coeffs]
                    v_entries = [(b_local_to_global[lidx], coeff) for lidx, coeff in v_coeffs]
                    w_entries = [(c_local_to_global[lidx], coeff) for lidx, coeff in w_coeffs]
                    add_column(u_entries, v_entries, w_entries)
            else:
                # 2x1x2 block (single A column, two B columns)
                # Rank 1 decomposition
                # C[i, k] = A[i, ja] * B[ja, k]
                # T[i, ja, k] = 1
                # Use U[i*m+ja, 0] = 1, V[ja*p+k, 0] = 1, W[k*n+i, 0] = 1
                # Wait, that's rank 4 schoolbook. For rank 1:
                # sum_r U[i*m+ja, r] * V[ja*p+k, r] * W[k*n+i, r] = 1
                # With R=1: U[i*m+ja, 0] * V[ja*p+k, 0] * W[k*n+i, 0] = 1 for all i,k
                # Set all to 1.
                i1 = 0
                i2 = 1
                u_entries = [(i1 * m + ja, 1), (i2 * m + ja, 1)]
                v_entries = [(ja * p + ka, 1), (ja * p + kb, 1)]
                w_entries = [(ka * n + i1, 1), (kb * n + i1, 1),
                             (ka * n + i2, 1), (kb * n + i2, 1)]
                add_column(u_entries, v_entries, w_entries)
    
    rank = len(columns)
    
    # Build U, V, W as lists of lists
    U = [[0] * rank for _ in range(U_rows)]
    V = [[0] * rank for _ in range(V_rows)]
    W = [[0] * rank for _ in range(W_rows)]
    
    for r, (u_col, v_col, w_col) in enumerate(columns):
        for i in range(U_rows):
            U[i][r] = u_col[i]
        for j in range(V_rows):
            V[j][r] = v_col[j]
        for k in range(W_rows):
            W[k][r] = w_col[k]
    
    # Verify the decomposition
    if time.time() < deadline - 10:
        import numpy as np
        T = np.zeros((n*m, m*p, p*n), dtype=object)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    T[i*m+j, j*p+k, k*n+i] = 1
        
        U_np = np.array(U, dtype=object)
        V_np = np.array(V, dtype=object)
        W_np = np.array(W, dtype=object)
        
        # Check: T[i,j,k] = sum_r U[i,j,r] * V[j,k,r] * W[k,i,r]
        # Reshape U to (n, m, rank), V to (m, p, rank), W to (p, n, rank)
        U_reshaped = U_np.reshape(n, m, rank)
        V_reshaped = V_np.reshape(m, p, rank)
        W_reshaped = W_np.reshape(p, n, rank)
        
        # Compute tensor product sum
        # T[i,j,k] = sum_r U[i,j,r] * V[j,k,r] * W[k,i,r]
        # This is a bit tricky with numpy. Let's just do it manually for verification.
        T_computed = np.zeros((n*m, m*p, p*n), dtype=object)
        for r in range(rank):
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        T_computed[i*m+j, j*p+k, k*n+i] += U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r]
        
        # Check if T_computed matches T
        max_err = 0
        for i in range(n*m):
            for j in range(m*p):
                for k in range(p*n):
                    if T[i,j,k] != T_computed[i,j,k]:
                        max_err = 1
                        break
                if max_err:
                    break
            if max_err:
                break
        
        if max_err:
            # Fallback to schoolbook
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
# EVOLVE-BLOCK-END
