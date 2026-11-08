"""Tensor decomposition for matrix multiplication 3x3 * 3x8.
Uses an optimized block decomposition targeting rank 68.
"""

DIMENSIONS = [3, 3, 8]

import numpy as np


def construct():
    n, m, p = DIMENSIONS  # 3, 3, 8

    # We decompose C = A * B where A is n x m (3x3), B is m x p (3x8), C is n x p (3x8).
    # T[i, j, k] = 1 for all valid indices, meaning A[i,j] * B[j,k] contributes to C[i,k].
    
    # Strategy:
    # The middle dimension m=3. We split the computation based on the column j of A (which is the row of B).
    # j=0, j=1, j=2.
    
    # For j=0 and j=1 (the first two "middle" indices), we can group the n-dimensions (rows of A/C) and p-dimensions (cols of B/C).
    # Specifically, we have A[0:2, 0:2] (rows 0,1 of A, cols 0,1) and B[0:2, :] (rows 0,1 of B, all 8 cols).
    # This is a 2x2 * 2x8 multiplication.
    # We can decompose the 2x2 * 2x8 block.
    # The 2x2 * 2x8 tensor has rank 7 (from Strassen for the 2x2*2x2 part) * 4 (for the 4 column blocks of 2)? No.
    # Strassen reduces 2x2*2x2 to rank 7.
    # For 2x2*2x8, we can apply Strassen to each of the four 2x2*2x2 column blocks?
    # A[0:2, 0:2] * B[0:2, 0:2] -> rank 7
    # A[0:2, 0:2] * B[0:2, 2:4] -> rank 7
    # A[0:2, 0:2] * B[0:2, 4:6] -> rank 7
    # A[0:2, 0:2] * B[0:2, 6:8] -> rank 7
    # Total for j=0,1 part: 28.
    # But wait, A[0:2, 0:2] means rows 0,1 of A, cols 0,1.
    # This covers A[0,0], A[0,1], A[1,0], A[1,1].
    # And B[0,0..7], B[1,0..7].
    # This covers contributions to C[0,0..7] and C[1,0..7] from j=0 and j=1.
    
    # Remaining terms:
    # j=0,1 but from A[2, 0:2] (row 2, cols 0,1): A[2,0]*B[0,k] + A[2,1]*B[1,k].
    # This is a 1x2 * 2x8 multiplication.
    # Rank of 1x2 * 2x8?
    # C[2, k] = A[2,0]*B[0,k] + A[2,1]*B[1,k].
    # This is a linear combination. The tensor is T[2, (0,k), (1,k)]? No.
    # Indices: i=2, j=0,1, k=0..7.
    # T[2, 0, k] = 1, T[2, 1, k] = 1.
    # This can be written as sum over k of (A[2,0]*B[0,k] + A[2,1]*B[1,k]).
    # This is a rank 8 tensor?
    # Let's check:
    # Term 1: A[2,0]*B[0,0] -> C[2,0]
    # ...
    # Term 8: A[2,0]*B[0,7] -> C[2,7]
    # Term 9: A[2,1]*B[1,0] -> C[2,0]
    # ...
    # Actually, for each k, we have two terms adding to C[2,k].
    # C[2,k] = A[2,0]B[0,k] + A[2,1]B[1,k].
    # This is a bilinear form.
    # Can we do it in rank 2?
    # U1 = A[2,0], V1 = [B[0,0]..B[0,7]], W1 = [C[2,0]..C[2,7]]? No, W must be single index.
    # The tensor T_{i,j,k} has support at i=2, j=0, k=0..7 and i=2, j=1, k=0..7.
    # This is equivalent to two rank-8 tensors? Or can we combine?
    # T = sum_k (e_2 e_0^T e_k^T + e_2 e_1^T e_k^T)
    # = sum_k e_2 (e_0 + e_1)^T e_k^T ? No, that would give A[2,0+1] which doesn't exist.
    # It is T = sum_k e_2 e_0^T e_k^T + sum_k e_2 e_1^T e_k^T.
    # These are disjoint in the j-index, so they are linearly independent.
    # Rank = 8 + 8 = 16.
    
    # Similarly for A[0:2, 2] (rows 0,1, col 2):
    # A[0,2]*B[2,k] + A[1,2]*B[2,k].
    # i=0,1; j=2; k=0..7.
    # Rank 8 (for i=0) + Rank 8 (for i=1) = 16.
    
    # And A[2,2] (row 2, col 2):
    # A[2,2]*B[2,k].
    # i=2, j=2, k=0..7.
    # Rank 8.
    
    # Total: 28 (Strassen) + 16 (row 2, cols 0,1) + 16 (rows 0,1, col 2) + 8 (row 2, col 2) = 68.
    
    # Let's build the factors.
    
    U_terms = []
    V_terms = []
    W_terms = []
    
    def add_strassen(row_a_start, col_a_start, row_b_start, col_b_start):
        """
        Adds Strassen decomposition for A[row_a_start:row_a_start+2, col_a_start:col_a_start+2]
        * B[row_b_start:row_b_start+2, col_b_start:col_b_start+2]
        """
        # A elements
        a = (row_a_start, col_a_start)
        b = (row_a_start, col_a_start + 1)
        c = (row_a_start + 1, col_a_start)
        d = (row_a_start + 1, col_a_start + 1)
        
        # B elements
        e = (row_b_start, col_b_start)
        f = (row_b_start, col_b_start + 1)
        g = (row_b_start + 1, col_b_start)
        h = (row_b_start + 1, col_b_start + 1)
        
        # C elements
        c00 = (row_a_start, col_b_start)
        c01 = (row_a_start, col_b_start + 1)
        c10 = (row_a_start + 1, col_b_start)
        c11 = (row_a_start + 1, col_b_start + 1)
        
        # m1 = (a+d)(e+h)
        U_terms.append([(a, 1), (d, 1)])
        V_terms.append([(e, 1), (h, 1)])
        W_terms.append([(c00, 1), (c11, 1)])
        
        # m2 = (c+d)e
        U_terms.append([(c, 1), (d, 1)])
        V_terms.append([(e, 1)])
        W_terms.append([(c10, 1), (c11, -1)])
        
        # m3 = a(f-h)
        U_terms.append([(a, 1)])
        V_terms.append([(f, 1), (h, -1)])
        W_terms.append([(c01, 1), (c11, 1)])
        
        # m4 = d(g-e)
        U_terms.append([(d, 1)])
        V_terms.append([(g, 1), (e, -1)])
        W_terms.append([(c00, 1), (c10, 1)])
        
        # m5 = (a+b)h
        U_terms.append([(a, 1), (b, 1)])
        V_terms.append([(h, 1)])
        W_terms.append([(c00, -1), (c01, 1)])
        
        # m6 = (c-a)(e+f)
        U_terms.append([(c, 1), (a, -1)])
        V_terms.append([(e, 1), (f, 1)])
        W_terms.append([(c11, 1)])
        
        # m7 = (b-d)(g+h)
        U_terms.append([(b, 1), (d, -1)])
        V_terms.append([(g, 1), (h, 1)])
        W_terms.append([(c00, 1)])
        
    def add_direct(i, j, k):
        U_terms.append([((i, j), 1)])
        V_terms.append([((j, k), 1)])
        W_terms.append([((i, k), 1)])
        
    # 1. Strassen for A[0:2, 0:2] * B[0:2, 0:2]
    add_strassen(0, 0, 0, 0)
    # 2. Strassen for A[0:2, 0:2] * B[0:2, 2:4]
    add_strassen(0, 0, 0, 2)
    # 3. Strassen for A[0:2, 0:2] * B[0:2, 4:6]
    add_strassen(0, 0, 0, 4)
    # 4. Strassen for A[0:2, 0:2] * B[0:2, 6:8]
    add_strassen(0, 0, 0, 6)
    
    # 5. Direct for A[2, 0:2] * B[0:2, 0:8]
    # A[2,0]*B[0,k] and A[2,1]*B[1,k] for k=0..7
    for j in [0, 1]:
        for k in range(8):
            add_direct(2, j, k)
            
    # 6. Direct for A[0:2, 2] * B[2, 0:8]
    # A[0,2]*B[2,k] and A[1,2]*B[2,k] for k=0..7
    for i in [0, 1]:
        for k in range(8):
            add_direct(i, 2, k)
            
    # 7. Direct for A[2, 2] * B[2, 0:8]
    # A[2,2]*B[2,k] for k=0..7
    for k in range(8):
        add_direct(2, 2, k)
        
    rank = len(W_terms)
    
    # Build matrices
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    for r in range(rank):
        for (i, j), val in U_terms[r]:
            U[i * m + j, r] = val
        for (j, k), val in V_terms[r]:
            V[j * p + k, r] = val
        for (i, k), val in W_terms[r]:
            W[k * n + i, r] = val
            
    # Convert to lists of ints/floats
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()
    
    # Cleanup small floats
    def clean(val):
        if abs(val - round(val)) < 1e-9:
            return int(round(val))
        return val
        
    U_list = [[clean(x) for x in row] for row in U_list]
    V_list = [[clean(x) for x in row] for row in V_list]
    W_list = [[clean(x) for x in row] for row in W_list]
    
    return {"U": U_list, "V": V_list, "W": W_list}
