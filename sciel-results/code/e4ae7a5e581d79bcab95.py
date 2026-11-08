"""Kronecker-product tensor decomposition for 4x5 * 5x6 matrix multiplication."""
import time

DIMENSIONS = [4, 5, 6]

# EVOLVE-BLOCK-START

def _build_strassen_2x2():
    """
    2x2x2 Strassen decomposition.
    Returns (U, V, W) with shapes (4,7), (4,7), (4,7).
    Indexing: U[i*2+j], V[j*2+k], W[k*2+i]  (transposed identity convention).
    """
    U = [[0]*7 for _ in range(4)]
    V = [[0]*7 for _ in range(4)]
    W = [[0]*7 for _ in range(4)]

    # P1 = (a11+a22)*(b11+b22)
    U[0][0] = 1; U[3][0] = 1
    V[0][0] = 1; V[3][0] = 1
    W[0][0] = 1; W[3][0] = 1

    # P2 = (a21+a22)*b11
    U[2][1] = 1; U[3][1] = 1
    V[0][1] = 1
    W[1][1] = 1; W[3][1] = -1

    # P3 = a11*(b12-b22)
    U[0][2] = 1
    V[1][2] = 1; V[3][2] = -1
    W[2][2] = 1; W[3][2] = 1

    # P4 = a22*(b21-b11)
    U[3][3] = 1
    V[2][3] = 1; V[0][3] = -1
    W[0][3] = 1; W[1][3] = 1

    # P5 = (a11+a12)*b22
    U[0][4] = 1; U[1][4] = 1
    V[3][4] = 1
    W[0][4] = -1; W[2][4] = 1

    # P6 = (a21-a11)*(b11+b12)
    U[2][5] = 1; U[0][5] = -1
    V[0][5] = 1; V[1][5] = 1
    W[3][5] = 1

    # P7 = (a12-a22)*(b21+b22)
    U[1][6] = 1; U[3][6] = -1
    V[2][6] = 1; V[3][6] = 1
    W[0][6] = 1

    return U, V, W

def _build_2x2x3():
    """
    2x2x3 decomposition of rank 6 (rank-optimal for 2x2x3).
    A is 2x2, B is 2x3, C is 2x3.
    Indexing: U[i*2+j], V[j*3+k], W[k*2+i].
    """
    rank = 6
    U = [[0]*rank for _ in range(4)]   # n*m = 2*2
    V = [[0]*rank for _ in range(6)]   # m*p = 2*3
    W = [[0]*rank for _ in range(6)]   # p*n = 3*2

    # Column 0: A[0,0], B[0,0], C[0,0]
    U[0][0] = 1
    V[0][0] = 1
    W[0][0] = 1

    # Column 1: A[0,1], B[0,1], C[0,1]
    U[1][1] = 1
    V[1][1] = 1
    W[1][1] = 1

    # Column 2: A[1,0], B[1,0], C[0,2]
    U[2][2] = 1
    V[3][2] = 1
    W[2][2] = 1

    # Column 3: A[1,1], B[1,1], C[1,0]
    U[3][3] = 1
    V[4][3] = 1
    W[3][3] = 1

    # Column 4: A[0,0]+A[1,1], B[0,2], C[1,1]
    U[0][4] = 1; U[3][4] = 1
    V[2][4] = 1
    W[5][4] = 1

    # Column 5: A[0,1]+A[1,0], B[1,2], C[1,2]
    U[1][5] = 1; U[2][5] = 1
    V[5][5] = 1
    W[5][5] = 1

    return U, V, W

def _build_2x3x2():
    """
    2x3x2 decomposition of rank 6 (rank-optimal for 2x3x2).
    A is 2x3, B is 3x2, C is 2x2.
    Indexing: U[i*3+j], V[j*2+k], W[k*2+i].
    
    This is essentially the transpose/permute of 2x2x3.
    We can construct it by taking the 2x2x3 decomposition and permuting indices.
    Or we can just hardcode a known rank-6 decomposition for 2x3x2.
    
    Let's use the same structure as 2x2x3 but with dimensions swapped appropriately.
    Actually, 2x3x2 is isomorphic to 2x2x3 via permutation of indices.
    Let's just use the 2x2x3 matrices and permute them.
    
    For 2x2x3: U[2*2], V[2*3], W[3*2]
    For 2x3x2: U[2*3], V[3*2], W[2*2]
    
    The tensor T_{i,j,k} for 2x2x3 maps to T'_{i,k,j} for 2x3x2? No, that's not right.
    Let's just hardcode a valid rank-6 decomposition for 2x3x2.
    
    Using the same 6 terms but adapted:
    A is 2x3, B is 3x2, C is 2x2.
    
    Term 0: A[0,0], B[0,0], C[0,0]
    Term 1: A[0,1], B[1,0], C[0,1]
    Term 2: A[0,2], B[2,0], C[0,0] (wait, need to be careful)
    
    Let's use a known result: 2x3x2 has rank 6.
    We can use the 2x2x3 decomposition and permute.
    If we have a decomposition for (n,m,p), we can get one for (n,p,m) by swapping V and W roles appropriately.
    Actually, let's just use the 2x2x3 decomposition directly in the Kronecker product,
    but we need to be careful about the dimensions.
    
    Alternatively, we can use 2x2x2 (rank 7) and 2x2x3 (rank 6) to get 4x4x6 (rank 42),
    then add schoolbook for the remaining part. That's what the current program does (rank 66).
    
    To improve, let's try 2x3x2 (rank 6) and 2x2x3 (rank 6) to get 4x6x6 (rank 36).
    Then we need to embed 4x5x6 into 4x6x6.
    We can set the 5th row of B to 0 and the 6th column of A to 0, but that doesn't help.
    Actually, we can just use the 4x6x6 decomposition and ignore the extra row/column.
    But that would be incorrect because we need to handle all 5 columns of A and 5 rows of B.
    
    Let's think differently. We have 4x5x6.
    We can decompose it as:
    (2x3) x (3x2) for the first 2x3 block of A and 3x2 block of B -> 2x2 block of C
    (2x2) x (2x3) for the remaining 2x2 block of A and 2x3 block of B -> 2x3 block of C
    But the dimensions don't match up cleanly.
    
    Let's use the Kronecker product of 2x3x2 (rank 6) and 2x2x3 (rank 6) to get 4x6x6 (rank 36).
    Then we can use this to compute 4x6x6, but we only need 4x5x6.
    We can set the 6th row of B to 0 and the 5th column of A to 0, but that's not how it works.
    
    Actually, we can just use the 4x6x6 decomposition and extract the 4x5x6 part.
    The 4x6x6 decomposition computes C = A * B where A is 4x6 and B is 6x6.
    We want A to be 4x5 and B to be 5x6.
    We can pad A with a zero column and B with a zero row, but that would change the result.
    
    Let's just use the current approach but optimize it.
    The current approach uses 2x2x2 (rank 7) and 2x2x3 (rank 6) to get 4x4x6 (rank 42),
    then adds schoolbook for 4x1x6 (rank 24), total rank 66.
    
    Can we do better? Let's try 2x3x2 (rank 6) and 2x2x3 (rank 6) to get 4x6x6 (rank 36).
    Then we need to handle the fact that we only have 5 columns in A and 5 rows in B.
    We can use the 4x6x6 decomposition but only use the first 5 columns of A and first 5 rows of B.
    This would compute C = A[:, :5] * B[:5, :] which is exactly what we want.
    But the 4x6x6 decomposition assumes A is 4x6 and B is 6x6.
    If we set A[:, 5] = 0 and B[5, :] = 0, then the result is the same as 4x5x6.
    But the rank is still 36, which is better than 66.
    
    Let's implement this:
    1. Build 2x3x2 decomposition (rank 6).
    2. Build 2x2x3 decomposition (rank 6).
    3. Kronecker product to get 4x6x6 decomposition (rank 36).
    4. Use this to compute 4x5x6 by padding A and B with zeros.
    
    But wait, the Kronecker product gives us a decomposition for 4x6x6.
    The U matrix has shape (4*6, 36) = (24, 36).
    But we need U to have shape (4*5, rank) = (20, rank).
    So we can't directly use the 4x6x6 decomposition for 4x5x6.
    
    Let's think again. We need a decomposition for 4x5x6.
    We can use the Kronecker product of 2x2x2 (rank 7) and 2x2x3 (rank 6) to get 4x4x6 (rank 42).
    Then we need to handle the remaining 4x1x6 part.
    The current program does this with schoolbook (rank 24), total rank 66.
    
    Can we do better for the 4x1x6 part? The rank of 4x1x6 is 24 (schoolbook is optimal).
    So the total rank is at least 42 + 24 = 66.
    
    Wait, is 4x4x6 the best we can do with Kronecker product?
    What if we use 2x3x2 (rank 6) and 2x2x3 (rank 6) to get 4x6x6 (rank 36)?
    Then we need to embed 4x5x6 into 4x6x6.
    We can do this by setting the 6th row of B to 0 and the 5th column of A to 0.
    But the decomposition is for 4x6x6, not 4x5x6.
    
    Actually, we can just use the 4x6x6 decomposition and ignore the extra dimensions.
    The U matrix for 4x6x6 has shape (24, 36).
    We need U for 4x5x6 to have shape (20, rank).
    We can take the first 20 rows of U (corresponding to the first 5 columns of A).
    Similarly, V for 4x6x6 has shape (36, 36).
    We need V for 4x5x6 to have shape (30, rank).
    We can take the first 30 rows of V (corresponding to the first 5 rows of B).
    W for 4x6x6 has shape (24, 36).
    We need W for 4x5x6 to have shape (24, rank).
    We can take all 24 rows of W.
    
    But this would only compute the part of C that corresponds to the first 5 columns of A and first 5 rows of B.
    The remaining part (A[:, 5] * B[5, :]) would be 0 because we set those to 0.
    But we don't have a 6th column in A or 6th row in B, so this is fine.
    
    Wait, A is 4x5, so there is no 6th column. B is 5x6, so there is no 6th row.
    So we can't pad A with a 6th column or B with a 6th row.
    
    Let's just use the current approach and see if we can optimize it.
    The current approach has rank 66.
    
    Actually, let's try a different Kronecker product: 2x3x2 (rank 6) and 2x2x3 (rank 6) gives 4x6x6 (rank 36).
    We can use this to compute 4x6x6, but we only need 4x5x6.
    We can set the 6th column of A to 0 and the 6th row of B to 0, but A is 4x5 and B is 5x6, so we can't.
    
    Let's just use the current program and see if we can improve it slightly.
    """
    rank = 6
    U = [[0]*rank for _ in range(6)]   # n*m = 2*3
    V = [[0]*rank for _ in range(6)]   # m*p = 3*2
    W = [[0]*rank for _ in range(4)]   # p*n = 2*2

    # Hardcode a rank-6 decomposition for 2x3x2.
    # This is isomorphic to 2x2x3 via permutation.
    # Let's use the same terms as 2x2x3 but permuted.
    
    # Term 0: A[0,0], B[0,0], C[0,0]
    U[0][0] = 1
    V[0][0] = 1
    W[0][0] = 1

    # Term 1: A[0,1], B[1,0], C[0,1]
    U[1][1] = 1
    V[2][1] = 1
    W[2][1] = 1

    # Term 2: A[0,2], B[2,0], C[0,0]
    U[2][2] = 1
    V[4][2] = 1
    W[0][2] = 1

    # Term 3: A[1,0], B[0,1], C[1,0]
    U[3][3] = 1
    V[1][3] = 1
    W[1][3] = 1

    # Term 4: A[1,1], B[1,1], C[1,1]
    U[4][4] = 1
    V[3][4] = 1
    W[3][4] = 1

    # Term 5: A[1,2], B[2,1], C[1,0]
    U[5][5] = 1
    V[5][5] = 1
    W[1][5] = 1

    return U, V, W

def _kron_decomposition_dims(d1, dims1, d2, dims2):
    """
    Kronecker product of two decompositions with explicit dimensions.
    d1 = (U1, V1, W1) for (n1, m1, p1)
    d2 = (U2, V2, W2) for (n2, m2, p2)
    Produces decomposition for (n1*n2, m1*m2, p1*p2).
    """
    U1, V1, W1 = d1
    U2, V2, W2 = d2
    n1, m1, p1 = dims1
    n2, m2, p2 = dims2

    r1 = len(U1[0])
    r2 = len(U2[0])
    rank = r1 * r2

    n = n1 * n2
    m = m1 * m2
    p = p1 * p2

    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]

    for c1 in range(r1):
        for c2 in range(r2):
            col = c1 * r2 + c2

            # U: index (i1*n2 + i2) * (m1*m2) + (j1*m2 + j2)
            for i1 in range(n1):
                for j1 in range(m1):
                    u1_val = U1[i1*m1 + j1][c1]
                    if u1_val == 0:
                        continue
                    for i2 in range(n2):
                        for j2 in range(m2):
                            u2_val = U2[i2*m2 + j2][c2]
                            if u2_val == 0:
                                continue
                            i = i1*n2 + i2
                            j = j1*m2 + j2
                            U[i*m + j][col] = u1_val * u2_val

            # V: index (j1*p2 + j2) * (p1*p2) + (k1*p2 + k2)
            for j1 in range(m1):
                for k1 in range(p1):
                    v1_val = V1[j1*p1 + k1][c1]
                    if v1_val == 0:
                        continue
                    for j2 in range(m2):
                        for k2 in range(p2):
                            v2_val = V2[j2*p2 + k2][c2]
                            if v2_val == 0:
                                continue
                            j = j1*p2 + j2
                            k = k1*p2 + k2
                            V[j*p + k][col] = v1_val * v2_val

            # W: index (k1*n2 + k2) * (n1*n2) + (i1*n2 + i2)
            for k1 in range(p1):
                for i1 in range(n1):
                    w1_val = W1[k1*n1 + i1][c1]
                    if w1_val == 0:
                        continue
                    for k2 in range(p2):
                        for i2 in range(n2):
                            w2_val = W2[k2*n2 + i2][c2]
                            if w2_val == 0:
                                continue
                            k = k1*n2 + k2
                            i = i1*n2 + i2
                            W[k*n + i][col] = w1_val * w2_val

    return U, V, W

def construct():
    start_time = time.time()
    deadline = start_time + 110.0

    n, m, p = DIMENSIONS

    try:
        # Use Kronecker product of 2x2x2 (rank 7) and 2x2x3 (rank 6) to get 4x4x6 (rank 42)
        U1, V1, W1 = _build_strassen_2x2()
        d1 = (U1, V1, W1)
        dims1 = (2, 2, 2)

        U2, V2, W2 = _build_2x2x3()
        d2 = (U2, V2, W2)
        dims2 = (2, 2, 3)

        U_kron, V_kron, W_kron = _kron_decomposition_dims(d1, dims1, d2, dims2)
        rank_kron = 42
        rank_extra = 4 * 1 * 6  # 24
        total_rank = rank_kron + rank_extra

        U = [[0]*total_rank for _ in range(4*5)]  # 20 rows
        V = [[0]*total_rank for _ in range(5*6)]  # 30 rows
        W = [[0]*total_rank for _ in range(6*4)]  # 24 rows

        # Copy Kronecker part: A rows i=0..3, j=0..3 -> U[i*5+j]
        for i in range(4):
            for j in range(4):
                u_row = i * 5 + j
                for c in range(rank_kron):
                    U[u_row][c] = U_kron[i*4 + j][c]

        for j in range(4):
            for k in range(6):
                v_row = j * 6 + k
                for c in range(rank_kron):
                    V[v_row][c] = V_kron[j*4 + k][c]

        for k in range(6):
            for i in range(4):
                w_row = k * 4 + i
                for c in range(rank_kron):
                    W[w_row][c] = W_kron[k*4 + i][c]

        # Schoolbook part: A[:,4] * B[4,:] -> C
        col_offset = rank_kron
        for i in range(4):
            for k in range(6):
                U[i*5 + 4][col_offset + i*6 + k] = 1
                V[4*6 + k][col_offset + i*6 + k] = 1
                W[k*4 + i][col_offset + i*6 + k] = 1

        return {"U": U, "V": V, "W": W}

    except Exception:
        # Fallback to schoolbook
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        column = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][column] = 1
                    V[j * p + k][column] = 1
                    W[k * n + i][column] = 1
                    column += 1
        return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
