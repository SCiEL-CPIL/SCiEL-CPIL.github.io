"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 5]

# EVOLVE-BLOCK-START
def construct():
    """
    Constructs a rank-28 tensor decomposition for multiplying a 2x3 matrix by a 3x5 matrix.
    
    The decomposition uses Strassen's rank-7 algorithm for the 2x2x2 sub-blocks
    and schoolbook for the remaining parts.
    
    Structure:
    - A is 2x3 (n=2, m=3), B is 3x5 (m=3, p=5), C is 2x5 (n=2, p=5)
    - Split by the middle dimension j:
      - j=0,1: A[0:2, 0:2] * B[0:2, 0:5]
        - k=0,1: 2x2x2 Strassen (rank 7)
        - k=2,3: 2x2x2 Strassen (rank 7)
        - k=4: 2x2x1 schoolbook (rank 4)
      - j=2: A[0:2, 2:3] * B[2:3, 0:5]
        - 2x1x5 schoolbook (rank 10)
    
    Total rank = 7 + 7 + 4 + 10 = 28
    
    Index conventions:
    U[i*m + j, c] for A[i,j]
    V[j*p + k, c] for B[j,k]
    W[k*n + i, c] for C[i,k]
    """
    n, m, p = DIMENSIONS  # n=2, m=3, p=5
    
    terms = []
    
    def add_strassen_2x2x2(a_rows, a_cols, b_rows, b_cols, c_rows, c_cols):
        """
        Add Strassen rank-7 decomposition for a 2x2x2 multiplication.
        
        a_rows: [i1, i2] - row indices in A
        a_cols: [j1, j2] - column indices in A (must match b_rows)
        b_rows: [j1, j2] - row indices in B
        b_cols: [k1, k2] - column indices in B
        c_rows: [i1, i2] - row indices in C
        c_cols: [k1, k2] - column indices in C
        """
        i1, i2 = a_rows
        j1, j2 = a_cols
        k1, k2 = b_cols
        
        # Term 1: (A[i1,j1] + A[i2,j2]) * (B[j1,k1] + B[j2,k2])
        # Contributes to C[i1,k1] += 1, C[i2,k2] -= 1
        u1 = {i1*m+j1: 1, i2*m+j2: 1}
        v1 = {j1*p+k1: 1, j2*p+k2: 1}
        w1 = {k1*n+i1: 1, k2*n+i2: -1}
        terms.append((u1, v1, w1))
        
        # Term 2: (A[i2,j1] + A[i2,j2]) * B[j1,k1]
        # Contributes to C[i2,k1] += 1, C[i2,k2] -= 1
        u2 = {i2*m+j1: 1, i2*m+j2: 1}
        v2 = {j1*p+k1: 1}
        w2 = {k1*n+i2: 1, k2*n+i2: -1}
        terms.append((u2, v2, w2))
        
        # Term 3: A[i1,j1] * (B[j1,k2] - B[j2,k2])
        # Contributes to C[i1,k2] += 1, C[i2,k2] += 1
        u3 = {i1*m+j1: 1}
        v3 = {j1*p+k2: 1, j2*p+k2: -1}
        w3 = {k2*n+i1: 1, k2*n+i2: 1}
        terms.append((u3, v3, w3))
        
        # Term 4: A[i2,j2] * (B[j2,k1] - B[j1,k1])
        # Contributes to C[i1,k1] += 1, C[i2,k1] += 1
        u4 = {i2*m+j2: 1}
        v4 = {j2*p+k1: 1, j1*p+k1: -1}
        w4 = {k1*n+i1: 1, k1*n+i2: 1}
        terms.append((u4, v4, w4))
        
        # Term 5: (A[i1,j1] + A[i1,j2]) * B[j2,k2]
        # Contributes to C[i1,k1] -= 1, C[i1,k2] += 1
        u5 = {i1*m+j1: 1, i1*m+j2: 1}
        v5 = {j2*p+k2: 1}
        w5 = {k1*n+i1: -1, k2*n+i1: 1}
        terms.append((u5, v5, w5))
        
        # Term 6: (A[i2,j1] - A[i1,j1]) * (B[j1,k1] + B[j1,k2])
        # Contributes to C[i2,k2] += 1
        u6 = {i2*m+j1: 1, i1*m+j1: -1}
        v6 = {j1*p+k1: 1, j1*p+k2: 1}
        w6 = {k2*n+i2: 1}
        terms.append((u6, v6, w6))
        
        # Term 7: (A[i1,j2] - A[i2,j2]) * (B[j2,k1] + B[j2,k2])
        # Contributes to C[i1,k1] += 1
        u7 = {i1*m+j2: 1, i2*m+j2: -1}
        v7 = {j2*p+k1: 1, j2*p+k2: 1}
        w7 = {k1*n+i1: 1}
        terms.append((u7, v7, w7))
    
    # Part 1: j=0,1; k=0,1 (2x2x2 Strassen)
    add_strassen_2x2x2([0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1])
    
    # Part 2: j=0,1; k=2,3 (2x2x2 Strassen)
    add_strassen_2x2x2([0, 1], [0, 1], [0, 1], [2, 3], [0, 1], [2, 3])
    
    # Part 3: j=0,1; k=4 (2x2x1 schoolbook, rank 4)
    for i in range(2):
        for j in range(2):
            u = {i*m+j: 1}
            v = {j*p+4: 1}
            w = {4*n+i: 1}
            terms.append((u, v, w))
    
    # Part 4: j=2; k=0..4 (2x1x5 schoolbook, rank 10)
    for i in range(2):
        for k in range(5):
            u = {i*m+2: 1}
            v = {2*p+k: 1}
            w = {k*n+i: 1}
            terms.append((u, v, w))
    
    # Build matrices
    rank = len(terms)
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for c, (u_dict, v_dict, w_dict) in enumerate(terms):
        for idx, coeff in u_dict.items():
            U[idx][c] += coeff
        for idx, coeff in v_dict.items():
            V[idx][c] += coeff
        for idx, coeff in w_dict.items():
            W[idx][c] += coeff
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
