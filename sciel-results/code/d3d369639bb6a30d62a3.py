"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 5]

# EVOLVE-BLOCK-START
def construct():
    """
    Constructs a tensor decomposition for multiplying a 2x3 matrix by a 3x5 matrix.
    
    Uses Strassen's algorithm (rank 7) for 2x2x2 blocks and schoolbook for the rest.
    Total rank = 7 + 7 + 4 + 10 = 28.
    
    The indices follow the convention:
    A is n x m (2x3), B is m x p (3x5), C is n x p (2x5).
    U has shape (n*m, rank) = (6, rank)
    V has shape (m*p, rank) = (15, rank)
    W has shape (p*n, rank) = (10, rank)
    
    U[i*m + j, c] corresponds to coefficient for A[i,j]
    V[j*p + k, c] corresponds to coefficient for B[j,k]
    W[k*n + i, c] corresponds to coefficient for C[i,k]
    """
    n, m, p = DIMENSIONS
    
    # We will build lists of (u_dict, v_dict, w_dict) tuples
    # where each dict maps index -> coefficient
    terms = []
    
    def add_term(u_dict, v_dict, w_dict):
        # Filter out zero coefficients
        u_dict = {k: v for k, v in u_dict.items() if v != 0}
        v_dict = {k: v for k, v in v_dict.items() if v != 0}
        w_dict = {k: v for k, v in w_dict.items() if v != 0}
        if u_dict and v_dict and w_dict:
            terms.append((u_dict, v_dict, w_dict))
    
    def strassen_2x2x2(i1, i2, j1, j2, k1, k2):
        """
        Adds 7 Strassen terms for multiplying a 2x2 block of A (rows i1,i2; cols j1,j2)
        by a 2x2 block of B (rows j1,j2; cols k1,k2) to produce a 2x2 block of C (rows i1,i2; cols k1,k2).
        
        The 7 multiplications are:
        M1 = (a[i1,j1] + a[i2,j2]) * (b[j1,k1] + b[j2,k2])
        M2 = (a[i2,j1] + a[i2,j2]) * b[j1,k1]
        M3 = a[i1,j1] * (b[j1,k2] - b[j2,k2])
        M4 = a[i2,j2] * (b[j2,k1] - b[j1,k1])
        M5 = (a[i1,j1] + a[i1,j2]) * b[j2,k2]
        M6 = (a[i2,j1] - a[i1,j1]) * (b[j1,k1] + b[j1,k2])
        M7 = (a[i1,j2] - a[i2,j2]) * (b[j2,k1] + b[j2,k2])
        
        Results:
        C[i1,k1] = M1 + M4 - M5 + M7
        C[i1,k2] = M3 + M5
        C[i2,k1] = M2 + M4
        C[i2,k2] = M1 - M2 + M3 + M6
        """
        # Term 1: M1 = (a[i1,j1] + a[i2,j2]) * (b[j1,k1] + b[j2,k2])
        # Contributes: C[i1,k1] += 1, C[i2,k2] += -1
        u1 = {i1*m+j1: 1, i2*m+j2: 1}
        v1 = {j1*p+k1: 1, j2*p+k2: 1}
        w1 = {k1*n+i1: 1, k2*n+i2: -1}
        add_term(u1, v1, w1)
        
        # Term 2: M2 = (a[i2,j1] + a[i2,j2]) * b[j1,k1]
        # Contributes: C[i2,k1] += 1, C[i2,k2] += -1
        u2 = {i2*m+j1: 1, i2*m+j2: 1}
        v2 = {j1*p+k1: 1}
        w2 = {k1*n+i2: 1, k2*n+i2: -1}
        add_term(u2, v2, w2)
        
        # Term 3: M3 = a[i1,j1] * (b[j1,k2] - b[j2,k2])
        # Contributes: C[i1,k2] += 1, C[i2,k2] += 1
        u3 = {i1*m+j1: 1}
        v3 = {j1*p+k2: 1, j2*p+k2: -1}
        w3 = {k2*n+i1: 1, k2*n+i2: 1}
        add_term(u3, v3, w3)
        
        # Term 4: M4 = a[i2,j2] * (b[j2,k1] - b[j1,k1])
        # Contributes: C[i1,k1] += 1, C[i2,k1] += 1
        u4 = {i2*m+j2: 1}
        v4 = {j2*p+k1: 1, j1*p+k1: -1}
        w4 = {k1*n+i1: 1, k1*n+i2: 1}
        add_term(u4, v4, w4)
        
        # Term 5: M5 = (a[i1,j1] + a[i1,j2]) * b[j2,k2]
        # Contributes: C[i1,k1] += -1, C[i1,k2] += 1
        u5 = {i1*m+j1: 1, i1*m+j2: 1}
        v5 = {j2*p+k2: 1}
        w5 = {k1*n+i1: -1, k2*n+i1: 1}
        add_term(u5, v5, w5)
        
        # Term 6: M6 = (a[i2,j1] - a[i1,j1]) * (b[j1,k1] + b[j1,k2])
        # Contributes: C[i2,k2] += 1
        u6 = {i2*m+j1: 1, i1*m+j1: -1}
        v6 = {j1*p+k1: 1, j1*p+k2: 1}
        w6 = {k2*n+i2: 1}
        add_term(u6, v6, w6)
        
        # Term 7: M7 = (a[i1,j2] - a[i2,j2]) * (b[j2,k1] + b[j2,k2])
        # Contributes: C[i1,k1] += 1
        u7 = {i1*m+j2: 1, i2*m+j2: -1}
        v7 = {j2*p+k1: 1, j2*p+k2: 1}
        w7 = {k1*n+i1: 1}
        add_term(u7, v7, w7)
    
    # Build all terms
    all_terms = []
    
    # Block 1: A[0:2,0:2] * B[0:2,0:2] -> C[0:2,0:2]
    # i1=0, i2=1, j1=0, j2=1, k1=0, k2=1
    strassen_2x2x2(0, 1, 0, 1, 0, 1)
    
    # Block 2: A[0:2,0:2] * B[0:2,2:4] -> C[0:2,2:4]
    # i1=0, i2=1, j1=0, j2=1, k1=2, k2=3
    strassen_2x2x2(0, 1, 0, 1, 2, 3)
    
    # Block 3: A[0:2,0:2] * B[0:2,4:5] -> C[0:2,4:5]
    # This is 2x2x1, schoolbook: 4 terms
    # C[0,4] = A[0,0]*B[0,4] + A[0,1]*B[1,4]
    # C[1,4] = A[1,0]*B[0,4] + A[1,1]*B[1,4]
    for i in range(2):
        for j in range(2):
            u = {i*m+j: 1}
            v = {j*p+4: 1}
            w = {4*n+i: 1}
            add_term(u, v, w)
    
    # Block 4: A[0:2,2:3] * B[2:3,0:5] -> C[0:2,0:5]
    # This is 2x1x5, schoolbook: 10 terms
    # C[i,k] = A[i,2]*B[2,k]
    for i in range(2):
        for k in range(5):
            u = {i*m+2: 1}
            v = {2*p+k: 1}
            w = {k*n+i: 1}
            add_term(u, v, w)
    
    # Now build U, V, W matrices
    rank = len(terms)
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for c, (u_dict, v_dict, w_dict) in enumerate(terms):
        for idx, val in u_dict.items():
            U[idx][c] = val
        for idx, val in v_dict.items():
            V[idx][c] = val
        for idx, val in w_dict.items():
            W[idx][c] = val
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
