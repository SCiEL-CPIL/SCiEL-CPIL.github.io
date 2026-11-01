"""Rank-minimal exact tensor decomposition for 2x2 * 2x6 matrix multiplication."""

def construct():
    """
    Construct a rank-21 decomposition of the matrix multiplication tensor
    for C = A * B where A is 2x2 and B is 2x6.
    
    The decomposition uses Strassen's algorithm on 3 blocks of 2x2 matrices
    from B. For each block, Strassen reduces the 8 scalar multiplications
    to 7, giving 21 total rank factors.
    
    W convention: W[k*n+i, r] where C[i,k] = sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r]
    """
    n, m, p = 2, 2, 6
    rank = 21
    
    U = [[0] * rank for _ in range(n * m)]  # 4 x 21
    V = [[0] * rank for _ in range(m * p)]  # 12 x 21
    W = [[0] * rank for _ in range(p * n)]  # 12 x 21
    
    # For each of the 3 blocks (l = 0, 1, 2), apply Strassen's 7 multiplications
    # Block l uses B columns (2l, 2l+1) and C columns (2l, 2l+1)
    
    for l in range(3):
        base = 7 * l  # Starting column index for this block in U, V, W
        
        # V indices for block l:
        # b11 -> V[0, 2l]   (A row 0, B col 0 of block)
        # b12 -> V[0, 2l+1]
        # b21 -> V[6, 2l]   (A row 1, B col 0 of block)  
        # b22 -> V[6, 2l+1]
        
        # W indices for block l (C output):
        # c11 -> W[4l + 0]   (C row 0, col 2l)
        # c12 -> W[4l + 2]   (C row 0, col 2l+1)
        # c21 -> W[4l + 1]   (C row 1, col 2l)
        # c22 -> W[4l + 3]   (C row 1, col 2l+1)
        
        # Strassen's 7 products for block l:
        # M1 = (a11+a22)*(b11+b22)
        # M2 = (a21+a22)*b11
        # M3 = a11*(b12-b22)
        # M4 = a22*(b21-b11)
        # M5 = (a11+a12)*b22
        # M6 = (a21-a11)*(b11+b12)
        # M7 = (a12-a22)*(b21+b22)
        
        # c11 = M1 - M4 + M6 + M7
        # c12 = M3 + M4
        # c21 = M2 + M4
        # c22 = M1 - M3 + M5
        
        # M1: U factors for a11 and a22, V factors for b11 and b22
        # a11 -> U[0, base+0], a22 -> U[3, base+0]
        U[0][base + 0] = 1
        U[3][base + 0] = 1
        V[0 * 6 + 2 * l][base + 0] = 1
        V[1 * 6 + 2 * l + 1][base + 0] = 1
        W[4 * l + 0][base + 0] = 1   # c11
        W[4 * l + 3][base + 0] = 1   # c22
        
        # M2: U factor for a21+a22, V factor for b11
        U[1][base + 1] = 1  # a21
        U[3][base + 1] = 1  # a22
        V[0 * 6 + 2 * l][base + 1] = 1
        W[4 * l + 1][base + 1] = 1  # c21
        
        # M3: U factor for a11, V factor for b12-b22
        U[0][base + 2] = 1
        V[0 * 6 + 2 * l + 1][base + 2] = 1
        V[1 * 6 + 2 * l + 1][base + 2] = -1
        W[4 * l + 2][base + 2] = 1   # c12
        W[4 * l + 3][base + 2] = -1  # c22
        
        # M4: U factor for a22, V factor for b21-b11
        U[3][base + 3] = 1
        V[1 * 6 + 2 * l][base + 3] = 1
        V[0 * 6 + 2 * l][base + 3] = -1
        W[4 * l + 0][base + 3] = -1  # c11
        W[4 * l + 2][base + 3] = 1   # c12
        W[4 * l + 1][base + 3] = 1   # c21
        
        # M5: U factor for a11+a12, V factor for b22
        U[0][base + 4] = 1  # a11
        U[2][base + 4] = 1  # a12
        V[1 * 6 + 2 * l + 1][base + 4] = 1
        W[4 * l + 3][base + 4] = 1  # c22
        
        # M6: U factor for a21-a11, V factor for b11+b12
        U[1][base + 5] = 1   # a21
        U[0][base + 5] = -1  # a11
        V[0 * 6 + 2 * l][base + 5] = 1
        V[0 * 6 + 2 * l + 1][base + 5] = 1
        W[4 * l + 0][base + 5] = 1  # c11
        
        # M7: U factor for a12-a22, V factor for b21+b22
        U[2][base + 6] = 1   # a12
        U[3][base + 6] = -1  # a22
        V[1 * 6 + 2 * l][base + 6] = 1
        V[1 * 6 + 2 * l + 1][base + 6] = 1
        W[4 * l + 0][base + 6] = 1  # c11
    
    return {"U": U, "V": V, "W": W}
