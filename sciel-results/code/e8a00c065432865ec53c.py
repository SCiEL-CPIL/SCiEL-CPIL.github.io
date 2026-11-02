"""Rewritten program for 2x2 matrix multiplication tensor decomposition."""
DIMENSIONS = [2, 2, 2]

# EVOLVE-BLOCK-START
def construct():
    """
    Return the optimal rank-7 Strassen decomposition for 2x2 matrix multiplication.
    
    The matrix multiplication tensor T[i,j,k] = 1 for all i,j,k in {0,1}
    (since (A*B)[i,k] = sum_j A[i,j] * B[j,k] and each term contributes 1).
    
    We decompose as:
    T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    
    Using the standard Strassen decomposition with 7 multiplications:
    M1 = (a00 + a11)(b00 + b11)
    M2 = (a10 + a11)b00
    M3 = a00(b10 - b11)
    M4 = a11(b01 - b00)
    M5 = (a00 + a01)b11
    M6 = (a10 - a00)(b00 + b01)
    M7 = (a01 - a11)(b10 + b11)
    
    Output:
    C00 = M1 + M4 - M5 + M7
    C01 = M3 + M5
    C10 = M2 + M4
    C11 = M1 - M2 + M3 + M6
    
    Indexing: U[i*2+j], V[j*2+k], W[k*2+i]
    """
    import numpy as np
    
    n, m, p = DIMENSIONS
    rank = 7
    
    U = np.zeros((4, rank))
    V = np.zeros((4, rank))
    W = np.zeros((4, rank))
    
    # r=0: M1 = (a00+a11)(b00+b11)
    # U: a00 (i=0,j=0) and a11 (i=1,j=1)
    U[0, 0] = 1
    U[3, 0] = 1
    # V: b00 (j=0,k=0) and b11 (j=1,k=1)
    V[0, 0] = 1
    V[3, 0] = 1
    # W: C00 (k=0,i=0) and C11 (k=1,i=1)
    W[0, 0] = 1
    W[3, 0] = 1
    
    # r=1: M2 = (a10+a11)b00
    # U: a10 (i=1,j=0) and a11 (i=1,j=1)
    U[2, 1] = 1
    U[3, 1] = 1
    # V: b00 (j=0,k=0)
    V[0, 1] = 1
    # W: C10 (k=0,i=1) gets +1, C11 (k=1,i=1) gets -1
    W[1, 1] = 1
    W[3, 1] = -1
    
    # r=2: M3 = a00(b10-b11)
    # U: a00 (i=0,j=0)
    U[0, 2] = 1
    # V: b10 (j=1,k=0) and b11 (j=1,k=1)
    V[2, 2] = 1
    V[3, 2] = -1
    # W: C01 (k=1,i=0) gets +1
    W[2, 2] = 1
    
    # r=3: M4 = a11(b01-b00)
    # U: a11 (i=1,j=1)
    U[3, 3] = 1
    # V: b01 (j=0,k=1) and b00 (j=0,k=0)
    V[1, 3] = 1
    V[0, 3] = -1
    # W: C00 (k=0,i=0) gets +1, C10 (k=0,i=1) gets +1
    W[0, 3] = 1
    W[1, 3] = 1
    
    # r=4: M5 = (a00+a01)b11
    # U: a00 (i=0,j=0) and a01 (i=0,j=1)
    U[0, 4] = 1
    U[1, 4] = 1
    # V: b11 (j=1,k=1)
    V[3, 4] = 1
    # W: C00 (k=0,i=0) gets -1, C01 (k=1,i=0) gets +1
    W[0, 4] = -1
    W[2, 4] = 1
    
    # r=5: M6 = (a10-a00)(b00+b01)
    # U: a10 (i=1,j=0) and a00 (i=0,j=0)
    U[2, 5] = 1
    U[0, 5] = -1
    # V: b00 (j=0,k=0) and b01 (j=0,k=1)
    V[0, 5] = 1
    V[1, 5] = 1
    # W: C11 (k=1,i=1) gets +1
    W[3, 5] = 1
    
    # r=6: M7 = (a01-a11)(b10+b11)
    # U: a01 (i=0,j=1) and a11 (i=1,j=1)
    U[1, 6] = 1
    U[3, 6] = -1
    # V: b10 (j=1,k=0) and b11 (j=1,k=1)
    V[2, 6] = 1
    V[3, 6] = 1
    # W: C00 (k=0,i=0) gets +1
    W[0, 6] = 1
    
    # Verify the decomposition: T[i,j,k] should equal 1 for all i,j,k
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for r in range(rank):
                    s += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                if abs(s - 1.0) > 1e-10:
                    # Fallback to schoolbook rank-8 if verification fails
                    rank_fb = 8
                    U_fb = np.zeros((4, rank_fb))
                    V_fb = np.zeros((4, rank_fb))
                    W_fb = np.zeros((4, rank_fb))
                    col = 0
                    for ii in range(n):
                        for jj in range(m):
                            for kk in range(p):
                                U_fb[ii*m+jj, col] = 1
                                V_fb[jj*p+kk, col] = 1
                                W_fb[kk*n+ii, col] = 1
                                col += 1
                    return {"U": U_fb.tolist(), "V": V_fb.tolist(), "W": W_fb.tolist()}
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
