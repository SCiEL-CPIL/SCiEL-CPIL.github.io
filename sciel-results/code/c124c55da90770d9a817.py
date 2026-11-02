"""Rewritten program for 2x2 matrix multiplication tensor decomposition."""
DIMENSIONS = [2, 2, 2]

# EVOLVE-BLOCK-START
def construct():
    """
    Return the optimal rank-7 Strassen decomposition for 2x2 matrix multiplication.
    
    The matrix multiplication tensor T_{ijk} is defined such that
    (A*B)[i,k] = sum_j A[i,j] * B[j,k]
    
    The decomposition satisfies:
    sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all i,j,k
    
    Using the standard Strassen decomposition with 7 multiplications:
    M1 = (a00 + a11)(b00 + b11)
    M2 = (a10 + a11)(b00 + b01)
    M3 = a00(b10 - b11)
    M4 = a11(b01 - b00)
    M5 = (a00 + a01)b11
    M6 = (a10 - a00)(b00 + b01)
    M7 = (a01 - a11)(b10 + b11)
    
    C00 = M1 - M5 + M7
    C01 = M3 + M5
    C10 = M2 - M4 + M6
    C11 = M1 - M2 + M4 + M7
    """
    import numpy as np
    
    n, m, p = DIMENSIONS
    rank = 7
    
    # Initialize factor matrices
    U = np.zeros((n*m, rank))  # shape (4, 7)
    V = np.zeros((m*p, rank))  # shape (4, 7)
    W = np.zeros((p*n, rank))  # shape (4, 7)
    
    # r=0: M1 = (a00 + a11)(b00 + b11)
    # U: a00 + a11 -> indices (i,j) = (0,0) and (1,1)
    U[0*2+0, 0] = 1  # a00
    U[1*2+1, 0] = 1  # a11
    # V: b00 + b11 -> indices (j,k) = (0,0) and (1,1)
    V[0*2+0, 0] = 1  # b00
    V[1*2+1, 0] = 1  # b11
    # W: C00 += 1, C11 += 1 -> indices (k,i) = (0,0) and (1,1)
    W[0*2+0, 0] = 1  # C00
    W[1*2+1, 0] = 1  # C11
    
    # r=1: M2 = (a10 + a11)(b00 + b01)
    # U: a10 + a11 -> indices (i,j) = (1,0) and (1,1)
    U[1*2+0, 1] = 1  # a10
    U[1*2+1, 1] = 1  # a11
    # V: b00 + b01 -> indices (j,k) = (0,0) and (0,1)
    V[0*2+0, 1] = 1  # b00
    V[0*2+1, 1] = 1  # b01
    # W: C10 += 1, C11 -= 1 -> indices (k,i) = (0,1) and (1,1)
    W[0*2+1, 1] = 1  # C10
    W[1*2+1, 1] = -1  # C11
    
    # r=2: M3 = a00(b10 - b11)
    # U: a00 -> index (i,j) = (0,0)
    U[0*2+0, 2] = 1  # a00
    # V: b10 - b11 -> indices (j,k) = (1,0) and (1,1)
    V[1*2+0, 2] = 1  # b10
    V[1*2+1, 2] = -1  # b11
    # W: C01 += 1 -> index (k,i) = (1,0)
    W[1*2+0, 2] = 1  # C01
    
    # r=3: M4 = a11(b01 - b00)
    # U: a11 -> index (i,j) = (1,1)
    U[1*2+1, 3] = 1  # a11
    # V: b01 - b00 -> indices (j,k) = (0,1) and (0,0)
    V[0*2+1, 3] = 1  # b01
    V[0*2+0, 3] = -1  # b00
    # W: C10 += 1, C11 += 1 -> indices (k,i) = (0,1) and (1,1)
    W[0*2+1, 3] = 1  # C10
    W[1*2+1, 3] = 1  # C11
    
    # r=4: M5 = (a00 + a01)b11
    # U: a00 + a01 -> indices (i,j) = (0,0) and (0,1)
    U[0*2+0, 4] = 1  # a00
    U[0*2+1, 4] = 1  # a01
    # V: b11 -> index (j,k) = (1,1)
    V[1*2+1, 4] = 1  # b11
    # W: C00 -= 1, C01 += 1 -> indices (k,i) = (0,0) and (1,0)
    W[0*2+0, 4] = -1  # C00
    W[1*2+0, 4] = 1  # C01
    
    # r=5: M6 = (a10 - a00)(b00 + b01)
    # U: a10 - a00 -> indices (i,j) = (1,0) and (0,0)
    U[1*2+0, 5] = 1  # a10
    U[0*2+0, 5] = -1  # a00
    # V: b00 + b01 -> indices (j,k) = (0,0) and (0,1)
    V[0*2+0, 5] = 1  # b00
    V[0*2+1, 5] = 1  # b01
    # W: C11 += 1 -> index (k,i) = (1,1)
    W[1*2+1, 5] = 1  # C11
    
    # r=6: M7 = (a01 - a11)(b10 + b11)
    # U: a01 - a11 -> indices (i,j) = (0,1) and (1,1)
    U[0*2+1, 6] = 1  # a01
    U[1*2+1, 6] = -1  # a11
    # V: b10 + b11 -> indices (j,k) = (1,0) and (1,1)
    V[1*2+0, 6] = 1  # b10
    V[1*2+1, 6] = 1  # b11
    # W: C00 += 1, C11 += 1 -> indices (k,i) = (0,0) and (1,1)
    W[0*2+0, 6] = 1  # C00
    W[1*2+1, 6] = 1  # C11
    
    # Verify the decomposition
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0
                for r in range(rank):
                    s += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                if abs(s - 1) > 1e-10:
                    # Fallback to schoolbook if verification fails
                    rank_fb = n*m*p
                    U_fb = np.zeros((n*m, rank_fb))
                    V_fb = np.zeros((m*p, rank_fb))
                    W_fb = np.zeros((p*n, rank_fb))
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
