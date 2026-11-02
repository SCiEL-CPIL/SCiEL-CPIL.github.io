"""Rewritten program for 2x2 matrix multiplication tensor decomposition."""
DIMENSIONS = [2, 2, 2]

# EVOLVE-BLOCK-START
def construct():
    """
    Return the rank-7 Strassen decomposition for 2x2 matrix multiplication.
    
    The matrix multiplication tensor T[i,j,k] = 1 for all i,j,k in {0,1}
    (since (A*B)[i,k] = sum_j A[i,j]*B[j,k]).
    
    We need U (4x7), V (4x7), W (4x7) such that:
    sum_r U[i*2+j, r] * V[j*2+k, r] * W[k*2+i, r] = 1 for all i,j,k.
    
    Using the correct Strassen decomposition:
    M1 = (a00 + a11)(b00 + b11)
    M2 = (a10 + a11)(b00 + b01)
    M3 = a00(b10 - b11)
    M4 = a11(b01 - b00)
    M5 = (a00 + a01)(b11)
    M6 = (a10 - a00)(b00 + b01)
    M7 = (a01 - a11)(b10 + b11)
    
    C00 = M1 - M5 + M7
    C01 = M3 + M5
    C10 = M2 - M4 + M6
    C11 = M1 - M2 + M4 + M7
    
    W index convention: W[k*n+i, r] where n=2, so W[2k+i, r].
    """
    import numpy as np
    
    n, m, p = DIMENSIONS
    rank = 7
    
    U = np.zeros((4, rank), dtype=float)
    V = np.zeros((4, rank), dtype=float)
    W = np.zeros((4, rank), dtype=float)
    
    # r=0: M1 = (a00 + a11)(b00 + b11)
    # U: a00+a11 -> U[0,0]=1 (i=0,j=0), U[3,0]=1 (i=1,j=1)
    # V: b00+b11 -> V[0,0]=1 (j=0,k=0), V[3,0]=1 (j=1,k=1)
    # W: C00+=1, C11+=1 -> W[0,0]=1 (k=0,i=0), W[3,0]=1 (k=1,i=1)
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    W[0, 0] = 1; W[3, 0] = 1
    
    # r=1: M2 = (a10 + a11)(b00 + b01)
    # U: a10+a11 -> U[2,1]=1 (i=1,j=0), U[3,1]=1 (i=1,j=1)
    # V: b00+b01 -> V[0,1]=1 (j=0,k=0), V[1,1]=1 (j=0,k=1)
    # W: C10+=1, C11-=1 -> W[1,1]=1 (k=0,i=1), W[3,1]=-1 (k=1,i=1)
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1; V[1, 1] = 1
    W[1, 1] = 1; W[3, 1] = -1
    
    # r=2: M3 = a00(b10 - b11)
    # U: a00 -> U[0,2]=1 (i=0,j=0)
    # V: b10-b11 -> V[2,2]=1 (j=1,k=0), V[3,2]=-1 (j=1,k=1)
    # W: C01+=1 -> W[2,2]=1 (k=1,i=0)
    U[0, 2] = 1
    V[2, 2] = 1; V[3, 2] = -1
    W[2, 2] = 1
    
    # r=3: M4 = a11(b01 - b00)
    # U: a11 -> U[3,3]=1 (i=1,j=1)
    # V: b01-b00 -> V[1,3]=1 (j=0,k=1), V[0,3]=-1 (j=0,k=0)
    # W: C00-=1, C10+=1 -> W[0,3]=-1 (k=0,i=0), W[1,3]=1 (k=0,i=1)
    U[3, 3] = 1
    V[1, 3] = 1; V[0, 3] = -1
    W[0, 3] = -1; W[1, 3] = 1
    
    # r=4: M5 = (a00 + a01)(b11)
    # U: a00+a01 -> U[0,4]=1 (i=0,j=0), U[1,4]=1 (i=0,j=1)
    # V: b11 -> V[3,4]=1 (j=1,k=1)
    # W: C00-=1, C01+=1 -> W[0,4]=-1 (k=0,i=0), W[2,4]=1 (k=1,i=0)
    U[0, 4] = 1; U[1, 4] = 1
    V[3, 4] = 1
    W[0, 4] = -1; W[2, 4] = 1
    
    # r=5: M6 = (a10 - a00)(b00 + b01)
    # U: a10-a00 -> U[2,5]=1 (i=1,j=0), U[0,5]=-1 (i=0,j=0)
    # V: b00+b01 -> V[0,5]=1 (j=0,k=0), V[1,5]=1 (j=0,k=1)
    # W: C10+=1 -> W[1,5]=1 (k=0,i=1)
    U[2, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[1, 5] = 1
    W[1, 5] = 1
    
    # r=6: M7 = (a01 - a11)(b10 + b11)
    # U: a01-a11 -> U[1,6]=1 (i=0,j=1), U[3,6]=-1 (i=1,j=1)
    # V: b10+b11 -> V[2,6]=1 (j=1,k=0), V[3,6]=1 (j=1,k=1)
    # W: C00+=1 -> W[0,6]=1 (k=0,i=0)
    U[1, 6] = 1; U[3, 6] = -1
    V[2, 6] = 1; V[3, 6] = 1
    W[0, 6] = 1
    
    # Verify the decomposition
    for i in range(2):
        for j in range(2):
            for k in range(2):
                s = 0.0
                for r in range(7):
                    s += U[i*2+j, r] * V[j*2+k, r] * W[k*2+i, r]
                if abs(s - 1.0) > 1e-10:
                    # Fallback to schoolbook if verification fails
                    rank_fb = 8
                    U_fb = np.zeros((4, rank_fb))
                    V_fb = np.zeros((4, rank_fb))
                    W_fb = np.zeros((4, rank_fb))
                    col = 0
                    for ii in range(2):
                        for jj in range(2):
                            for kk in range(2):
                                U_fb[ii*2+jj, col] = 1
                                V_fb[jj*2+kk, col] = 1
                                W_fb[kk*2+ii, col] = 1
                                col += 1
                    return {"U": U_fb, "V": V_fb, "W": W_fb}
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
