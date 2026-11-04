import time
import numpy as np

DIMENSIONS = [4, 4, 8]

_START_TIME = time.time()
_DEADLINE = _START_TIME + 145

def construct():
    n, m, p = DIMENSIONS
    
    # Correct Strassen decomposition for 2x2x2
    # Using the standard formula, carefully assigning indices
    U1 = np.zeros((4, 7), dtype=float)
    V1 = np.zeros((4, 7), dtype=float)
    W1 = np.zeros((4, 7), dtype=float)
    
    # M1 = (A00+A11)(B00+B11)
    # A: (0,0)->U[0], (1,1)->U[3]; B: (0,0)->V[0], (1,1)->V[3]
    # C: all four positions get +1
    U1[0,0]=1; U1[3,0]=1
    V1[0,0]=1; V1[3,0]=1
    W1[0,0]=1; W1[1,0]=1; W1[2,0]=1; W1[3,0]=1
    
    # M2 = (A10+A11)B00
    # A: (1,0)->U[2], (1,1)->U[3]; B: (0,0)->V[0]
    # C: (1,0)->+1, (1,1)->-1
    U1[2,1]=1; U1[3,1]=1
    V1[0,1]=1
    W1[2,1]=1; W1[3,1]=-1
    
    # M3 = A00(B01-B11)
    # A: (0,0)->U[0]; B: (0,1)->V[1], (1,1)->V[3]
    # C: (0,1)->+1, (1,1)->+1
    U1[0,2]=1
    V1[1,2]=1; V1[3,2]=-1
    W1[1,2]=1; W1[3,2]=1
    
    # M4 = A11(B10-B00)
    # A: (1,1)->U[3]; B: (1,0)->V[2], (0,0)->V[0]
    # C: (0,0)->+1, (1,0)->+1
    U1[3,3]=1
    V1[2,3]=1; V1[0,3]=-1
    W1[0,3]=1; W1[2,3]=1
    
    # M5 = (A00+A01)B11
    # A: (0,0)->U[0], (0,1)->U[1]; B: (1,1)->V[3]
    # C: (0,0)->-1, (0,1)->+1
    U1[0,4]=1; U1[1,4]=1
    V1[3,4]=1
    W1[0,4]=-1; W1[1,4]=1
    
    # M6 = (A10-A00)(B00+B01)
    # A: (1,0)->U[2], (0,0)->U[0]; B: (0,0)->V[0], (0,1)->V[1]
    # C: (1,1)->+1
    U1[2,5]=1; U1[0,5]=-1
    V1[0,5]=1; V1[1,5]=1
    W1[3,5]=1
    
    # M7 = (A01-A11)(B10+B11)
    # A: (0,1)->U[1], (1,1)->U[3]; B: (1,0)->V[2], (1,1)->V[3]
    # C: (0,0)->+1, (0,1)->+1
    U1[1,6]=1; U1[3,6]=-1
    V1[2,6]=1; V1[3,6]=1
    W1[0,6]=1; W1[1,6]=1
    
    # Verify Strassen
    max_err1 = 0
    for i in range(2):
        for j in range(2):
            for k in range(2):
                s = 0
                for r in range(7):
                    s += U1[i*2+j, r] * V1[j*2+k, r] * W1[k*2+i, r]
                max_err1 = max(max_err1, abs(s - 1.0))
    
    if max_err1 > 1e-9:
        # Strassen is wrong, use schoolbook for 2x2x2
        U1 = np.zeros((4, 8), dtype=float)
        V1 = np.zeros((4, 8), dtype=float)
        W1 = np.zeros((4, 8), dtype=float)
        col = 0
        for i in range(2):
            for j in range(2):
                for k in range(2):
                    U1[i*2+j, col] = 1
                    V1[j*2+k, col] = 1
                    W1[k*2+i, col] = 1
                    col += 1
    
    # Schoolbook for 2x2x4
    U2 = np.zeros((4, 8), dtype=float)
    V2 = np.zeros((8, 8), dtype=float)
    W2 = np.zeros((8, 8), dtype=float)
    col = 0
    for i in range(2):
        for j in range(2):
            for k in range(4):
                U2[i*2+j, col] = 1
                V2[j*4+k, col] = 1
                W2[k*2+i, col] = 1
                col += 1
    
    # Kronecker product
    r1 = U1.shape[1]
    r2 = U2.shape[1]
    U = np.kron(U1, U2)
    V = np.kron(V1, V2)
    W = np.kron(W1, W2)
    
    # Verify final
    max_err = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = np.sum(U[i*m+j, :] * V[j*p+k, :] * W[k*n+i, :])
                max_err = max(max_err, abs(s - 1.0))
    
    if max_err > 1e-9:
        # Full schoolbook fallback
        rank_sb = n * m * p
        U_sb = np.zeros((n*m, rank_sb))
        V_sb = np.zeros((m*p, rank_sb))
        W_sb = np.zeros((p*n, rank_sb))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_sb[i*m+j, col] = 1
                    V_sb[j*p+k, col] = 1
                    W_sb[k*n+i, col] = 1
                    col += 1
        return {"U": U_sb.tolist(), "V": V_sb.tolist(), "W": W_sb.tolist()}
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist}
