import numpy as np

def construct():
    n, m, p = 2, 3, 6
    # We'll build the decomposition with rank 33
    # 3 k-groups, each with 7 Strassen + 4 direct = 11 terms
    rank = 33
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    col = 0
    for g in range(3):
        k0 = 2*g
        k1 = 2*g + 1
        
        # Strassen M1 = (A00+A11)(B0k0+B1k1)
        U[0, col] = 1  # A[0,0]
        U[4, col] = 1  # A[1,1]
        V[k0, col] = 1  # B[0,k0]
        V[6+k1, col] = 1  # B[1,k1]
        W[2*k0, col] = 1  # C[0,k0]
        W[2*k1+1, col] = 1  # C[1,k1]
        col += 1
        
        # Strassen M2 = (A10+A11)B0k0
        U[3, col] = 1  # A[1,0]
        U[4, col] = 1  # A[1,1]
        V[k0, col] = 1  # B[0,k0]
        W[2*k0+1, col] = 1  # C[1,k0]
        W[2*k1+1, col] = -1  # C[1,k1]
        col += 1
        
        # Strassen M3 = A00(B0k1-B1k1)
        U[0, col] = 1  # A[0,0]
        V[6+k1, col] = -1  # -B[1,k1]
        V[k1, col] = 1  # B[0,k1]
        W[2*k1, col] = 1  # C[0,k1]
        W[2*k1+1, col] = 1  # C[1,k1]
        col += 1
        
        # Strassen M4 = A11(B1k0-B0k0)
        U[4, col] = 1  # A[1,1]
        V[6+k0, col] = 1  # B[1,k0]
        V[k0, col] = -1  # -B[0,k0]
        W[2*k0, col] = 1  # C[0,k0]
        W[2*k0+1, col] = 1  # C[1,k0]
        col += 1
        
        # Strassen M5 = (A00+A01)B1k1
        U[0, col] = 1  # A[0,0]
        U[1, col] = 1  # A[0,1]
        V[6+k1, col] = 1  # B[1,k1]
        W[2*k0, col] = -1  # C[0,k0]
        W[2*k1, col] = 1  # C[0,k1]
        col += 1
        
        # Strassen M6 = (A10-A00)(B0k0+B1k0)
        U[3, col] = 1  # A[1,0]
        U[0, col] = -1  # -A[0,0]
        V[k0, col] = 1  # B[0,k0]
        V[6+k0, col] = 1  # B[1,k0]
        W[2*k1+1, col] = 1  # C[1,k1]
        col += 1
        
        # Strassen M7 = (A01-A11)(B0k1+B1k1)
        U[1, col] = 1  # A[0,1]
        U[4, col] = -1  # -A[1,1]
        V[k1, col] = 1  # B[0,k1]
        V[6+k1, col] = 1  # B[1,k1]
        W[2*k0, col] = 1  # C[0,k0]
        col += 1
        
        # Direct P1 = A02*B2k0
        U[2, col] = 1  # A[0,2]
        V[12+k0, col] = 1  # B[2,k0]
        W[2*k0, col] = 1  # C[0,k0]
        col += 1
        
        # Direct P2 = A02*B2k1
        U[2, col] = 1  # A[0,2]
        V[12+k1, col] = 1  # B[2,k1]
        W[2*k1, col] = 1  # C[0,k1]
        col += 1
        
        # Direct P3 = A12*B2k0
        U[5, col] = 1  # A[1,2]
        V[12+k0, col] = 1  # B[2,k0]
        W[2*k0+1, col] = 1  # C[1,k0]
        col += 1
        
        # Direct P4 = A12*B2k1
        U[5, col] = 1  # A[1,2]
        V[12+k1, col] = 1  # B[2,k1]
        W[2*k1+1, col] = 1  # C[1,k1]
        col += 1
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
