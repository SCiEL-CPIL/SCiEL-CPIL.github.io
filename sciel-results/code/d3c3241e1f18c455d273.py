import numpy as np

def construct():
    n, m, p = 2, 5, 5
    # We'll use rank 16: 7 (Strassen for cols 0-1) + 7 (Strassen for cols 2-3) + 2 (direct for col 4)
    rank = 16
    U = np.zeros((n*m, rank))  # (10, 16)
    V = np.zeros((m*p, rank))  # (25, 16)
    W = np.zeros((p*n, rank))  # (10, 16)
    
    # Strassen 2x2x2 terms for a block with A rows i0,i1 and A cols j0,j0+1, B rows j0,j0+1 and B cols k0,k0+1
    # U indices: i*m+j
    # V indices: j*p+k
    # W indices: k*n+i
    
    def add_strassen(U, V, W, i0, j0, k0, col):
        # P1 = (A[i0,j0]+A[i1,j1])(B[j0,k0]+B[j1,k1])
        U[i0*m+j0, col] += 1
        U[i1*m+j1, col] += 1  # i1 = i0+1, j1 = j0+1
        V[j0*p+k0, col] += 1
        V[j1*p+k1, col] += 1
        W[k0*n+i0, col] += 1
        W[k1*n+i1, col] += 1
        
        # P2 = (A[i1,j0]+A[i1,j1])B[j0,k0]
        U[i1*m+j0, col+1] += 1
        U[i1*m+j1, col+1] += 1
        V[j0*p+k0, col+1] += 1
        W[k0*n+i1, col+1] += 1
        
        # P3 = A[i0,j0](B[j1,k0]-B[j1,k1])
        U[i0*m+j0, col+2] += 1
        V[j1*p+k0, col+2] += 1
        V[j1*p+k1, col+2] += -1
        W[k1*n+i0, col+2] += 1
        
        # P4 = A[i1,j1](B[j0,k1]-B[j0,k0])
        U[i1*m+j1, col+3] += 1
        V[j0*p+k1, col+3] += 1
        V[j0*p+k0, col+3] += -1
        W[k0*n+i0, col+3] += 1
        W[k0*n+i1, col+3] += 1
        
        # P5 = (A[i0,j0]+A[i0,j1])B[j1,k1]
        U[i0*m+j0, col+4] += 1
        U[i0*m+j1, col+4] += 1
        V[j1*p+k1, col+4] += 1
        W[k0*n+i0, col+4] += -1
        W[k1*n+i0, col+4] += 1
        
        # P6 = (A[i1,j0]-A[i0,j0])(B[j0,k0]+B[j0,k1])
        U[i1*m+j0, col+5] += 1
        U[i0*m+j0, col+5] += -1
        V[j0*p+k0, col+5] += 1
        V[j0*p+k1, col+5] += 1
        W[k1*n+i1, col+5] += 1
        
        # P7 = (A[i0,j1]-A[i1,j1])(B[j1,k0]+B[j1,k1])
        U[i0*m+j1, col+6] += 1
        U[i1*m+j1, col+6] += -1
        V[j1*p+k0, col+6] += 1
        V[j1*p+k1, col+6] += 1
        W[k0*n+i0, col+6] += 1
    
    # Block 1: i0=0, j0=0, k0=0 (A rows 0-1, A cols 0-1, B rows 0-1, B cols 0-1)
    add_strassen(U, V, W, 0, 0, 0, 0)
    
    # Block 2: i0=0, j0=2, k0=2 (A rows 0-1, A cols 2-3, B rows 2-3, B cols 2-3)
    add_strassen(U, V, W, 0, 2, 2, 7)
    
    # Block 3: direct for col 4
    # C[0,4] = A[0,4]*B[4,4], C[1,4] = A[1,4]*B[4,4]
    U[0*m+4, 14] = 1
    V[4*p+4, 14] = 1
    W[4*n+0, 14] = 1
    
    U[1*m+4, 15] = 1
    V[4*p+4, 15] = 1
    W[4*n+1, 15] = 1
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
