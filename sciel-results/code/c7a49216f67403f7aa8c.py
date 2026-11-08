import numpy as np
import time

DIMENSIONS = [2, 5, 6]
SEED = 42

def _strassen_222():
    """Return U, V, W for 2x2x2 matrix multiplication with rank 7 (Strassen)."""
    # Strassen's 7-multiplication algorithm for 2x2x2
    # U: (n*m, rank) = (4, 7)
    # V: (m*p, rank) = (4, 7)
    # W: (p*n, rank) = (4, 7)
    
    # The 7 rank-1 tensors for Strassen
    # M1 = (a11 - a22)(b21 + b22)
    # M2 = (a21 + a22) b11
    # M3 = a11 (b11 - b21)
    # M4 = a22 (b22 - b12)
    # M5 = (a11 + a12) b22
    # M6 = (a21 - a11)(b11 + b12)
    # M7 = (a12 - a22)(b21 + b22)
    
    # Indexing: A is 2x2, row i, col j -> index i*2+j
    # B is 2x2, row j, col k -> index j*2+k
    # C is 2x2, row i, col k -> index k*2+i  (trace convention: W[k*n+i, r])
    
    U = np.zeros((4, 7), dtype=float)
    V = np.zeros((4, 7), dtype=float)
    W = np.zeros((4, 7), dtype=float)
    
    # M1: (a11 - a22)(b21 + b22)
    # U indices: a11=0, a22=3
    U[0, 0] = 1; U[3, 0] = -1
    # V indices: b21=2, b22=3
    V[2, 0] = 1; V[3, 0] = 1
    # W: contributes to C
    # M1 = (a11-a22)(b21+b22)
    # This contributes to multiple C elements. In Strassen, the combination is:
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    # So W for M1: C11 gets +1, C22 gets +1
    # C11 index: k=0,i=0 -> 0*2+0 = 0
    # C22 index: k=1,i=1 -> 1*2+1 = 3
    W[0, 0] = 1; W[3, 0] = 1
    
    # M2: (a21 + a22) b11
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    # C21 = M2 + M4 -> C21 index: k=0,i=1 -> 0*2+1 = 1
    # C22 = M1 - M2 + M3 + M6 -> C22 index: 3
    W[1, 1] = 1; W[3, 1] = -1
    
    # M3: a11 (b11 - b21)
    U[0, 2] = 1
    V[0, 2] = 1; V[2, 2] = -1
    # C12 = M3 + M5 -> C12 index: k=1,i=0 -> 1*2+0 = 2
    # C22 = M1 - M2 + M3 + M6 -> C22 index: 3
    W[2, 2] = 1; W[3, 2] = 1
    
    # M4: a22 (b22 - b12)
    U[3, 3] = 1
    V[1, 3] = -1; V[3, 3] = 1
    # C11 = M1 + M4 - M5 + M7 -> C11 index: 0
    # C21 = M2 + M4 -> C21 index: 1
    W[0, 3] = 1; W[1, 3] = 1
    
    # M5: (a11 + a12) b22
    U[0, 4] = 1; U[1, 4] = 1
    V[3, 4] = 1
    # C11 = M1 + M4 - M5 + M7 -> C11 index: 0
    # C12 = M3 + M5 -> C12 index: 2
    W[0, 4] = -1; W[2, 4] = 1
    
    # M6: (a21 - a11)(b11 + b12)
    U[2, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[1, 5] = 1
    # C22 = M1 - M2 + M3 + M6 -> C22 index: 3
    W[3, 5] = 1
    
    # M7: (a12 - a22)(b21 + b22)
    U[1, 6] = 1; U[3, 6] = -1
    V[2, 6] = 1; V[3, 6] = 1
    # C11 = M1 + M4 - M5 + M7 -> C11 index: 0
    W[0, 6] = 1
    
    return U, V, W

def _outer_212():
    """Return U, V, W for 2x1x2 matrix multiplication with rank 4 (outer product)."""
    # A is 2x1, B is 1x2, C is 2x2
    # C = A * B = outer product
    # U: (n*m, rank) = (2*1, 4) = (2, 4)
    # V: (m*p, rank) = (1*2, 4) = (2, 4)
    # W: (p*n, rank) = (2*2, 4) = (4, 4)
    
    U = np.zeros((2, 4), dtype=float)
    V = np.zeros((2, 4), dtype=float)
    W = np.zeros((4, 4), dtype=float)
    
    # C[i,k] = A[i,0] * B[0,k]
    # For each (i,k) pair, one rank-1 term
    # i in {0,1}, k in {0,1}
    col = 0
    for i in range(2):
        for k in range(2):
            U[i, col] = 1
            V[k, col] = 1
            # W index: k*n + i = k*2 + i
            W[k*2 + i, col] = 1
            col += 1
    
    return U, V, W

def _block_combine(n1, n2, m, p, U1, V1, W1, U2, V2, W2):
    """Combine two block diagonal matrices to form a larger one.
    For 2x(m1+m2)x2, split into 2xm1x2 and 2xm2x2."""
    rank = U1.shape[1] + U2.shape[1]
    U = np.zeros((2*m, rank), dtype=float)
    V = np.zeros((m*p, rank), dtype=float)
    W = np.zeros((p*2, rank), dtype=float)
    
    # First block: rows 0..m1-1 of m dimension
    U[:2*n1, :U1.shape[1]] = U1
    V[:m1*p, :V1.shape[1]] = V1
    W[:p*n1, :W1.shape[1]] = W1
    
    # Second block: rows m1..m-1 of m dimension
    U[2*n1:, U1.shape[1]:] = U2
    V[m1*p:, U1.shape[1]:] = V2
    W[p*n1:, U1.shape[1]:] = W2
    
    return U, V, W

def _block_combine_p(n, m, p1, p2, U1, V1, W1, U2, V2, W2):
    """Combine two blocks in the p dimension.
    For 2x5x(p1+p2), split into 2x5xp1 and 2x5xp2."""
    rank = U1.shape[1] + U2.shape[1]
    U = np.zeros((n*m, rank), dtype=float)
    V = np.zeros((m*p, rank), dtype=float)
    W = np.zeros((p*n, rank), dtype=float)
    
    # First block: cols 0..p1-1
    U[:, :U1.shape[1]] = U1
    V[:, :m*p1] = V1
    W[:, :p1*n] = W1
    
    # Second block: cols p1..p-1
    U[:, U1.shape[1]:] = U2
    V[:, m*p1:] = V2
    W[:, p1*n:] = W2
    
    return U, V, W

def construct():
    deadline = time.time() + 100  # Safety margin
    
    # Step 1: 2x2x2 with rank 7 (Strassen)
    U_222, V_222, W_222 = _strassen_222()
    
    # Step 2: 2x1x2 with rank 4 (outer product)
    U_212, V_212, W_212 = _outer_212()
    
    # Step 3: 2x3x2 by combining 2x2x2 and 2x1x2
    # m dimension: 2 + 1 = 3
    U_232, V_232, W_232 = _block_combine(2, 2, 2, 2, U_222, V_222, W_222, 
                                           U_212, V_212, W_212)
    # rank = 7 + 4 = 11
    
    # Step 4: 2x5x2 by combining 2x2x2 and 2x3x2
    # m dimension: 2 + 3 = 5
    U_252, V_252, W_252 = _block_combine(2, 3, 2, 2, U_222, V_222, W_222,
                                           U_232, V_232, W_232)
    # rank = 7 + 11 = 18
    
    # Step 5: 2x5x6 by combining three 2x5x2 blocks
    # p dimension: 2 + 2 + 2 = 6
    U_256, V_256, W_256 = _block_combine_p(2, 5, 2, 2, U_252, V_252, W_252,
                                             U_252, V_252, W_252)
    # Now p=4, rank = 18 + 18 = 36
    U_256, V_256, W_256 = _block_combine_p(2, 5, 2, 2, U_252, V_252, W_252,
                                             U_256, V_256, W_256)
    # Now p=6, rank = 18 + 36 = 54
    
    if time.time() > deadline:
        return {"U": U_256.tolist(), "V": V_256.tolist(), "W": W_256.tolist()}
    
    # Convert to lists
    return {"U": U_256.tolist(), "V": V_256.tolist(), "W": W_256.tolist()}
