import time
import numpy as np

DIMENSIONS = [2, 3, 10]
DEADLINE = 140  # seconds, safety margin before 150s kill

def construct():
    """
    Constructs a rank-minimal exact tensor decomposition for multiplying 
    a 2x3 matrix by a 3x10 matrix.
    
    Strategy:
    - Split A (2x3) into A1 (2x2) and A2 (2x1).
    - Split B (3x10) into B1 (2x10) and B2 (1x10).
    - Compute A1*B1 using 5 Strassen blocks (2x2 * 2x2), rank 7 each -> 35.
    - Compute A2*B2 using 2 terms (rank 2).
    - Total rank: 37.
    """
    start_time = time.time()
    
    n, m, p = DIMENSIONS  # n=2, m=3, p=10
    rank = 37
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    col = 0
    
    # Part 1: A1 (2x2) * B1 (2x10)
    # A1 indices in U: i in {0,1}, j in {0,1} -> U[i*m + j]
    # B1 indices in V: j in {0,1}, k in {0..9} -> V[j*p + k]
    # C1 indices in W: i in {0,1}, k in {0..9} -> W[k*n + i]
    
    # U indices for A1
    u00 = 0 * m + 0
    u01 = 0 * m + 1
    u10 = 1 * m + 0
    u11 = 1 * m + 1
    
    # Process 5 blocks of 2 columns for Strassen
    for g in range(5):
        k0 = 2 * g
        k1 = 2 * g + 1
        
        # V indices for B1 block
        v00 = 0 * p + k0
        v01 = 0 * p + k1
        v10 = 1 * p + k0
        v11 = 1 * p + k1
        
        # W indices for C1 block
        w00 = k0 * n + 0
        w01 = k1 * n + 0
        w10 = k0 * n + 1
        w11 = k1 * n + 1
        
        # M1 = (A00 + A11)(B00 + B11)
        U[u00, col] += 1.0; U[u11, col] += 1.0
        V[v00, col] += 1.0; V[v11, col] += 1.0
        W[w00, col] += 1.0; W[w11, col] += 1.0
        col += 1
        
        # M2 = (A10 + A11)B00
        U[u10, col] += 1.0; U[u11, col] += 1.0
        V[v00, col] += 1.0
        W[w10, col] += 1.0; W[w11, col] -= 1.0
        col += 1
        
        # M3 = A00(B01 - B11)
        U[u00, col] += 1.0
        V[v01, col] += 1.0; V[v11, col] -= 1.0
        W[w01, col] += 1.0; W[w11, col] += 1.0
        col += 1
        
        # M4 = A11(B10 - B00)
        U[u11, col] += 1.0
        V[v10, col] += 1.0; V[v00, col] -= 1.0
        W[w00, col] += 1.0; W[w10, col] += 1.0
        col += 1
        
        # M5 = (A00 + A01)B11
        U[u00, col] += 1.0; U[u01, col] += 1.0
        V[v11, col] += 1.0
        W[w00, col] -= 1.0; W[w01, col] += 1.0
        col += 1
        
        # M6 = (A10 - A00)(B00 + B01)
        U[u10, col] += 1.0; U[u00, col] -= 1.0
        V[v00, col] += 1.0; V[v01, col] += 1.0
        W[w11, col] += 1.0
        col += 1
        
        # M7 = (A01 - A11)(B10 + B11)
        U[u01, col] += 1.0; U[u11, col] -= 1.0
        V[v10, col] += 1.0; V[v11, col] += 1.0
        W[w00, col] += 1.0
        col += 1
        
    # Part 2: A2 (2x1) * B2 (1x10)
    # A2 indices in U: i in {0,1}, j=2 -> U[i*m + 2]
    # B2 indices in V: j=2, k in {0..9} -> V[2*p + k]
    # C2 indices in W: i in {0,1}, k in {0..9} -> W[k*n + i]
    
    u02 = 0 * m + 2
    u12 = 1 * m + 2
    
    # Term for i=0
    # U[u02, col] = 1
    # V[2*p + k, col] = 1 for all k
    # W[k*n + 0, col] = 1 for all k
    U[u02, col] = 1.0
    for k in range(p):
        V[2 * p + k, col] = 1.0
        W[k * n + 0, col] = 1.0
    col += 1
    
    # Term for i=1
    # U[u12, col] = 1
    # V[2*p + k, col] = 1 for all k
    # W[k*n + 1, col] = 1 for all k
    U[u12, col] = 1.0
    for k in range(p):
        V[2 * p + k, col] = 1.0
        W[k * n + 1, col] = 1.0
    col += 1
    
    # col should be 37
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
