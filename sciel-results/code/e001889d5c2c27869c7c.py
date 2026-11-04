"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 3]

# EVOLVE-BLOCK-START
import numpy as np

def construct():
    n, m, p = DIMENSIONS
    # Rank 17 decomposition for 2x3x3
    rank = 17
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    # Index helpers
    # U index: i * m + j
    # V index: j * p + k
    # W index: k * n + i
    
    col_idx = 0
    
    # Block 1: Strassen 2x2x2 for A[0:2, 0:2] * B[0:2, 0:2] -> C[0:2, 0:2]
    # A submatrix indices: (0,0)=0, (0,1)=1, (1,0)=3, (1,1)=4
    # B submatrix indices: (0,0)=0, (0,1)=1, (1,0)=3, (1,1)=4
    # C submatrix indices: (0,0)=0, (0,1)=1, (1,0)=2, (1,1)=3  (W: k*n+i)
    # W[0,0]=0, W[1,0]=1, W[0,1]=2, W[1,1]=3
    
    # M1 = (A00 + A11)(B00 + B11)
    U[0, col_idx] = 1; U[4, col_idx] = 1
    V[0, col_idx] = 1; V[4, col_idx] = 1
    W[0, col_idx] = 1  # C00
    W[3, col_idx] = 1  # C11
    col_idx += 1
    
    # M2 = (A10 + A11)B00
    U[3, col_idx] = 1; U[4, col_idx] = 1
    V[0, col_idx] = 1
    W[1, col_idx] = 1  # C10
    W[3, col_idx] = -1 # C11
    col_idx += 1
    
    # M3 = A00(B01 - B11)
    U[0, col_idx] = 1
    V[1, col_idx] = 1; V[4, col_idx] = -1
    W[2, col_idx] = 1  # C01
    W[3, col_idx] = 1  # C11
    col_idx += 1
    
    # M4 = A11(B10 - B00)
    U[4, col_idx] = 1
    V[3, col_idx] = 1; V[0, col_idx] = -1
    W[0, col_idx] = 1  # C00
    W[1, col_idx] = 1  # C10
    col_idx += 1
    
    # M5 = (A00 + A01)B11
    U[0, col_idx] = 1; U[1, col_idx] = 1
    V[4, col_idx] = 1
    W[0, col_idx] = -1 # C00
    W[2, col_idx] = 1  # C01
    col_idx += 1
    
    # M6 = (A10 - A00)(B00 + B10)
    U[3, col_idx] = 1; U[0, col_idx] = -1
    V[0, col_idx] = 1; V[3, col_idx] = 1
    W[3, col_idx] = 1  # C11
    col_idx += 1
    
    # M7 = (A01 - A11)(B10 + B11)
    U[1, col_idx] = 1; U[4, col_idx] = -1
    V[3, col_idx] = 1; V[4, col_idx] = 1
    W[0, col_idx] = 1  # C00
    col_idx += 1
    
    # Block 2: Schoolbook for A[0:2, 0:2] * B[0:2, 2] -> C[0:2, 2]
    # This is 2x2x1 multiplication.
    # C02 += A00 B02 + A01 B12
    # C12 += A10 B02 + A11 B12
    # W index for C02 is k=2, i=0 -> 2*2 + 0 = 4
    # W index for C12 is k=2, i=1 -> 2*2 + 1 = 5
    
    # Col: A00 B02
    U[0, col_idx] = 1
    V[2, col_idx] = 1
    W[4, col_idx] = 1
    col_idx += 1
    
    # Col: A01 B12
    U[1, col_idx] = 1
    V[5, col_idx] = 1
    W[4, col_idx] = 1
    col_idx += 1
    
    # Col: A10 B02
    U[3, col_idx] = 1
    V[2, col_idx] = 1
    W[5, col_idx] = 1
    col_idx += 1
    
    # Col: A11 B12
    U[4, col_idx] = 1
    V[5, col_idx] = 1
    W[5, col_idx] = 1
    col_idx += 1
    
    # Block 3: Schoolbook for A[0:2, 2] * B[2, 0:3] -> C[0:2, 0:3]
    # This is 2x1x3 multiplication.
    # C0k += A02 B2k
    # C1k += A12 B2k
    
    # Col: A02 B20
    U[2, col_idx] = 1
    V[6, col_idx] = 1
    W[0, col_idx] = 1 # C00
    col_idx += 1
    
    # Col: A02 B21
    U[2, col_idx] = 1
    V[7, col_idx] = 1
    W[2, col_idx] = 1 # C01
    col_idx += 1
    
    # Col: A02 B22
    U[2, col_idx] = 1
    V[8, col_idx] = 1
    W[4, col_idx] = 1 # C02
    col_idx += 1
    
    # Col: A12 B20
    U[5, col_idx] = 1
    V[6, col_idx] = 1
    W[1, col_idx] = 1 # C10
    col_idx += 1
    
    # Col: A12 B21
    U[5, col_idx] = 1
    V[7, col_idx] = 1
    W[3, col_idx] = 1 # C11
    col_idx += 1
    
    # Col: A12 B22
    U[5, col_idx] = 1
    V[8, col_idx] = 1
    W[5, col_idx] = 1 # C12
    col_idx += 1
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
