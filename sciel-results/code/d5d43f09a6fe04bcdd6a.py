import numpy as np
from itertools import product
import time

DIMENSIONS = [2, 3, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    # Rank 29 construction using Strassen for 2x2x2 sub-block
    rank = 29
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    col = 0
    
    # 1. Strassen terms for i in {0,1}, j in {0,1}, k in {0,1}
    # Indices:
    # U: i*m+j -> i*3+j
    # V: j*p+k -> j*5+k
    # W: k*n+i -> k*2+i
    
    # M1 = (A00 + A11)(B00 + B11)
    # U: A00 (0), A11 (4)
    # V: B00 (0), B11 (6)
    # W: C00 (0), C11 (3)
    U[0, col] = 1; U[4, col] = 1
    V[0, col] = 1; V[6, col] = 1
    W[0, col] = 1; W[3, col] = 1
    col += 1
    
    # M2 = (A10 + A11)B00
    # U: A10 (3), A11 (4)
    # V: B00 (0)
    # W: C10 (1), C11 (3) -> C10 +1, C11 -1
    U[3, col] = 1; U[4, col] = 1
    V[0, col] = 1
    W[1, col] = 1; W[3, col] = -1
    col += 1
    
    # M3 = A00(B01 - B11)
    # U: A00 (0)
    # V: B01 (1), B11 (6) -> B01 +1, B11 -1
    # W: C01 (2), C11 (3) -> C01 +1, C11 +1
    U[0, col] = 1
    V[1, col] = 1; V[6, col] = -1
    W[2, col] = 1; W[3, col] = 1
    col += 1
    
    # M4 = A11(B10 - B00)
    # U: A11 (4)
    # V: B10 (5), B00 (0) -> B10 +1, B00 -1
    # W: C00 (0), C10 (1) -> C00 +1, C10 +1
    U[4, col] = 1
    V[5, col] = 1; V[0, col] = -1
    W[0, col] = 1; W[1, col] = 1
    col += 1
    
    # M5 = (A00 + A01)B11
    # U: A00 (0), A01 (1)
    # V: B11 (6)
    # W: C00 (0), C01 (2) -> C00 -1, C01 +1
    U[0, col] = 1; U[1, col] = 1
    V[6, col] = 1
    W[0, col] = -1; W[2, col] = 1
    col += 1
    
    # M6 = (A10 - A00)(B00 + B01)
    # U: A10 (3), A00 (0) -> A10 +1, A00 -1
    # V: B00 (0), B01 (1)
    # W: C11 (3) -> C11 +1
    U[3, col] = 1; U[0, col] = -1
    V[0, col] = 1; V[1, col] = 1
    W[3, col] = 1
    col += 1
    
    # M7 = (A01 - A11)(B10 + B11)
    # U: A01 (1), A11 (4) -> A01 +1, A11 -1
    # V: B10 (5), B11 (6)
    # W: C00 (0) -> C00 +1
    U[1, col] = 1; U[4, col] = -1
    V[5, col] = 1; V[6, col] = 1
    W[0, col] = 1
    col += 1
    
    # 2. Schoolbook terms for i in {0,1}, j=2, k in {0,1}
    # These handle A_i2 * B_2k contributions to C_ki
    for i in range(2):
        for k in range(2):
            U[i*3+2, col] = 1
            V[2*5+k, col] = 1
            W[k*2+i, col] = 1
            col += 1
            
    # 3. Schoolbook terms for i in {0,1}, j in {0,1,2}, k in {2,3,4}
    for i in range(2):
        for j in range(3):
            for k in range(2, 5):
                U[i*3+j, col] = 1
                V[j*5+k, col] = 1
                W[k*2+i, col] = 1
                col += 1
                
    # Verify col count
    assert col == rank, f"Expected rank {rank}, got {col}"
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
