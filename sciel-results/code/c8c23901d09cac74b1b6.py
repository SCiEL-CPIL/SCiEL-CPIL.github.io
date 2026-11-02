import numpy as np

DIMENSIONS = [2, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank = 39
    
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    # Strassen decomposition for the 2x2x2 sub-block (i,j,k in {0,1})
    # Global indices:
    # U: i*m + j
    # V: j*p + k
    # W: k*n + i
    
    # Term 1: M1 = (A00+A11)(B00+B11)
    U[0, 0] = 1; U[5, 0] = 1
    V[0, 0] = 1; V[6, 0] = 1
    W[0, 0] = 1; W[3, 0] = 1
    
    # Term 2: M2 = (A10+A11)B00
    U[4, 1] = 1; U[5, 1] = 1
    V[0, 1] = 1
    W[0, 1] = -1; W[2, 1] = 1
    
    # Term 3: M3 = A00(B10-B11)
    U[0, 2] = 1
    V[5, 2] = 1; V[6, 2] = -1
    W[0, 2] = 1; W[1, 2] = 1
    
    # Term 4: M4 = A11(B01-B00)
    U[5, 3] = 1
    V[1, 3] = 1; V[0, 3] = -1
    W[2, 3] = 1; W[3, 3] = 1
    
    # Term 5: M5 = (A00+A10)B11
    U[0, 4] = 1; U[4, 4] = 1
    V[6, 4] = 1
    W[1, 4] = 1; W[3, 4] = -1
    
    # Term 6: M6 = (A10-A00)(B00+B10)
    U[4, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[5, 5] = 1
    W[0, 5] = 1
    
    # Term 7: M7 = (A01-A11)(B11+B01)
    U[1, 6] = 1; U[5, 6] = -1
    V[6, 6] = 1; V[1, 6] = 1
    W[3, 6] = 1
    
    # Remaining 32 terms
    col = 7
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # Skip if in the 2x2x2 block handled by Strassen
                if j in [0, 1] and k in [0, 1]:
                    continue
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
                
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
