import numpy as np
import time

DIMENSIONS = [2, 3, 7]

# EVOLVE-BLOCK-START
def construct():
    """
    Returns a rank-41 tensor decomposition for matrix multiplication M_{2,3,7}.
    This uses the schoolbook decomposition for most terms, but applies Strassen's algorithm
    to the 2x2x2 sub-block (i in {0,1}, j in {0,1}, k in {0,1}) to reduce rank from 8 to 7.
    """
    n, m, p = DIMENSIONS
    # Total rank is 42 - 8 + 7 = 41
    rank = 41
    
    U = np.zeros((n * m, rank), dtype=np.int64)
    V = np.zeros((m * p, rank), dtype=np.int64)
    W = np.zeros((p * n, rank), dtype=np.int64)
    
    # Indices for the Strassen sub-block
    # i in {0, 1}, j in {0, 1}, k in {0, 1}
    # We assign columns 0 to 6 for Strassen terms
    # We assign columns 7 to 40 for the remaining schoolbook terms
    
    # --- Strassen Part (r = 0..6) ---
    # M1 = (A00 + A11)(B00 + B11)
    # U: A00, A11 -> (0,0), (1,1)
    # V: B00, B11 -> (0,0), (1,1)
    # W: C00, C11 -> (0,0), (1,1)
    U[0*m + 0, 0] = 1
    U[1*m + 1, 0] = 1
    V[0*p + 0, 0] = 1
    V[1*p + 1, 0] = 1
    W[0*n + 0, 0] = 1
    W[1*n + 1, 0] = 1
    
    # M2 = (A10 + A11)B00
    # U: A10, A11 -> (1,0), (1,1)
    # V: B00 -> (0,0)
    # W: C10 -> (0,1)
    U[1*m + 0, 1] = 1
    U[1*m + 1, 1] = 1
    V[0*p + 0, 1] = 1
    W[0*n + 1, 1] = 1
    
    # M3 = A00(B01 - B11)
    # U: A00 -> (0,0)
    # V: B01, -B11 -> (0,1), (1,1)
    # W: C01 -> (1,0)
    U[0*m + 0, 2] = 1
    V[0*p + 1, 2] = 1
    V[1*p + 1, 2] = -1
    W[1*n + 0, 2] = 1
    
    # M4 = A11(B10 - B00)
    # U: A11 -> (1,1)
    # V: B10, -B00 -> (1,0), (0,0)
    # W: C00, C10 -> (0,0), (0,1)
    U[1*m + 1, 3] = 1
    V[1*p + 0, 3] = 1
    V[0*p + 0, 3] = -1
    W[0*n + 0, 3] = 1
    W[0*n + 1, 3] = 1
    
    # M5 = (A00 + A01)B11
    # U: A00, A01 -> (0,0), (0,1)
    # V: B11 -> (1,1)
    # W: -C00, C01 -> (0,0), (1,0)
    U[0*m + 0, 4] = 1
    U[0*m + 1, 4] = 1
    V[1*p + 1, 4] = 1
    W[0*n + 0, 4] = -1
    W[1*n + 0, 4] = 1
    
    # M6 = (A10 - A00)(B00 + B01)
    # U: A10, -A00 -> (1,0), (0,0)
    # V: B00, B01 -> (0,0), (0,1)
    # W: C11 -> (1,1)
    U[1*m + 0, 5] = 1
    U[0*m + 0, 5] = -1
    V[0*p + 0, 5] = 1
    V[0*p + 1, 5] = 1
    W[1*n + 1, 5] = 1
    
    # M7 = (A01 - A11)(B10 + B11)
    # U: A01, -A11 -> (0,1), (1,1)
    # V: B10, B11 -> (1,0), (1,1)
    # W: C00 -> (0,0)
    U[0*m + 1, 6] = 1
    U[1*m + 1, 6] = -1
    V[1*p + 0, 6] = 1
    V[1*p + 1, 6] = 1
    W[0*n + 0, 6] = 1
    
    # --- Schoolbook Part (r = 7..40) ---
    # Iterate over all (i, j, k) not in the Strassen sub-block
    # Strassen sub-block: i in {0,1}, j in {0,1}, k in {0,1}
    # We skip these 8 combinations.
    
    col_idx = 7
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # Skip if in Strassen sub-block
                if i < 2 and j < 2 and k < 2:
                    continue
                
                # Schoolbook term
                U[i*m + j, col_idx] = 1
                V[j*p + k, col_idx] = 1
                W[k*n + i, col_idx] = 1
                col_idx += 1
                
    # Verify col_idx is 41
    # 42 total - 8 skipped = 34 terms. 7 + 34 = 41.
    
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
# EVOLVE-BLOCK-END
