"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 4]
import time
import numpy as np

# EVOLVE-BLOCK-START
def construct():
    """
    Constructs a rank-49 decomposition for 4x4 matrix multiplication
    using the tensor product of the rank-7 Strassen decomposition for 2x2.
    """
    start_time = time.time()
    deadline = start_time + 140  # Safety margin
    
    n, m, p = DIMENSIONS
    
    # Define the 2x2 Strassen decomposition matrices
    # U2[i, j, c] where index is i*2+j
    # V2[j, k, c] where index is j*2+k
    # W2[k, i, c] where index is k*2+i
    
    U2 = np.zeros((4, 7), dtype=int)
    V2 = np.zeros((4, 7), dtype=int)
    W2 = np.zeros((4, 7), dtype=int)
    
    # M0: (A00 + A11)(B00 + B11)
    U2[0, 0] = 1; U2[3, 0] = 1
    V2[0, 0] = 1; V2[3, 0] = 1
    # C00 (idx 0): 1, C11 (idx 3): 1
    W2[0, 0] = 1; W2[3, 0] = 1
    
    # M1: (A10 + A11)B00
    U2[2, 1] = 1; U2[3, 1] = 1
    V2[0, 1] = 1
    # C10 (idx 1): 1, C11 (idx 3): -1
    W2[1, 1] = 1; W2[3, 1] = -1
    
    # M2: A00(B01 - B11)
    U2[0, 2] = 1
    V2[1, 2] = 1; V2[3, 2] = -1
    # C01 (idx 2): 1, C11 (idx 3): 1
    W2[2, 2] = 1; W2[3, 2] = 1
    
    # M3: A11(B10 - B00)
    U2[3, 3] = 1
    V2[2, 3] = 1; V2[0, 3] = -1
    # C00 (idx 0): 1, C10 (idx 1): 1
    W2[0, 3] = 1; W2[1, 3] = 1
    
    # M4: (A00 + A01)B11
    U2[0, 4] = 1; U2[1, 4] = 1
    V2[3, 4] = 1
    # C00 (idx 0): -1, C01 (idx 2): 1
    W2[0, 4] = -1; W2[2, 4] = 1
    
    # M5: (A10 - A00)(B00 + B01)
    U2[2, 5] = 1; U2[0, 5] = -1
    V2[0, 5] = 1; V2[1, 5] = 1
    # C10 (idx 1): 1, C11 (idx 3): 1
    W2[1, 5] = 1; W2[3, 5] = 1
    
    # M6: (A01 - A11)(B10 + B11)
    U2[1, 6] = 1; U2[3, 6] = -1
    V2[2, 6] = 1; V2[3, 6] = 1
    # C00 (idx 0): -1
    W2[0, 6] = -1
    
    # Construct 4x4 matrices using Kronecker product logic
    rank = 49
    U = np.zeros((n * m, rank), dtype=int)
    V = np.zeros((m * p, rank), dtype=int)
    W = np.zeros((p * n, rank), dtype=int)
    
    # Vectorized construction
    for i in range(n):
        i1 = i // 2
        i2 = i % 2
        for j in range(m):
            j1 = j // 2
            j2 = j % 2
            u_row = i * m + j
            # Compute the row of U for this (i, j)
            # It is the Kronecker product of U2[i1*2+j1] and U2[i2*2+j2]
            u_val = np.kron(U2[i1 * 2 + j1], U2[i2 * 2 + j2])
            U[u_row, :] = u_val
            
    for j in range(m):
        j1 = j // 2
        j2 = j % 2
        for k in range(p):
            k1 = k // 2
            k2 = k % 2
            v_row = j * p + k
            v_val = np.kron(V2[j1 * 2 + k1], V2[j2 * 2 + k2])
            V[v_row, :] = v_val
            
    for k in range(p):
        k1 = k // 2
        k2 = k % 2
        for i in range(n):
            i1 = i // 2
            i2 = i % 2
            w_row = k * n + i
            w_val = np.kron(W2[k1 * 2 + i1], W2[k2 * 2 + i2])
            W[w_row, :] = w_val
            
    # Convert to lists as required by the interface
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
