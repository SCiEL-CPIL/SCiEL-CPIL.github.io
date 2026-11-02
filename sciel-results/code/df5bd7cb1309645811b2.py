import numpy as np

def construct():
    """
    Construct a rank-25 exact tensor decomposition for 2x2 * 2x7 matrix multiplication.
    
    This implementation uses a hybrid approach:
    - Strassen's rank-7 algorithm for the first three 2x2 column blocks (columns 0-1, 2-3, 4-5).
    - Standard rank-4 schoolbook multiplication for the final single column (column 6).
    
    Total rank: 7 + 7 + 7 + 4 = 25.
    """
    n, m, p = 2, 2, 7
    rank = 25
    
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    # Helper to add Strassen block for columns k, k+1
    def add_strassen(k_start, col_offset):
        # U indices: 0: A00, 1: A01, 2: A10, 3: A11
        # V indices: 7*j + k
        # W indices: 2*k + i
        
        # M1
        U[0, col_offset] += 1.0
        U[3, col_offset] += 1.0
        V[k_start, col_offset] += 1.0      # B0,k_start
        V[7 + k_start + 1, col_offset] += 1.0 # B1,k_start+1
        W[2 * k_start, col_offset] += 1.0    # C0,k_start
        W[2 * (k_start + 1) + 1, col_offset] += 1.0 # C1,k_start+1
        
        # M2
        U[2, col_offset + 1] += 1.0
        U[3, col_offset + 1] += 1.0
        V[k_start, col_offset + 1] += 1.0
        W[2 * k_start + 1, col_offset + 1] += 1.0
        W[2 * (k_start + 1) + 1, col_offset + 1] -= 1.0
        
        # M3
        U[0, col_offset + 2] += 1.0
        V[7 + k_start + 1, col_offset + 2] += 1.0
        V[7 + k_start + 2, col_offset + 2] -= 1.0 # B1,k_start+1 minus B1,k_start+1? No.
        # Correction: M3 = A00 * (B0,k_start+1 - B1,k_start+1)
        V[7 + k_start + 1, col_offset + 2] = 1.0
        V[7 + k_start + 2, col_offset + 2] = -1.0
        W[2 * (k_start + 1), col_offset + 2] += 1.0
        W[2 * (k_start + 1) + 1, col_offset + 2] += 1.0
        
        # M4
        U[3, col_offset + 3] += 1.0
        V[7 + k_start, col_offset + 3] += 1.0
        V[k_start, col_offset + 3] -= 1.0 # B1,k_start - B0,k_start
        # Correction: M4 = A11 * (B1,k_start - B0,k_start)
        V[7 + k_start, col_offset + 3] = 1.0
        V[k_start, col_offset + 3] = -1.0
        W[2 * k_start, col_offset + 3] += 1.0
        W[2 * k_start + 1, col_offset + 3] += 1.0
        
        # M5
        U[0, col_offset + 4] += 1.0
        U[1, col_offset + 4] += 1.0
        V[7 + k_start + 1, col_offset + 4] += 1.0
        W[2 * k_start, col_offset + 4] -= 1.0
        W[2 * (k_start + 1), col_offset + 4] += 1.0
        
        # M6
        U[2, col_offset + 5] += 1.0
        U[0, col_offset + 5] -= 1.0
        V[k_start, col_offset + 5] += 1.0
        V[7 + k_start + 1, col_offset + 5] += 1.0
        W[2 * (k_start + 1) + 1, col_offset + 5] += 1.0
        
        # M7
        U[1, col_offset + 6] += 1.0
        U[3, col_offset + 6] -= 1.0
        V[7 + k_start, col_offset + 6] += 1.0
        V[7 + k_start + 1, col_offset + 6] += 1.0
        W[2 * k_start, col_offset + 6] += 1.0

    # Block 1: cols 0, 1
    add_strassen(0, 0)
    # Block 2: cols 2, 3
    add_strassen(2, 7)
    # Block 3: cols 4, 5
    add_strassen(4, 14)
    
    # Block 4: col 6 (Schoolbook)
    k = 6
    off = 21
    U[0, off] = 1.0
    V[k, off] = 1.0
    W[2*k, off] = 1.0
    
    U[1, off+1] = 1.0
    V[7+k, off+1] = 1.0
    W[2*k, off+1] = 1.0
    
    U[2, off+2] = 1.0
    V[k, off+2] = 1.0
    W[2*k+1, off+2] = 1.0
    
    U[3, off+3] = 1.0
    V[7+k, off+3] = 1.0
    W[2*k+1, off+3] = 1.0

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
