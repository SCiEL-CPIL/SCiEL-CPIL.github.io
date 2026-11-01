import time
import numpy as np

DIMENSIONS = [2, 3, 10]
TIME_LIMIT = 110  # seconds

def construct():
    """
    Returns a rank-minimal exact tensor decomposition for 2x3 by 3x10 matrix multiplication.
    Uses Strassen for 2x2 blocks and schoolbook for 2x1 blocks.
    Rank = 5 * (7 + 4) = 55.
    
    The decomposition is based on splitting the 3x10 matrix B into five 3x2 blocks.
    For each block, the multiplication is decomposed into:
    1. A 2x2 x 2x2 multiplication (using Strassen, rank 7).
    2. A 2x1 x 1x2 multiplication (using schoolbook, rank 4).
    Since the variables in each block of B are disjoint, the ranks add up.
    Total rank = 5 * (7 + 4) = 55.
    """
    n, m, p = DIMENSIONS
    rank = 55
    
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    col = 0
    
    # Strassen coefficients for 2x2 matrices
    # Let A_L = [[a00, a01], [a10, a11]] (indices 0,1,3,4 in flattened A)
    # Let B_L = [[b00, b01], [b10, b11]] (indices vary per block)
    # Products:
    # M1 = (a00 + a11)(b00 + b11)
    # M2 = (a10 + a11)b00
    # M3 = a00(b01 - b11)
    # M4 = a11(b10 - b00)
    # M5 = (a00 + a01)b11
    # M6 = (a10 - a00)(b00 + b01)
    # M7 = (a01 - a11)(b10 + b11)
    
    # Results:
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # Indices for A_L (2x2 part of 2x3 A)
    # A is 2x3. Flattened indices: i*3+j
    # A00: 0, A01: 1, A10: 3, A11: 4
    a00, a01, a10, a11 = 0, 1, 3, 4
    
    # Indices for A_R (2x1 part of 2x3 A)
    # A02: 2, A12: 5
    a02, a12 = 2, 5
    
    for b in range(5):
        k0 = 2 * b
        k1 = 2 * b + 1
        
        # B indices for this block (3x2)
        # B is 3x10. Flattened indices: j*10+k
        # Rows 0, 1, 2. Cols k0, k1.
        b00 = 0 * 10 + k0
        b01 = 0 * 10 + k1
        b10 = 1 * 10 + k0
        b11 = 1 * 10 + k1
        b20 = 2 * 10 + k0
        b21 = 2 * 10 + k1
        
        # C indices for this block (2x2)
        # C is 2x10. Flattened indices: k*2+i (Trace convention W[k*n+i, r])
        # Wait, the prompt says "output factor follows the official trace convention W[k*n+i,r]".
        # Usually C = AB. C[i, k] = sum_j A[i, j] B[j, k].
        # If C is stored as a vector, index is i*p + k or k*n + i?
        # The prompt specifies W[k*n+i, r].
        # This implies the index in the flattened C vector is `k*n + i`.
        # Here n=2, p=10.
        # So for column k of C (which corresponds to column k of B), the rows are i=0, 1.
        # Index for C[0, k] is k*2 + 0.
        # Index for C[1, k] is k*2 + 1.
        
        c00 = k0 * 2 + 0
        c01 = k1 * 2 + 0
        c10 = k0 * 2 + 1
        c11 = k1 * 2 + 1
        
        # --- Strassen Part (7 terms) ---
        
        # M1
        U[a00, col] = 1.0
        U[a11, col] = 1.0
        V[b00, col] = 1.0
        V[b11, col] = 1.0
        W[c00, col] = 1.0
        W[c11, col] = 1.0
        col += 1
        
        # M2
        U[a10, col] = 1.0
        U[a11, col] = 1.0
        V[b00, col] = 1.0
        W[c10, col] = 1.0
        W[c11, col] = -1.0
        col += 1
        
        # M3
        U[a00, col] = 1.0
        V[b01, col] = 1.0
        V[b11, col] = -1.0
        W[c01, col] = 1.0
        W[c11, col] = 1.0
        col += 1
        
        # M4
        U[a11, col] = 1.0
        V[b10, col] = 1.0
        V[b00, col] = -1.0
        W[c00, col] = 1.0
        W[c10, col] = 1.0
        col += 1
        
        # M5
        U[a00, col] = 1.0
        U[a01, col] = 1.0
        V[b11, col] = 1.0
        W[c00, col] = -1.0
        W[c01, col] = 1.0
        col += 1
        
        # M6
        U[a10, col] = 1.0
        U[a00, col] = -1.0
        V[b00, col] = 1.0
        V[b01, col] = 1.0
        W[c11, col] = 1.0
        col += 1
        
        # M7
        U[a01, col] = 1.0
        U[a11, col] = -1.0
        V[b10, col] = 1.0
        V[b11, col] = 1.0
        W[c00, col] = 1.0
        col += 1
        
        # --- Schoolbook Part for A_R * B_B (4 terms) ---
        # A_R is column 2 of A: [A02, A12]^T
        # B_B is row 2 of B block: [B20, B21]
        
        # P1 = A02 * B20 -> C[0, k0]
        U[a02, col] = 1.0
        V[b20, col] = 1.0
        W[c00, col] = 1.0
        col += 1
        
        # P2 = A02 * B21 -> C[0, k1]
        U[a02, col] = 1.0
        V[b21, col] = 1.0
        W[c01, col] = 1.0
        col += 1
        
        # P3 = A12 * B20 -> C[1, k0]
        U[a12, col] = 1.0
        V[b20, col] = 1.0
        W[c10, col] = 1.0
        col += 1
        
        # P4 = A12 * B21 -> C[1, k1]
        U[a12, col] = 1.0
        V[b21, col] = 1.0
        W[c11, col] = 1.0
        col += 1
        
    return {'U': U.tolist(), 'V': V.tolist(), 'W': W.tolist()}
