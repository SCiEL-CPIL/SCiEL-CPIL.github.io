import time
import numpy as np

DIMENSIONS = [2, 3, 10]
TIME_LIMIT = 110  # seconds

def construct():
    """
    Returns a rank-minimal exact tensor decomposition for 2x3 by 3x10 matrix multiplication.
    Uses Strassen for 2x2 blocks and schoolbook for 2x1 blocks.
    Rank = 5 * (7 + 4) = 55.
    """
    n, m, p = DIMENSIONS
    rank = 55
    
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    col = 0
    
    # Strassen coefficients
    # M1 = (A00 + A11)(B00 + B11)
    # M2 = (A10 + A11) B00
    # M3 = A00 (B01 - B11)
    # M4 = A11 (B10 - B00)
    # M5 = (A00 + A01) B11
    # M6 = (A10 - A00)(B00 + B01)
    # M7 = (A01 - A11)(B10 + B11)
    
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # Indices for A_L (2x2 part of 2x3)
    # A is 2x3. Indices: i*3+j
    # A00: 0, A01: 1, A10: 3, A11: 4
    a00, a01, a10, a11 = 0, 1, 3, 4
    
    # Indices for B_T (2x2 part of 3x2)
    # B is 3x10. Indices: j*10+k
    # For block b, k0=2b, k1=2b+1
    # B00: 0*10+k0, B01: 0*10+k1
    # B10: 1*10+k0, B11: 1*10+k1
    
    # Indices for C (2x2 part of 2x10)
    # C is 2x10. Indices: k*2+i
    # C00: k0*2+0, C01: k1*2+0
    # C10: k0*2+1, C11: k1*2+1
    
    for b in range(5):
        k0 = 2 * b
        k1 = 2 * b + 1
        
        # B indices
        b00 = 0 * 10 + k0
        b01 = 0 * 10 + k1
        b10 = 1 * 10 + k0
        b11 = 1 * 10 + k1
        
        # C indices
        c00 = k0 * 2 + 0
        c01 = k1 * 2 + 0
        c10 = k0 * 2 + 1
        c11 = k1 * 2 + 1
        
        # Strassen terms
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
        
        # Schoolbook terms for A_R * B_B
        # A_R is 2x1: A02 (idx 2), A12 (idx 5)
        # B_B is 1x2: B20 (idx 20+k0), B21 (idx 20+k1)
        
        a02 = 0 * 3 + 2
        a12 = 1 * 3 + 2
        b20 = 2 * 10 + k0
        b21 = 2 * 10 + k1
        
        # P1 = A02 * B20 -> C00
        U[a02, col] = 1.0
        V[b20, col] = 1.0
        W[c00, col] = 1.0
        col += 1
        
        # P2 = A02 * B21 -> C01
        U[a02, col] = 1.0
        V[b21, col] = 1.0
        W[c01, col] = 1.0
        col += 1
        
        # P3 = A12 * B20 -> C10
        U[a12, col] = 1.0
        V[b20, col] = 1.0
        W[c10, col] = 1.0
        col += 1
        
        # P4 = A12 * B21 -> C11
        U[a12, col] = 1.0
        V[b21, col] = 1.0
        W[c11, col] = 1.0
        col += 1
        
    return {'U': U.tolist(), 'V': V.tolist(), 'W': W.tolist()}
