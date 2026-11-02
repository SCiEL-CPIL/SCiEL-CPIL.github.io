import numpy as np

def construct():
    """
    Constructs a rank-28 decomposition for 2x2x8 matrix multiplication.
    This is based on applying Strassen's algorithm (rank 7) to each of the 4 
    2x2 blocks of the 2x8 matrix.
    """
    n, m, p = 2, 2, 8
    num_a = n * m  # 4
    num_b = m * p  # 16
    num_c = p * n  # 16
    
    rank = 28
    
    # Initialize matrices
    U = np.zeros((num_a, rank), dtype=float)
    V = np.zeros((num_b, rank), dtype=float)
    W = np.zeros((num_c, rank), dtype=float)
    
    # Strassen coefficients for 2x2x2
    # Indices for A: 00, 01, 10, 11 -> 0, 1, 2, 3
    # Indices for B: 00, 01, 10, 11 -> 0, 1, 2, 3 (relative to block)
    # Indices for C: 00, 01, 10, 11 -> 0, 1, 2, 3 (relative to block)
    
    # Strassen terms:
    # M1 = (a00+a11)(b00+b11)
    # M2 = (a10+a11)b00
    # M3 = a00(b10-b11)
    # M4 = a11(b01-b00)
    # M5 = (a00+a01)b11
    # M6 = (a10-a00)(b00+b10)
    # M7 = (a01-a11)(b01+b11)
    
    # C reconstruction:
    # c00 = M1 + M4 - M5 + M7
    # c01 = M3 + M5
    # c10 = M2 + M4
    # c11 = M1 - M2 + M3 + M6
    
    # We iterate over 4 blocks k=0..3
    # Block k corresponds to B columns 2k, 2k+1 and C columns 2k, 2k+1
    # B index: b = j * p + k_col. Here k_col is 0..7.
    # For block k, k_col is 2k and 2k+1.
    # j is 0 or 1.
    # So B indices for block k are:
    # b_00_k = 0 * 8 + 2k = 2k
    # b_01_k = 0 * 8 + 2k+1 = 2k+1
    # b_10_k = 1 * 8 + 2k = 8+2k
    # b_11_k = 1 * 8 + 2k+1 = 8+2k+1
    
    # C index: c = k_col * n + i. Here n=2.
    # c_00_k = 2k * 2 + 0 = 4k
    # c_01_k = (2k+1) * 2 + 0 = 4k+2
    # c_10_k = 2k * 2 + 1 = 4k+1
    # c_11_k = (2k+1) * 2 + 1 = 4k+3
    
    for k in range(4):
        # Base index for this block's terms in U, V, W
        base = k * 7
        
        # B indices
        b00 = 2 * k
        b01 = 2 * k + 1
        b10 = 8 + 2 * k
        b11 = 8 + 2 * k + 1
        
        # C indices
        c00 = 4 * k
        c01 = 4 * k + 2
        c10 = 4 * k + 1
        c11 = 4 * k + 3
        
        # Term 0: M1
        U[0, base+0] = 1 # a00
        U[3, base+0] = 1 # a11
        V[b00, base+0] = 1
        V[b11, base+0] = 1
        W[c00, base+0] = 1
        W[c11, base+0] = 1
        
        # Term 1: M2
        U[2, base+1] = 1 # a10
        U[3, base+1] = 1 # a11
        V[b00, base+1] = 1
        W[c10, base+1] = 1
        W[c11, base+1] = -1
        
        # Term 2: M3
        U[0, base+2] = 1 # a00
        V[b10, base+2] = 1
        V[b11, base+2] = -1
        W[c01, base+2] = 1
        W[c11, base+2] = 1
        
        # Term 3: M4
        U[3, base+3] = 1 # a11
        V[b01, base+3] = 1
        V[b00, base+3] = -1
        W[c00, base+3] = 1
        W[c10, base+3] = 1
        
        # Term 4: M5
        U[0, base+4] = 1 # a00
        U[1, base+4] = 1 # a01
        V[b11, base+4] = 1
        W[c00, base+4] = -1
        W[c01, base+4] = 1
        
        # Term 5: M6
        U[2, base+5] = 1 # a10
        U[0, base+5] = -1 # a00
        V[b00, base+5] = 1
        V[b10, base+5] = 1
        W[c11, base+5] = 1
        
        # Term 6: M7
        U[1, base+6] = 1 # a01
        U[3, base+6] = -1 # a11
        V[b01, base+6] = 1
        V[b11, base+6] = 1
        W[c00, base+6] = 1
        
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
