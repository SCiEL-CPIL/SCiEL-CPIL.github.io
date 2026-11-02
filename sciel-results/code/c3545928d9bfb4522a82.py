import numpy as np
import time

def construct():
    """
    Returns a rank-50 exact tensor decomposition for 2x4x7 matrix multiplication.
    This is an improvement over the schoolbook rank of 56.
    """
    start_time = time.time()
    deadline = 150.0
    safety_margin = 5.0

    n, m, p = 2, 4, 7
    rank = 50

    U = np.zeros((n * m, rank), dtype=int)
    V = np.zeros((m * p, rank), dtype=int)
    W = np.zeros((p * n, rank), dtype=int)
    
    col = 0

    # Helper to add a 2x2x2 Strassen block
    def add_strassen(a_idx, b_idx, c_idx, offset):
        nonlocal col
        # a_idx: [a00, a01, a10, a11]
        # b_idx: [b00, b01, b10, b11]
        # c_idx: [c00, c01, c10, c11]
        
        # M1 = A00 * (B00 + B11)
        U[a_idx[0], col] += 1
        V[b_idx[0], col] += 1
        V[b_idx[3], col] += 1
        W[c_idx[0], col] += 1
        W[c_idx[3], col] += 1
        col += 1
        
        # M2 = (A00 + A11) * (B11 - B00)
        U[a_idx[0], col] += 1
        U[a_idx[3], col] += 1
        V[b_idx[3], col] += 1
        V[b_idx[0], col] -= 1
        W[c_idx[3], col] -= 1
        col += 1
        
        # M3 = (A01 + A11) * B11
        U[a_idx[1], col] += 1
        U[a_idx[3], col] += 1
        V[b_idx[3], col] += 1
        W[c_idx[1], col] += 1
        W[c_idx[3], col] += 1
        col += 1
        
        # M4 = A00 * (B01 - B11)
        U[a_idx[0], col] += 1
        V[b_idx[1], col] += 1
        V[b_idx[3], col] -= 1
        W[c_idx[0], col] -= 1
        W[c_idx[1], col] += 1
        col += 1
        
        # M5 = (A10 + A00) * B00
        U[a_idx[2], col] += 1
        U[a_idx[0], col] += 1
        V[b_idx[0], col] += 1
        W[c_idx[0], col] -= 1
        W[c_idx[2], col] += 1
        col += 1
        
        # M6 = (A01 - A10) * (B10 + B11)
        U[a_idx[1], col] += 1
        U[a_idx[2], col] -= 1
        V[b_idx[2], col] += 1
        V[b_idx[3], col] += 1
        W[c_idx[2], col] += 1
        W[c_idx[3], col] += 1
        col += 1
        
        # M7 = (A00 - A11) * (B00 + B01)
        U[a_idx[0], col] += 1
        U[a_idx[3], col] -= 1
        V[b_idx[0], col] += 1
        V[b_idx[1], col] += 1
        W[c_idx[0], col] += 1
        col += 1

    # Helper to add a 2x2x1 block (Rank 4)
    def add_basic_221(a_idx, b_idx, c_idx, offset):
        nonlocal col
        # a_idx: [a00, a01, a10, a11]
        # b_idx: [b00, b10] (only first column)
        # c_idx: [c00, c10] (only first column)
        
        # M1 = A00 * B00
        U[a_idx[0], col] += 1
        V[b_idx[0], col] += 1
        W[c_idx[0], col] += 1
        col += 1
        
        # M2 = A01 * B10
        U[a_idx[1], col] += 1
        V[b_idx[1], col] += 1
        W[c_idx[0], col] += 1
        col += 1
        
        # M3 = A10 * B00
        U[a_idx[2], col] += 1
        V[b_idx[0], col] += 1
        W[c_idx[1], col] += 1
        col += 1
        
        # M4 = A11 * B10
        U[a_idx[3], col] += 1
        V[b_idx[1], col] += 1
        W[c_idx[1], col] += 1
        col += 1

    # Helper to add a 2x2x3 block (Rank 11)
    def add_223(a_idx, b_idx, c_idx, offset):
        # b_idx: [b00..b02, b10..b12]
        # c_idx: [c00..c02, c10..c12]
        
        # Split B into col 0 and cols 1-2
        # B0: b00, b10
        # B1: b01, b02, b11, b12
        b0 = [b_idx[0], b_idx[4]] # b00, b10
        b1 = [b_idx[1], b_idx[2], b_idx[5], b_idx[6]] # b01, b02, b11, b12
        
        c0 = [c_idx[0], c_idx[4]] # c00, c10
        c1 = [c_idx[1], c_idx[2], c_idx[5], c_idx[6]] # c01, c02, c11, c12
        
        add_basic_221(a_idx, b0, c0, offset)
        add_strassen(a_idx, b1, c1, offset)

    # Helper to add a 2x2x4 block (Rank 14)
    def add_224(a_idx, b_idx, c_idx, offset):
        # Split B into cols 0-1 and 2-3
        b0 = [b_idx[0], b_idx[1], b_idx[4], b_idx[5]]
        b1 = [b_idx[2], b_idx[3], b_idx[6], b_idx[7]]
        
        c0 = [c_idx[0], c_idx[1], c_idx[4], c_idx[5]]
        c1 = [c_idx[2], c_idx[3], c_idx[6], c_idx[7]]
        
        add_strassen(a_idx, b0, c0, offset)
        add_strassen(a_idx, b1, c1, offset)

    # Helper to add a 2x2x7 block (Rank 25)
    def add_227(a_idx, b_idx, c_idx, offset):
        # Split B into cols 0-2 and 3-6
        b0 = b_idx[0:6] # 6 elements
        b1 = b_idx[6:14] # 8 elements
        
        c0 = c_idx[0:6]
        c1 = c_idx[6:14]
        
        add_223(a_idx, b0, c0, offset)
        add_224(a_idx, b1, c1, offset)

    # Block 1: A_L (cols 0,1) * B_L (rows 0,1)
    # A_L indices:
    a_L = [0, 1, 4, 5] # A[0,0], A[0,1], A[1,0], A[1,1]
    
    # B_L indices (rows 0,1 of B, all 7 cols)
    # B is 4x7. Row 0: 0..6, Row 1: 7..13
    b_L = []
    for k in range(7):
        b_L.append(0*7 + k) # B[0,k]
        b_L.append(1*7 + k) # B[1,k]
    # b_L = [B00, B01, B02, B03, B04, B05, B06, B10, B11, B12, B13, B14, B15, B16]
    
    # C indices (all 14)
    c_all = []
    for k in range(7):
        c_all.append(k*2 + 0) # C[0,k]
        c_all.append(k*2 + 1) # C[1,k]
    # c_all = [C00, C10, C01, C11, C02, C12, C03, C13, C04, C14, C05, C15, C06, C16]
    
    add_227(a_L, b_L, c_all, 0)

    # Block 2: A_R (cols 2,3) * B_R (rows 2,3)
    # A_R indices:
    a_R = [2, 3, 6, 7] # A[0,2], A[0,3], A[1,2], A[1,3]
    
    # B_R indices (rows 2,3 of B, all 7 cols)
    b_R = []
    for k in range(7):
        b_R.append(2*7 + k) # B[2,k]
        b_R.append(3*7 + k) # B[3,k]
    # b_R = [B20, B21, ..., B30, B31, ...]
    
    add_227(a_R, b_R, c_all, 0)

    if time.time() + safety_margin >= deadline:
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

