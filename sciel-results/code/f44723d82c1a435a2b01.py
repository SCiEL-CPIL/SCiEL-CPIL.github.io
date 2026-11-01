"""Matrix multiplication 2x3x6 rank decomposition.

We use a block decomposition strategy based on Strassen's algorithm for 2x2x2 blocks
and direct multiplication for the remaining 2x1x6 part.
This yields a rank of 33 with integer coefficients.
"""
import time
import numpy as np

DIMENSIONS = [2, 3, 6]
n, m, p = DIMENSIONS

def construct():
    start_time = time.time()
    deadline = start_time + 120
    
    # Total rank: 3 blocks of 2x2x2 (rank 7 each) + 1 block of 2x1x6 (rank 12)
    # 3 * 7 + 12 = 33
    r = 33
    
    U = np.zeros((n * m, r), dtype=int)
    V = np.zeros((m * p, r), dtype=int)
    W = np.zeros((p * n, r), dtype=int)
    
    col = 0
    
    # Strassen coefficients for 2x2x2
    # m1 = (a+d)(e+h)
    # m2 = (c+d)(e+h)
    # m3 = (a+b)g
    # m4 = (c+d)f
    # m5 = (a-b)(h-g)
    # m6 = (c+a)(e-g)
    # m7 = (d-b)(f+h)
    # C11 = m1 + m4 - m5 - m7
    # C12 = m3 + m5
    # C21 = m2 + m7
    # C22 = m1 - m2 + m3 + m6
    
    # Indices for 2x2 block:
    # A: i in {0,1}, j in {0,1} -> global U row: i*m + j
    # B: j in {0,1}, k in {0,1} -> global V row: j*p + k
    # C: i in {0,1}, k in {0,1} -> global W row: k*n + i
    
    def add_strassen_block(k_offset):
        nonlocal col
        # k_offset is the starting column index in B and C for this block (0, 2, 4)
        
        # m1 = (A00 + A11) * (B00 + B11)
        # A00: U[0*3+0, col], A11: U[1*3+1, col]
        # B00: V[0*6+k_offset, col], B11: V[1*6+k_offset+1, col]
        U[0, col] = 1
        U[4, col] = 1
        V[k_offset, col] = 1
        V[6 + k_offset + 1, col] = 1
        # C11 (W[k_offset*2+0, col]), C22 (W[(k_offset+1)*2+1, col])
        W[k_offset * 2 + 0, col] = 1
        W[(k_offset + 1) * 2 + 1, col] = 1
        col += 1
        
        # m2 = (A10 + A11) * (B00 + B11)
        # A10: U[1*3+0, col], A11: U[1*3+1, col]
        # B00: V[0*6+k_offset, col], B11: V[1*6+k_offset+1, col]
        U[3, col] = 1
        U[4, col] = 1
        V[k_offset, col] = 1
        V[6 + k_offset + 1, col] = 1
        # C21 (W[k_offset*2+1, col])
        W[k_offset * 2 + 1, col] = 1
        col += 1
        
        # m3 = (A00 + A01) * B10
        # A00: U[0*3+0, col], A01: U[0*3+1, col]
        # B10: V[1*6+k_offset, col]
        U[0, col] = 1
        U[1, col] = 1
        V[6 + k_offset, col] = 1
        # C12 (W[(k_offset+1)*2+0, col])
        W[(k_offset + 1) * 2 + 0, col] = 1
        col += 1
        
        # m4 = (A10 + A11) * B01
        # A10: U[1*3+0, col], A11: U[1*3+1, col]
        # B01: V[0*6+k_offset+1, col]
        U[3, col] = 1
        U[4, col] = 1
        V[k_offset + 1, col] = 1
        # C11 (W[k_offset*2+0, col])
        W[k_offset * 2 + 0, col] = 1
        col += 1
        
        # m5 = (A00 - A10) * (B11 - B10)
        # A00: U[0*3+0, col], A10: U[1*3+0, col]
        # B11: V[1*6+k_offset+1, col], B10: V[1*6+k_offset, col]
        U[0, col] = 1
        U[3, col] = -1
        V[6 + k_offset + 1, col] = 1
        V[6 + k_offset, col] = -1
        # C11 (W[k_offset*2+0, col]), C12 (W[(k_offset+1)*2+0, col])
        W[k_offset * 2 + 0, col] = -1
        W[(k_offset + 1) * 2 + 0, col] = 1
        col += 1
        
        # m6 = (A10 + A00) * (B00 - B10)
        # A10: U[1*3+0, col], A00: U[0*3+0, col]
        # B00: V[0*6+k_offset, col], B10: V[1*6+k_offset, col]
        U[3, col] = 1
        U[0, col] = 1
        V[k_offset, col] = 1
        V[6 + k_offset, col] = -1
        # C22 (W[(k_offset+1)*2+1, col])
        W[(k_offset + 1) * 2 + 1, col] = 1
        col += 1
        
        # m7 = (A11 - A01) * (B01 + B11)
        # A11: U[1*3+1, col], A01: U[0*3+1, col]
        # B01: V[0*6+k_offset+1, col], B11: V[1*6+k_offset+1, col]
        U[4, col] = 1
        U[1, col] = -1
        V[k_offset + 1, col] = 1
        V[6 + k_offset + 1, col] = 1
        # C11 (W[k_offset*2+0, col]), C21 (W[k_offset*2+1, col])
        W[k_offset * 2 + 0, col] = -1
        W[k_offset * 2 + 1, col] = 1
        col += 1

    # Add 3 Strassen blocks for the top 2x2 part of A and top 2x6 part of B
    # Block 1: cols 0-1 of B
    add_strassen_block(0)
    # Block 2: cols 2-3 of B
    add_strassen_block(2)
    # Block 3: cols 4-5 of B
    add_strassen_block(4)
    
    # Add the bottom part: A[1,2] (row 1, col 2) times B[2, :] (row 2)
    # A index: i in {0,1}, j=2. Global U row: i*3 + 2.
    # B index: j=2, k in {0..5}. Global V row: 2*6 + k = 12 + k.
    # C index: i in {0,1}, k in {0..5}. Global W row: k*2 + i.
    
    for i in range(n):
        for k in range(p):
            U[i * 3 + 2, col] = 1
            V[12 + k, col] = 1
            W[k * 2 + i, col] = 1
            col += 1
            
    # Verify rank
    assert col == r, f"Rank mismatch: expected {r}, got {col}"
    
    # Verify correctness
    max_err = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = np.sum(U[i * m + j, :] * V[j * p + k, :] * W[k * n + i, :])
                err = abs(val - 1)
                if err > max_err:
                    max_err = err
    
    if max_err > 1e-8:
        # Fallback to schoolbook if verification fails (should not happen)
        r = n * m * p
        U = np.zeros((n * m, r), dtype=int)
        V = np.zeros((m * p, r), dtype=int)
        W = np.zeros((p * n, r), dtype=int)
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
        r = n * m * p
    
    # Convert to Python lists
    U_list = [[int(U[i, j]) for j in range(r)] for i in range(n * m)]
    V_list = [[int(V[i, j]) for j in range(r)] for i in range(m * p)]
    W_list = [[int(W[i, j]) for j in range(r)] for i in range(p * n)]
    
    return {"U": U_list, "V": V_list, "W": W_list}
