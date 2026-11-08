"""Program for matrix multiplication 3x3x4 tensor decomposition."""
import numpy as np

DIMENSIONS = [3, 3, 4]

def construct():
    n, m, p = DIMENSIONS  # n=3, m=3, p=4
    # Strategy: Use Strassen-based decomposition for 3x3x2 blocks.
    # Split B into two 3x2 blocks. Each block uses 17 terms (Strassen + direct).
    # Total rank: 34.
    
    rank = 34
    U = np.zeros((n * m, rank), dtype=int)  # 9 x 34
    V = np.zeros((m * p, rank), dtype=int)  # 12 x 34
    W = np.zeros((p * n, rank), dtype=int)  # 12 x 34
    
    col = 0
    
    def add_strassen_block(a_rows, b_cols, c_rows, c_cols):
        """Add Strassen terms for a 2x2x2 sub-block."""
        nonlocal col
        # a_rows: [r0, r1], b_cols: [c0, c1], c_rows: [r0, r1], c_cols: [c0, c1]
        # A indices: a_rows[0]*m + b_cols[0], a_rows[0]*m + b_cols[1],
        #            a_rows[1]*m + b_cols[0], a_rows[1]*m + b_cols[1]
        # B indices: b_cols[0]*p + c_cols[0], b_cols[0]*p + c_cols[1],
        #            b_cols[1]*p + c_cols[0], b_cols[1]*p + c_cols[1]
        # C indices: c_cols[0]*n + c_rows[0], c_cols[0]*n + c_rows[1],
        #            c_cols[1]*n + c_rows[0], c_cols[1]*n + c_rows[1]
        
        a00 = a_rows[0] * m + b_cols[0]
        a01 = a_rows[0] * m + b_cols[1]
        a10 = a_rows[1] * m + b_cols[0]
        a11 = a_rows[1] * m + b_cols[1]
        
        b00 = b_cols[0] * p + c_cols[0]
        b01 = b_cols[0] * p + c_cols[1]
        b10 = b_cols[1] * p + c_cols[0]
        b11 = b_cols[1] * p + c_cols[1]
        
        c00 = c_cols[0] * n + c_rows[0]
        c01 = c_cols[0] * n + c_rows[1]
        c10 = c_cols[1] * n + c_rows[0]
        c11 = c_cols[1] * n + c_rows[1]
        
        # P1 = (a00+a11)(b00+b11)
        U[a00][col] = 1; U[a11][col] = 1
        V[b00][col] = 1; V[b11][col] = 1
        W[c00][col] = 1; W[c11][col] = 1
        col += 1
        
        # P2 = (a10+a11)(b00)
        U[a10][col] = 1; U[a11][col] = 1
        V[b00][col] = 1
        W[c01][col] = 1; W[c11][col] = -1
        col += 1
        
        # P3 = (a00)(b01-b11)
        U[a00][col] = 1
        V[b01][col] = 1; V[b11][col] = -1
        W[c10][col] = 1; W[c11][col] = 1
        col += 1
        
        # P4 = (a11)(b10-b00)
        U[a11][col] = 1
        V[b10][col] = 1; V[b00][col] = -1
        W[c00][col] = 1; W[c01][col] = 1
        col += 1
        
        # P5 = (a00+a01)(b11)
        U[a00][col] = 1; U[a01][col] = 1
        V[b11][col] = 1
        W[c00][col] = -1; W[c10][col] = 1
        col += 1
        
        # P6 = (a10-a00)(b00+b01)
        U[a10][col] = 1; U[a00][col] = -1
        V[b00][col] = 1; V[b01][col] = 1
        W[c11][col] = 1
        col += 1
        
        # P7 = (a01-a11)(b10+b11)
        U[a01][col] = 1; U[a11][col] = -1
        V[b10][col] = 1; V[b11][col] = 1
        W[c00][col] = 1
        col += 1
    
    def add_direct_term(a_idx, b_idx, c_idx):
        """Add a single direct multiplication term."""
        nonlocal col
        U[a_idx][col] = 1
        V[b_idx][col] = 1
        W[c_idx][col] = 1
        col += 1
    
    # Left block: columns 0,1 of B
    # Strassen for A[0:2,0:2] * B[0:2,0:2] -> C[0:2,0:2]
    add_strassen_block([0, 1], [0, 1], [0, 1], [0, 1])
    
    # Direct terms for A[0:2,2] * B[2,0:2] -> C[0:2,0:2]
    for i in range(2):
        for j in range(2):
            a_idx = i * m + 2
            b_idx = 2 * p + j
            c_idx = j * n + i
            add_direct_term(a_idx, b_idx, c_idx)
    
    # Direct terms for A[2,0:2] * B[0:2,0:2] -> C[2,0:2]
    for i in range(2):
        for j in range(2):
            a_idx = 2 * m + i
            b_idx = i * p + j
            c_idx = j * n + 2
            add_direct_term(a_idx, b_idx, c_idx)
    
    # Direct terms for A[2,2] * B[2,0:2] -> C[2,0:2]
    for j in range(2):
        a_idx = 2 * m + 2
        b_idx = 2 * p + j
        c_idx = j * n + 2
        add_direct_term(a_idx, b_idx, c_idx)
    
    # Right block: columns 2,3 of B
    # Strassen for A[0:2,0:2] * B[0:2,2:4] -> C[0:2,2:4]
    add_strassen_block([0, 1], [0, 1], [0, 1], [2, 3])
    
    # Direct terms for A[0:2,2] * B[2,2:4] -> C[0:2,2:4]
    for i in range(2):
        for j in range(2):
            a_idx = i * m + 2
            b_idx = 2 * p + (2 + j)
            c_idx = (2 + j) * n + i
            add_direct_term(a_idx, b_idx, c_idx)
    
    # Direct terms for A[2,0:2] * B[0:2,2:4] -> C[2,2:4]
    for i in range(2):
        for j in range(2):
            a_idx = 2 * m + i
            b_idx = i * p + (2 + j)
            c_idx = (2 + j) * n + 2
            add_direct_term(a_idx, b_idx, c_idx)
    
    # Direct terms for A[2,2] * B[2,2:4] -> C[2,2:4]
    for j in range(2):
        a_idx = 2 * m + 2
        b_idx = 2 * p + (2 + j)
        c_idx = (2 + j) * n + 2
        add_direct_term(a_idx, b_idx, c_idx)
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
