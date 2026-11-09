import time
import numpy as np

def construct():
    deadline = time.time() + 115.0
    n, m, p = 4, 4, 6
    
    # Build 4x4x4 Strassen decomposition (rank 49) for left 4 cols of B
    U_left, V_left, W_left = _strassen_4x4x4()
    
    # Build 4x4x2 decomposition (rank 32) for right 2 cols of B
    U_right, V_right, W_right = _schoolbook_4x4x2()
    
    # Combine: total rank = 49 + 32 = 81
    rank = 49 + 32
    U = np.zeros((16, rank))
    V = np.zeros((24, rank))
    W = np.zeros((24, rank))
    
    # Left part (cols 0-3 of B): V index = j*6 + k where k in 0..3
    # The _strassen_4x4x4 returns V with indices j*4+k (for 4x4 B)
    # We need to remap: for k in 0..3, V[j*4+k] -> V[j*6+k]
    for j in range(4):
        for k in range(4):
            V[j * 6 + k, :49] = V_left[j * 4 + k, :49]
    
    # W for left part: W[k*4+i] where k in 0..3 -> W[k*4+i] (same, since p*n=24, k*4+i for k in 0..3 is 0..15)
    W[:, :49] = W_left
    
    # Right part (cols 4-5 of B)
    U[:, 49:] = U_right
    
    # V for right part: B[j, k] where k in 4..5, so V index = j*6 + k
    # _schoolbook_4x4x2 returns V with indices j*2+k (for 4x2 B, k in 0..1)
    # Remap: V_right[j*2+k] -> V[j*6 + (4+k)]
    for j in range(4):
        for k in range(2):
            V[j * 6 + (4 + k), 49:] = V_right[j * 2 + k, :]
    
    # W for right part: C[i, k] where k in 4..5, so W index = k*4 + i
    # _schoolbook_4x4x2 returns W with indices k*4+i (for 2x4 C, k in 0..1)
    # Remap: W_right[k*4+i] -> W[(4+k)*4 + i]
    for k in range(2):
        for i in range(4):
            W[(4 + k) * 4 + i, 49:] = W_right[k * 4 + i, :]
    
    return {
        "U": [list(map(float, row)) for row in U],
        "V": [list(map(float, row)) for row in V],
        "W": [list(map(float, row)) for row in W]
    }

def _strassen_2x2x2():
    """
    Returns U, V, W for 2x2x2 matrix multiplication using Strassen (rank 7).
    Indexing: A[i,j] -> i*2+j, B[i,j] -> i*2+j, C[i,j] -> i*2+j
    """
    # Strassen formulas:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    #
    # C00 = M1 - M4 + M6
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M3 + M7
    
    M_A = [
        [1, 0, 0, 1],   # M1: A00 + A11
        [0, 0, 1, 1],   # M2: A10 + A11
        [1, 0, 0, 0],   # M3: A00
        [0, 0, 0, 1],   # M4: A11
        [1, 1, 0, 0],   # M5: A00 + A01
        [-1, 0, 1, 0],  # M6: A10 - A00
        [0, 1, 0, -1],  # M7: A01 - A11
    ]
    
    M_B = [
        [1, 0, 0, 1],   # M1: B00 + B11
        [1, 0, 0, 0],   # M2: B00
        [0, 1, 0, -1],  # M3: B01 - B11
        [0, 0, 1, -1],  # M4: B10 - B00
        [0, 0, 0, 1],   # M5: B11
        [1, 1, 0, 0],   # M6: B00 + B01
        [0, 0, 1, 1],   # M7: B10 + B11
    ]
    
    M_C = [
        [1, 0, 0, -1, 0, 1, 0],   # C00 = M1 - M4 + M6
        [0, 0, 1, 0, 1, 0, 0],    # C01 = M3 + M5
        [0, 1, 0, 1, 0, 0, 0],    # C10 = M2 + M4
        [1, 0, -1, 0, 0, 0, 1],   # C11 = M1 - M3 + M7
    ]
    
    U = np.array(M_A, dtype=float)  # (7, 4)
    V = np.array(M_B, dtype=float)  # (7, 4)
    W = np.array(M_C, dtype=float)  # (4, 7)
    
    return U, V, W

def _strassen_4x4x4():
    """
    Returns U, V, W for 4x4x4 matrix multiplication using Strassen (rank 49).
    Indexing: A[i,j] -> i*4+j, B[i,j] -> i*4+j, C[i,j] -> i*4+j
    """
    U_full = np.zeros((16, 49))
    V_full = np.zeros((16, 49))
    W_full = np.zeros((16, 49))
    
    def a_idx(bi, bj, i, j):
        return (2*bi + i) * 4 + (2*bj + j)
    
    def b_idx(bi, bj, i, j):
        return (2*bi + i) * 4 + (2*bj + j)
    
    def c_idx(bi, bj, i, j):
        return (2*bi + i) * 4 + (2*bj + j)
    
    M_defs = [
        ({(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}),
        ({(1,0): 1, (1,1): 1}, {(0,0): 1}, {(1,0): 1}),
        ({(0,0): 1}, {(0,1): 1, (1,1): -1}, {(0,1): 1}),
        ({(1,1): 1}, {(1,0): 1, (0,0): -1}, {(0,0): -1, (1,0): 1}),
        ({(0,0): 1, (0,1): 1}, {(1,1): 1}, {(0,1): 1}),
        ({(1,0): 1, (0,0): -1}, {(0,0): 1, (0,1): 1}, {(0,0): 1}),
        ({(0,1): 1, (1,1): -1}, {(1,0): 1, (1,1): 1}, {(1,1): 1}),
    ]
    
    U2, V2, W2 = _strassen_2x2x2()
    
    col = 0
    for m_idx in range(7):
        A_blocks, B_blocks, C_blocks = M_defs[m_idx]
        
        for r in range(7):
            for a_elem in range(4):
                coeff_a = U2[r, a_elem]
                if coeff_a == 0:
                    continue
                i = a_elem // 2
                j = a_elem % 2
                for (bi, bj), c in A_blocks.items():
                    full_idx = a_idx(bi, bj, i, j)
                    U_full[full_idx, col] += coeff_a * c
            
            for b_elem in range(4):
                coeff_b = V2[r, b_elem]
                if coeff_b == 0:
                    continue
                i = b_elem // 2
                j = b_elem % 2
                for (bi, bj), c in B_blocks.items():
                    full_idx = b_idx(bi, bj, i, j)
                    V_full[full_idx, col] += coeff_b * c
            
            for c_elem in range(4):
                coeff_c = W2[c_elem, r]
                if coeff_c == 0:
                    continue
                ci = c_elem // 2
                cj = c_elem % 2
                for (bi, bj), c in C_blocks.items():
                    full_idx = c_idx(bi, bj, ci, cj)
                    W_full[full_idx, col] += coeff_c * c
            
            col += 1
    
    return U_full, V_full, W_full

def _schoolbook_4x4x2():
    """
    Returns U, V, W for 4x4 * 4x2 matrix multiplication (schoolbook, rank 32).
    U: (16, 32), V: (8, 32), W: (8, 32)
    Indexing: A[i,j] -> i*4+j, B[i,k] -> i*2+k, C[i,k] -> i*2+k
    """
    n, m, p = 4, 4, 2
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1.0
                V[j * p + k, col] = 1.0
                W[k * n + i, col] = 1.0
                col += 1
    
    return U, V, W
