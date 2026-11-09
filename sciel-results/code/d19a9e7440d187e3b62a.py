import time
import numpy as np

DIMENSIONS = [3, 5, 7]

def _schoolbook(n, m, p):
    """
    Returns U, V, W for schoolbook decomposition of n x m * m x p.
    U: (n*m, n*m*p)
    V: (m*p, n*m*p)
    W: (p*n, n*m*p)
    """
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    return U, V, W

def _strassen_222():
    """
    Returns U, V, W for Strassen decomposition of 2 x 2 * 2 x 2.
    Rank 7.
    U: (4, 7)
    V: (4, 7)
    W: (4, 7)
    """
    # Strassen coefficients
    # M1 = (A11 + A22)(B11 + B22)
    # M2 = (A21 + A22)B11
    # M3 = A11(B12 - B22)
    # M4 = A22(B21 - B11)
    # M5 = (A11 + A12)B22
    # M6 = (A21 - A11)(B11 + B12)
    # M7 = (A12 - A22)(B21 + B22)
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    # Indices for 2x2:
    # A: 00, 01, 10, 11 -> 0, 1, 2, 3
    # B: 00, 01, 10, 11 -> 0, 1, 2, 3
    # C: 00, 01, 10, 11 -> 0, 1, 2, 3
    
    U = np.zeros((4, 7))
    V = np.zeros((4, 7))
    W = np.zeros((4, 7))
    
    # M1
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    W[0, 0] = 1; W[3, 0] = 1 # C11, C22
    
    # M2
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    W[2, 1] = 1 # C21
    
    # M3
    U[0, 2] = 1
    V[1, 2] = 1; V[3, 2] = -1
    W[1, 2] = 1; W[3, 2] = 1 # C12, C22
    
    # M4
    U[3, 3] = 1
    V[2, 3] = 1; V[0, 3] = -1
    W[0, 3] = 1; W[2, 3] = 1 # C11, C21
    
    # M5
    U[0, 4] = 1; U[1, 4] = 1
    V[3, 4] = 1
    W[0, 4] = -1; W[1, 4] = 1 # C11, C12
    
    # M6
    U[2, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[1, 5] = 1
    W[3, 5] = 1 # C22
    
    # M7
    U[1, 6] = 1; U[3, 6] = -1
    V[2, 6] = 1; V[3, 6] = 1
    W[0, 6] = 1 # C11
    
    return U, V, W

def _combine_decompositions(decomps, n, m, p):
    """
    Combines a list of decompositions into one.
    decomps: list of tuples (U, V, W, offset_i, offset_j, offset_k)
    offset_i, offset_j, offset_k are starting indices for A, B, C.
    """
    total_rank = sum(U.shape[1] for U, V, W, _, _, _ in decomps)
    U_full = np.zeros((n * m, total_rank))
    V_full = np.zeros((m * p, total_rank))
    W_full = np.zeros((p * n, total_rank))
    
    col_offset = 0
    for U, V, W, off_i, off_j, off_k in decomps:
        r = U.shape[1]
        # Map local indices to global
        # Local U is (n_local * m_local, r)
        # Global U is (n * m, total_rank)
        # We need to place U into the correct block
        
        # This is tricky because the dimensions might be smaller.
        # We assume the decompositions are for sub-blocks.
        
        # For simplicity, we will construct the full matrices directly in construct().
        pass
    return U_full, V_full, W_full

def construct():
    n, m, p = DIMENSIONS # 3, 5, 7
    
    # Split A (3x5) into A1 (3x3) and A2 (3x2)
    # Split B (5x7) into B1 (3x7) and B2 (2x7)
    
    # A1 B1: 3x3 * 3x7
    # Split B1 (3x7) into B11 (3x3) and B12 (3x4)
    # A1 B11: 3x3 * 3x3 -> Schoolbook (27)
    # A1 B12: 3x3 * 3x4 -> Schoolbook (36)
    
    # A2 B2: 3x2 * 2x7
    # Split B2 (2x7) into B21 (2x4) and B22 (2x3)
    # A2 B21: 3x2 * 2x4 -> Improved (22)
    #   Split A2 (3x2) into A21 (2x2) and A22 (1x2)
    #   A21 B21: 2x2 * 2x4 -> 2x Strassen (14)
    #   A22 B21: 1x2 * 2x4 -> Schoolbook (8)
    # A2 B22: 3x2 * 2x3 -> Schoolbook (18)
    
    # Total Rank: 27 + 36 + 14 + 8 + 18 = 103
    
    total_rank = 103
    U = np.zeros((n * m, total_rank))
    V = np.zeros((m * p, total_rank))
    W = np.zeros((p * n, total_rank))
    
    col = 0
    
    # 1. A1 B11: 3x3 * 3x3 (Schoolbook)
    # A1 rows: 0-2, B11 cols: 0-2
    # B1 rows: 0-2, B11 cols: 0-2
    # C11 rows: 0-2, C11 cols: 0-2
    
    U_sub, V_sub, W_sub = _schoolbook(3, 3, 3)
    # Place U_sub into U[0:9, col:col+27]
    U[0:9, col:col+27] = U_sub
    # Place V_sub into V[0:9, col:col+27]
    V[0:9, col:col+27] = V_sub
    # Place W_sub into W[0:9, col:col+27]
    W[0:9, col:col+27] = W_sub
    col += 27
    
    # 2. A1 B12: 3x3 * 3x4 (Schoolbook)
    # A1 rows: 0-2, B12 cols: 3-6
    # B1 rows: 0-2, B12 cols: 3-6
    # C12 rows: 0-2, C12 cols: 3-6
    
    U_sub, V_sub, W_sub = _schoolbook(3, 3, 4)
    U[0:9, col:col+36] = U_sub
    # V_sub indices: j (0-2), k (0-3) -> global j (0-2), k (3-6)
    # V_sub is (3*4, 36) = (12, 36)
    # Global V is (5*7, 103) = (35, 103)
    # We need to map V_sub[j*4+k] to V[j*7 + (3+k)]
    V_mapped = np.zeros((35, 36))
    for j in range(3):
        for k in range(4):
            V_mapped[j*7 + 3 + k, :] = V_sub[j*4 + k, :]
    V[0:35, col:col+36] += V_mapped # Add to existing? No, disjoint columns.
    # W_sub indices: k (0-3), i (0-2) -> global k (3-6), i (0-2)
    W_mapped = np.zeros((21, 36))
    for k in range(4):
        for i in range(3):
            W_mapped[(3+k)*3 + i, :] = W_sub[k*3 + i, :]
    W[0:21, col:col+36] += W_mapped
    col += 36
    
    # 3. A2 B21: 3x2 * 2x4
    # A2 rows: 3-4, B21 cols: 0-3
    # B2 rows: 3-4, B21 cols: 0-3
    # C21 rows: 3-4, C21 cols: 0-3
    
    # 3a. A21 B21: 2x2 * 2x4 (2x Strassen)
    # A21 rows: 3-4, B21 cols: 0-3
    # B2 rows: 3-4, B21 cols: 0-3
    # C21 rows: 3-4, C21 cols: 0-3
    
    # Strassen 1 for cols 0-1
    U_s, V_s, W_s = _strassen_222()
    U[3:5, col:col+7] = U_s
    # V_s: j (0-1), k (0-1) -> global j (3-4), k (0-1)
    V_mapped = np.zeros((35, 7))
    for j in range(2):
        for k in range(2):
            V_mapped[(3+j)*7 + k, :] = V_s[j*2 + k, :]
    V[0:35, col:col+7] += V_mapped
    # W_s: k (0-1), i (0-1) -> global k (0-1), i (3-4)
    W_mapped = np.zeros((21, 7))
    for k in range(2):
        for i in range(2):
            W_mapped[k*3 + (3+i), :] = W_s[k*2 + i, :]
    W[0:21, col:col+7] += W_mapped
    col += 7
    
    # Strassen 2 for cols 2-3
    U_s, V_s, W_s = _strassen_222()
    U[3:5, col:col+7] = U_s
    V_mapped = np.zeros((35, 7))
    for j in range(2):
        for k in range(2):
            V_mapped[(3+j)*7 + (2+k), :] = V_s[j*2 + k, :]
    V[0:35, col:col+7] += V_mapped
    W_mapped = np.zeros((21, 7))
    for k in range(2):
        for i in range(2):
            W_mapped[(2+k)*3 + (3+i), :] = W_s[k*2 + i, :]
    W[0:21, col:col+7] += W_mapped
    col += 7
    
    # 3b. A22 B21: 1x2 * 2x4 (Schoolbook)
    # A22 row: 5, B21 cols: 0-3
    # B2 rows: 3-4, B21 cols: 0-3
    # C22 rows: 5, C22 cols: 0-3
    
    U_sub, V_sub, W_sub = _schoolbook(1, 2, 4)
    U[5:6, col:col+8] = U_sub
    V_mapped = np.zeros((35, 8))
    for j in range(2):
        for k in range(4):
            V_mapped[(3+j)*7 + k, :] = V_sub[j*4 + k, :]
    V[0:35, col:col+8] += V_mapped
    W_mapped = np.zeros((21, 8))
    for k in range(4):
        for i in range(1):
            W_mapped[k*3 + 5, :] = W_sub[k*1 + i, :]
    W[0:21, col:col+8] += W_mapped
    col += 8
    
    # 4. A2 B22: 3x2 * 2x3 (Schoolbook)
    # A2 rows: 3-5, B22 cols: 4-6
    # B2 rows: 3-4, B22 cols: 4-6
    # C22 rows: 3-5, C22 cols: 4-6
    
    U_sub, V_sub, W_sub = _schoolbook(3, 2, 3)
    U[3:6, col:col+18] = U_sub
    V_mapped = np.zeros((35, 18))
    for j in range(2):
        for k in range(3):
            V_mapped[(3+j)*7 + (4+k), :] = V_sub[j*3 + k, :]
    V[0:35, col:col+18] += V_mapped
    W_mapped = np.zeros((21, 18))
    for k in range(3):
        for i in range(3):
            W_mapped[(4+k)*3 + (3+i), :] = W_sub[k*3 + i, :]
    W[0:21, col:col+18] += W_mapped
    col += 18
    
    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
