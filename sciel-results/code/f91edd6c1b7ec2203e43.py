import numpy as np

DIMENSIONS = [3, 5, 6]

def _strassen_2x2x2():
    """
    Returns U, V, W for 2x2x2 matrix multiplication with rank 7 (Strassen).
    U shape: (4, 7)
    V shape: (4, 7)
    W shape: (4, 7)
    """
    # Strassen formulas:
    # M1 = (A11 + A22) * (B11 + B22)
    # M2 = (A21 + A22) * B11
    # M3 = A11 * (B12 - B22)
    # M4 = A22 * (B21 - B11)
    # M5 = (A11 + A12) * B22
    # M6 = (A21 - A11) * (B11 + B12)
    # M7 = (A12 - A22) * (B21 + B22)
    #
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    # U indices: 0:A11, 1:A12, 2:A21, 3:A22
    # V indices: 0:B11, 1:B12, 2:B21, 3:B22
    # W indices: 0:C11, 1:C12, 2:C21, 3:C22
    
    U = np.zeros((4, 7), dtype=int)
    V = np.zeros((4, 7), dtype=int)
    W = np.zeros((4, 7), dtype=int)
    
    # M1
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    W[0, 0] = 1; W[3, 0] = 1
    
    # M2
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    W[2, 1] = 1; W[3, 1] = -1 # C21 = M2 + M4, C22 = M1 - M2 ... wait.
    # Let's re-verify Strassen C formulas.
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    # Correct W assignments:
    W[0, 0] = 1 # M1
    W[3, 0] = 1 # M1
    
    W[2, 1] = 1 # M2 (C21)
    W[3, 1] = -1 # M2 (C22)
    
    W[1, 2] = 1 # M3 (C12)
    W[3, 2] = 1 # M3 (C22)
    
    W[0, 3] = 1 # M4 (C11)
    W[2, 3] = 1 # M4 (C21)
    
    W[0, 4] = -1 # M5 (C11)
    W[1, 4] = 1 # M5 (C12)
    
    W[3, 5] = 1 # M6 (C22)
    
    W[0, 6] = 1 # M7 (C11)
    
    # U and V assignments for M5, M6, M7
    # M5 = (A11 + A12) * B22
    U[0, 4] = 1; U[1, 4] = 1
    V[3, 4] = 1
    
    # M6 = (A21 - A11) * (B11 + B12)
    U[2, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[1, 5] = 1
    
    # M7 = (A12 - A22) * (B21 + B22)
    U[1, 6] = 1; U[3, 6] = -1
    V[2, 6] = 1; V[3, 6] = 1
    
    return U, V, W

def _schoolbook_block(n, m, p):
    """
    Returns U, V, W for nxmxp matrix multiplication with rank n*m*p.
    """
    rank = n * m * p
    U = np.zeros((n * m, rank), dtype=int)
    V = np.zeros((m * p, rank), dtype=int)
    W = np.zeros((p * n, rank), dtype=int)
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    return U, V, W

def construct():
    n, m, p = DIMENSIONS
    # Splits
    # n=3 -> [2, 1]
    # m=5 -> [2, 2, 1]
    # p=6 -> [2, 2, 2]
    
    n_splits = [2, 1]
    m_splits = [2, 2, 1]
    p_splits = [2, 2, 2]
    
    U_blocks = []
    V_blocks = []
    W_blocks = []
    
    for i_idx, ni in enumerate(n_splits):
        for j_idx, mj in enumerate(m_splits):
            for k_idx, pk in enumerate(p_splits):
                if ni == 2 and mj == 2 and pk == 2:
                    U_b, V_b, W_b = _strassen_2x2x2()
                else:
                    U_b, V_b, W_b = _schoolbook_block(ni, mj, pk)
                
                U_blocks.append(U_b)
                V_blocks.append(V_b)
                W_blocks.append(W_b)
    
    U = np.hstack(U_blocks)
    V = np.hstack(V_blocks)
    W = np.hstack(W_blocks)
    
    return {"U": U, "V": V, "W": W}
