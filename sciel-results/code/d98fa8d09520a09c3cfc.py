import time
import numpy as np

def _verify_fast(U, V, W, n, m, p):
    """Vectorized verification using numpy."""
    rank = len(U[0])
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    U3 = U_arr.reshape(n, m, rank)
    V3 = V_arr.reshape(m, p, rank)
    W3 = W_arr.reshape(p, n, rank)
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = np.dot(U3[i,j,:], V3[j,k,:] * W3[k,i,:])
                if abs(val - 1.0) > 1e-9:
                    return False
    return True

def _is_valid_entry(x):
    try:
        x = float(x)
    except:
        return False
    if abs(x - round(x)) < 1e-9:
        return True
    if abs(2*x - round(2*x)) < 1e-9:
        return True
    return False

def _check_all_entries(U, V, W):
    for row in U:
        for x in row:
            if not _is_valid_entry(x):
                return False
    for row in V:
        for x in row:
            if not _is_valid_entry(x):
                return False
    for row in W:
        for x in row:
            if not _is_valid_entry(x):
                return False
    return True

def _schoolbook(n, m, p):
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}

def _strassen_2x2x2_factors():
    """
    Returns U, V, W for 2x2x2 matrix multiplication (Rank 7).
    """
    n, m, p = 2, 2, 2
    rank = 7
    U = [[0] * rank for _ in range(4)]
    V = [[0] * rank for _ in range(4)]
    W = [[0] * rank for _ in range(4)]
    
    # Term 0: M1 = (A00 + A11)(B00 + B11)
    U[0][0] = 1; U[3][0] = 1
    V[0][0] = 1; V[3][0] = 1
    W[0][0] = 1; W[3][0] = 1
    
    # Term 1: M2 = (A10 + A11)B00
    U[2][1] = 1; U[3][1] = 1
    V[0][1] = 1
    W[2][1] = 1; W[3][1] = -1
    
    # Term 2: M3 = A00(B10 - B11)
    U[0][2] = 1
    V[2][2] = 1; V[3][2] = -1
    W[1][2] = 1; W[3][2] = 1
    
    # Term 3: M4 = A11(B01 - B00)
    U[3][3] = 1
    V[1][3] = 1; V[0][3] = -1
    W[0][3] = 1; W[2][3] = 1
    
    # Term 4: M5 = (A00 + A01)B11
    U[0][4] = 1; U[1][4] = 1
    V[3][4] = 1
    W[0][4] = -1; W[1][4] = 1
    
    # Term 5: M6 = (A10 - A00)(B00 + B01)
    U[2][5] = 1; U[0][5] = -1
    V[0][5] = 1; V[1][5] = 1
    W[0][5] = 1; W[2][5] = 1
    
    # Term 6: M7 = (A01 - A11)(B10 + B11)
    U[1][6] = 1; U[3][6] = -1
    V[2][6] = 1; V[3][6] = 1
    W[0][6] = -1; W[3][6] = 1
    
    return U, V, W

def _embed_factors(base_U, base_V, base_W, i_off, j_off, k_off, n, m, p, start_col):
    """
    Embeds smaller factors into larger ones.
    base_U is (n_sub*m_sub, rank_sub)
    """
    n_sub = len(base_U) // m_sub
    # This function is tricky because dimensions might differ.
    # Instead, I will construct the 3x3x3 explicitly.
    pass

def _construct_3x3x3_rank26():
    """
    Constructs a Rank-26 decomposition for 3x3x3.
    Uses Strassen (Rank 7) for the 2x2x2 block (i,j,k in {0,1}).
    Uses Schoolbook for the other 7 blocks.
    """
    n, m, p = 3, 3, 3
    rank = 26
    U = [[0] * rank for _ in range(9)]
    V = [[0] * rank for _ in range(9)]
    W = [[0] * rank for _ in range(9)]
    
    # Block (0,0,0) to (1,1,1): Strassen
    sU, sV, sW = _strassen_2x2x2_factors()
    for c in range(7):
        for i in range(2):
            for j in range(2):
                U[i*3 + j][c] = sU[i*2 + j][c]
        for j in range(2):
            for k in range(2):
                V[j*3 + k][c] = sV[j*2 + k][c]
        for k in range(2):
            for i in range(2):
                W[k*3 + i][c] = sW[k*2 + i][c]
    
    # Remaining blocks: Schoolbook
    # We need to cover all (i,j,k) not in {0,1}x{0,1}x{0,1}
    # Actually, the blocks are defined by ranges.
    # Block 0: i in 0-1, j in 0-1, k in 0-1 (Done)
    # Block 1: i in 0-1, j in 0-1, k in 2
    # Block 2: i in 0-1, j in 2, k in 0-1
    # Block 3: i in 0-1, j in 2, k in 2
    # Block 4: i in 2, j in 0-1, k in 0-1
    # Block 5: i in 2, j in 0-1, k in 2
    # Block 6: i in 2, j in 2, k in 0-1
    # Block 7: i in 2, j in 2, k in 2
    
    col = 7
    def add_schoolbook(i_range, j_range, k_range):
        nonlocal col
        for i in i_range:
            for j in j_range:
                for k in k_range:
                    U[i*3 + j][col] = 1
                    V[j*3 + k][col] = 1
                    W[k*3 + i][col] = 1
                    col += 1

    add_schoolbook([0,1], [0,1], [2]) # 2x2x1 = 4 terms
    add_schoolbook([0,1], [2], [0,1]) # 2x1x2 = 4 terms
    add_schoolbook([0,1], [2], [2])   # 2x1x1 = 2 terms
    add_schoolbook([2], [0,1], [0,1]) # 1x2x2 = 4 terms
    add_schoolbook([2], [0,1], [2])   # 1x2x1 = 2 terms
    add_schoolbook([2], [2], [0,1])   # 1x1x2 = 2 terms
    add_schoolbook([2], [2], [2])     # 1x1x1 = 1 term
    
    return U, V, W, col

def _construct_3x4x4_rank47():
    n, m, p = 3, 4, 4
    U_core, V_core, W_core, rank_core = _construct_3x3x3_rank26()
    
    # Strips:
    # 1. i in 0-2, j in 0-2, k=3 (3x3x1) -> 9 terms
    # 2. i in 0-2, j=3, k in 0-2 (3x1x3) -> 9 terms
    # 3. i in 0-2, j=3, k=3 (3x1x1) -> 3 terms
    # Total Rank: 26 + 9 + 9 + 3 = 47
    
    rank = 47
    U = [[0] * rank for _ in range(12)] # 3x4
    V = [[0] * rank for _ in range(16)] # 4x4
    W = [[0] * rank for _ in range(12)] # 4x3
    
    # Embed Core
    for c in range(rank_core):
        for i in range(3):
            for j in range(3):
                U[i*4 + j][c] = U_core[i*3 + j][c]
        for j in range(3):
            for k in range(3):
                V[j*4 + k][c] = V_core[j*3 + k][c]
        for k in range(3):
            for i in range(3):
                W[k*3 + i][c] = W_core[k*3 + i][c]
                
    col = rank_core
    
    # Strip 1: j in 0-2, k=3
    for i in range(3):
        for j in range(3):
            U[i*4 + j][col] = 1
            V[j*4 + 3][col] = 1
            W[3*3 + i][col] = 1 # k=3, i
            col += 1
            
    # Strip 2: j=3, k in 0-2
    for i in range(3):
        for k in range(3):
            U[i*4 + 3][col] = 1
            V[3*4 + k][col] = 1
            W[k*3 + i][col] = 1
            col += 1
            
    # Strip 3: j=3, k=3
    for i in range(3):
        U[i*4 + 3][col] = 1
        V[3*4 + 3][col] = 1
        W[3*3 + i][col] = 1
        col += 1
        
    return {"U": U, "V": V, "W": W}

def construct():
    n, m, p = DIMENSIONS
    # Try Rank 47
    try:
        result = _construct_3x4x4_rank47()
        if _verify_fast(result['U'], result['V'], result['W'], n, m, p):
            if _check_all_entries(result['U'], result['V'], result['W']):
                return result
    except Exception:
        pass
    
    # Fallback to Schoolbook
    return _schoolbook(n, m, p)
