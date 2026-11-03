"""Rewritten program: Strassen-based decomposition for 4x4 * 4x9 multiplication."""
DIMENSIONS = [4, 4, 9]

def _int_or_half(x):
    # Ensure values are integers or half-integers; return as int if whole
    if x == int(x):
        return int(x)
    return x

def _build_strassen_4x4x4():
    """
    Build rank-49 Strassen decomposition for 4x4 * 4x4 -> 4x4.
    Returns U (16x49), V (16x49), W (16x49) where:
      U[i*4+j, col] = coefficient of A[i,j]
      V[j*4+k, col] = coefficient of B[j,k]
      W[k*4+i, col] = coefficient of C[i,k]
    Uses 2x2x2 Strassen recursively with 7 top-level and 7 bottom-level terms.
    """
    # 2x2x2 Strassen: 7 terms
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B10-B11)
    # M4 = A11(B01-B10)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B10)
    # M7 = (A01-A11)(B01+B11)
    # C00 = M1+M4-M5+M7
    # C01 = M3+M5
    # C10 = M2+M4
    # C11 = M1-M2+M3+M6
    
    # For 4x4x4, partition into 2x2 blocks of 2x2.
    # Top-level: 7 combinations of block-level operations
    # Each block operation is a 2x2x2 Strassen (7 terms)
    # Total: 7*7 = 49 terms
    
    # Define 2x2x2 Strassen vectors (u, v, w) for each of 7 terms
    # u: coefficients for A elements [00, 01, 10, 11]
    # v: coefficients for B elements [00, 01, 10, 11]
    # w: coefficients for C elements [00, 01, 10, 11]
    s2_u = [
        [1, 0, 0, 1],   # M1: (A00+A11)
        [0, 0, 1, 1],   # M2: (A10+A11)
        [1, 0, 0, 0],   # M3: A00
        [0, 0, 0, 1],   # M4: A11
        [1, 1, 0, 0],   # M5: (A00+A01)
        [-1, 0, 1, 0],  # M6: (A10-A00)
        [0, 1, 0, -1],  # M7: (A01-A11)
    ]
    s2_v = [
        [1, 0, 0, 1],   # M1: (B00+B11)
        [1, 0, 0, 0],   # M2: B00
        [0, -1, 1, 0],  # M3: (B10-B11)
        [0, 1, -1, 0],  # M4: (B01-B10)
        [0, 0, 0, 1],   # M5: B11
        [1, 0, 1, 0],   # M6: (B00+B10)
        [0, 1, 0, 1],   # M7: (B01+B11)
    ]
    s2_w = [
        [1, 0, 0, 1],   # M1 contributes to C00 and C11
        [0, 0, 1, 0],   # M2 contributes to C10
        [0, 1, 0, 0],   # M3 contributes to C01
        [1, 0, 0, 0],   # M4 contributes to C00
        [-1, 0, 0, 1],  # M5 contributes to C00 and C11
        [0, 0, 0, 1],   # M6 contributes to C11
        [1, 0, 0, 0],   # M7 contributes to C00
    ]
    
    # Top-level 7 operations on 2x2 blocks
    # Block indices: 00, 01, 10, 11
    # Each top-level term k defines:
    #   alpha_k: linear combo of A-blocks (4 coefficients)
    #   beta_k: linear combo of B-blocks (4 coefficients)
    #   gamma_k: linear combo of C-blocks (4 coefficients)
    
    top_u = [
        [1, 0, 0, 1],   # X1 = A00 + A11
        [0, 0, 1, 1],   # X2 = A10 + A11
        [1, 0, 0, 0],   # X3 = A00
        [0, 0, 0, 1],   # X4 = A11
        [1, 1, 0, 0],   # X5 = A00 + A01
        [-1, 0, 1, 0],  # X6 = A10 - A00
        [0, 1, 0, -1],  # X7 = A01 - A11
    ]
    top_v = [
        [1, 0, 0, 1],   # Y1 = B00 + B11
        [1, 0, 0, 0],   # Y2 = B00
        [0, -1, 1, 0],  # Y3 = B10 - B11
        [0, 1, -1, 0],  # Y4 = B01 - B10
        [0, 0, 0, 1],   # Y5 = B11
        [1, 0, 1, 0],   # Y6 = B00 + B10
        [0, 1, 0, 1],   # Y7 = B01 + B11
    ]
    top_w = [
        [1, 0, 0, 1],   # Z1 -> C00 + C11
        [0, 0, 1, 0],   # Z2 -> C10
        [0, 1, 0, 0],   # Z3 -> C01
        [1, 0, 0, 0],   # Z4 -> C00
        [-1, 0, 0, 1],  # Z5 -> C00 + C11
        [0, 0, 0, 1],   # Z6 -> C11
        [1, 0, 0, 0],   # Z7 -> C00
    ]
    
    n_blocks = 2  # 2x2 blocks
    rank = 49
    U = [[0] * rank for _ in range(16)]
    V = [[0] * rank for _ in range(16)]
    W = [[0] * rank for _ in range(16)]
    
    col = 0
    for k in range(7):
        for r in range(7):
            # U column: for each (i,j) in 4x4, determine block and local pos
            for i in range(4):
                for j in range(4):
                    bi = i // 2  # block row
                    bj = j // 2  # block col
                    li = i % 2   # local row
                    lj = j % 2   # local col
                    block_idx = bi * 2 + bj
                    coeff = top_u[k][block_idx] * s2_u[r][li * 2 + lj]
                    U[i * 4 + j][col] = coeff
            
            # V column: for each (j,k) in 4x4
            for j in range(4):
                for k2 in range(4):
                    bj = j // 2
                    bk = k2 // 2
                    lj = j % 2
                    lk = k2 % 2
                    block_idx = bj * 2 + bk
                    coeff = top_v[k][block_idx] * s2_v[r][lj * 2 + lk]
                    V[j * 4 + k2][col] = coeff
            
            # W column: for each (k,i) in 4x4 (output index k*n+i)
            for k2 in range(4):
                for i in range(4):
                    bk = k2 // 2
                    bi = i // 2
                    lk = k2 % 2
                    li = i % 2
                    block_idx = bk * 2 + bi
                    coeff = top_w[k][block_idx] * s2_w[r][lk * 2 + li]
                    W[k2 * 4 + i][col] = coeff
            
            col += 1
    
    # Convert to int where possible
    for row in U:
        for idx in range(len(row)):
            row[idx] = _int_or_half(row[idx])
    for row in V:
        for idx in range(len(row)):
            row[idx] = _int_or_half(row[idx])
    for row in W:
        for idx in range(len(row)):
            row[idx] = _int_or_half(row[idx])
    
    return U, V, W

def _build_strassen_4x4x1():
    """
    Build rank-16 decomposition for 4x4 * 4x1 -> 4x1.
    This is just schoolbook since one dimension is 1.
    Returns U (16x16), V (4x16), W (4x16)
    U[i*4+j, col] for A[i,j]
    V[j*1+k, col] -> V[j, col] for B[j,0]
    W[k*4+i, col] -> W[i, col] for C[i,0]
    """
    n, m, p = 4, 4, 1
    rank = n * m * p  # 16
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    
    return U, V, W

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 4, 4, 9
    
    # Split p=9 into 4+4+1
    # Three sub-problems:
    # 1. 4x4 * 4x4 -> 4x4 (output cols 0-3), rank 49
    # 2. 4x4 * 4x4 -> 4x4 (output cols 4-7), rank 49
    # 3. 4x4 * 4x1 -> 4x1 (output col 8), rank 16
    # Total rank: 49 + 49 + 16 = 114
    
    total_rank = 114
    
    U_full = [[0] * total_rank for _ in range(n * m)]  # 16 x 114
    V_full = [[0] * total_rank for _ in range(m * p)]  # 36 x 114
    W_full = [[0] * total_rank for _ in range(p * n)]  # 36 x 114
    
    col_offset = 0
    
    # Sub-problem 1: 4x4 * 4x4, output cols 0-3
    U1, V1, W1 = _build_strassen_4x4x4()
    for i in range(16):
        for c in range(49):
            U_full[i][col_offset + c] = U1[i][c]
    for j in range(16):
        for c in range(49):
            # V index: j*p + k. For first 4 cols of B, j is 0..3, k is 0..3
            # V1[j][c] corresponds to B[j, k] where j is 0..15 for 4x4
            # In full V, row = j*4 + k. Here j_local is 0..3 (row of 4x4 B), k_local 0..3
            # V1 row index is j_local*4 + k_local
            V_full[j][col_offset + c] = V1[j][c]
    for idx in range(16):
        for c in range(49):
            # W1[idx] where idx = k*4 + i for 4x4 output
            # In full W, row = k*n + i = k*4 + i, k is 0..8
            # For cols 0-3, k is 0..3
            W_full[idx][col_offset + c] = W1[idx][c]
    
    col_offset += 49
    
    # Sub-problem 2: 4x4 * 4x4, output cols 4-7
    U2, V2, W2 = _build_strassen_4x4x4()
    for i in range(16):
        for c in range(49):
            U_full[i][col_offset + c] = U2[i][c]
    for j in range(16):
        for c in range(49):
            # V row: j_local*4 + k_local, but these are B rows 4..7
            # In full V, row = j_global*4 + k. j_global = 4 + j_local for j_local 0..3? No.
            # B is 4x9. Rows are 0..3, cols are 0..8.
            # Wait, V has shape (m*p, rank) = (4*9, rank) = (36, rank).
            # V index: j*p + k where j is 0..3 (row of B), k is 0..8 (col of B).
            # For sub-problem 2, we use B cols 4-7.
            # V2 row index: j_local*4 + k_local where j_local 0..3, k_local 0..3
            # In full V, this maps to row = j_local*9 + (4 + k_local)
            j_local = j // 4
            k_local = j % 4
            full_row = j_local * 9 + (4 + k_local)
            V_full[full_row][col_offset + c] = V2[j][c]
    for idx in range(16):
        for c in range(49):
            # W1 row: k_local*4 + i, k_local 0..3
            # Full W row: k_global*n + i = k_global*4 + i, k_global = 4 + k_local
            k_local = idx // 4
            i_local = idx % 4
            full_row = (4 + k_local) * 4 + i_local
            W_full[full_row][col_offset + c] = W2[idx][c]
    
    col_offset += 49
    
    # Sub-problem 3: 4x4 * 4x1, output col 8
    U3, V3, W3 = _build_strassen_4x4x1()
    for i in range(16):
        for c in range(16):
            U_full[i][col_offset + c] = U3[i][c]
    for j in range(4):
        for c in range(16):
            # V3 row: j*1 + 0 = j
            # Full V row: j*9 + 8
            full_row = j * 9 + 8
            V_full[full_row][col_offset + c] = V3[j][c]
    for idx in range(4):
        for c in range(16):
            # W3 row: k*n + i = 0*4 + i = i, k=0
            # But we need k=8 for col 8.
            # W3 was built for p=1, so W3 row = k*4 + i where k=0 only.
            # Actually W3 has shape (1*4, 16) = (4, 16).
            # W3[i][c] corresponds to C[i,0].
            # Full W row for C[i,8] is 8*4 + i = 32 + i.
            full_row = 8 * 4 + idx
            W_full[full_row][col_offset + c] = W3[idx][c]
    
    # Convert all to int where possible
    for row in U_full:
        for idx in range(total_rank):
            row[idx] = _int_or_half(row[idx])
    for row in V_full:
        for idx in range(total_rank):
            row[idx] = _int_or_half(row[idx])
    for row in W_full:
        for idx in range(total_rank):
            row[idx] = _int_or_half(row[idx])
    
    return {"U": U_full, "V": V_full, "W": W_full}
# EVOLVE-BLOCK-END
