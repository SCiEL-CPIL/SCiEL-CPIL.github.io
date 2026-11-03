import numpy as np
import time

DIMENSIONS = [4, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    # Target: 4x5x5 matrix multiplication
    # Strategy: Split m=5 into 1+4.
    # Sub-problems:
    # 1. 4x1x1 (Rank 4)
    # 2. 4x4x1 (Rank 16)
    # 3. 4x1x4 (Rank 16)
    # 4. 4x4x4 (Rank 56 via Strassen blocks)
    
    n, m, p = 4, 5, 5
    
    U_list = []
    V_list = []
    W_list = []
    
    # Helper to add a block decomposition
    def add_block(n_b, m_b, p_b, offset_i, offset_j, offset_k):
        # n_b x m_b x p_b
        # U shape: (n_b * m_b, rank_b)
        # V shape: (m_b * p_b, rank_b)
        # W shape: (p_b * n_b, rank_b)
        # Indices:
        # U row: i * m_b + j
        # V row: j * p_b + k
        # W row: k * n_b + i
        # Global indices:
        # U row: (offset_i + i) * m + (offset_j + j)
        # V row: (offset_j + j) * p + (offset_k + k)
        # W row: (offset_k + k) * n + (offset_i + i)
        
        # Get decomposition for this block
        Ub, Vb, Wb = get_decomposition(n_b, m_b, p_b)
        rank_b = len(Ub[0]) if Ub else 0
        
        # Create full sized matrices for this block
        U_full = [[0] * rank_b for _ in range(n * m)]
        V_full = [[0] * rank_b for _ in range(m * p)]
        W_full = [[0] * rank_b for _ in range(p * n)]
        
        for i in range(n_b):
            for j in range(m_b):
                u_idx = i * m_b + j
                gu_idx = (offset_i + i) * m + (offset_j + j)
                for c in range(rank_b):
                    U_full[gu_idx][c] = Ub[u_idx][c]
        
        for j in range(m_b):
            for k in range(p_b):
                v_idx = j * p_b + k
                gv_idx = (offset_j + j) * p + (offset_k + k)
                for c in range(rank_b):
                    V_full[gv_idx][c] = Vb[v_idx][c]
        
        for k in range(p_b):
            for i in range(n_b):
                w_idx = k * n_b + i
                gw_idx = (offset_k + k) * n + (offset_i + i)
                for c in range(rank_b):
                    W_full[gw_idx][c] = Wb[w_idx][c]
        
        U_list.append(U_full)
        V_list.append(V_full)
        W_list.append(W_full)

    # 1. 4x1x1 at (0,0,0)
    add_block(4, 1, 1, 0, 0, 0)
    
    # 2. 4x4x1 at (0,1,0) -> A rows 0-3, cols 1-4. B rows 1-4, col 0. C col 0.
    # A is 4x1? No.
    # C_col0 = A_col0 * B_00 + A_cols1-4 * B_rows1-4_col0
    # Term 1: 4x1x1 (Done)
    # Term 2: 4x4x1
    add_block(4, 4, 1, 0, 1, 0)
    
    # 3. 4x1x4 at (0,0,1) -> A col 0. B row 0, cols 1-4. C cols 1-4.
    # Term 3: 4x1x4
    add_block(4, 1, 4, 0, 0, 1)
    
    # 4. 4x4x4 at (0,1,1) -> A cols 1-4. B rows 1-4, cols 1-4. C cols 1-4.
    add_block(4, 4, 4, 0, 1, 1)
    
    # Concatenate
    total_rank = sum(len(U[0]) for U in U_list)
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    
    offset = 0
    for Ub, Vb, Wb in zip(U_list, V_list, W_list):
        rank_b = len(Ub[0])
        for i in range(n * m):
            for c in range(rank_b):
                U[i][offset + c] = Ub[i][c]
        for i in range(m * p):
            for c in range(rank_b):
                V[i][offset + c] = Vb[i][c]
        for i in range(p * n):
            for c in range(rank_b):
                W[i][offset + c] = Wb[i][c]
        offset += rank_b
        
    return {"U": U, "V": V, "W": W}

def get_decomposition(n, m, p):
    # Returns U, V, W for n x m x p
    if n == 1 and m == 1 and p == 1:
        return [[1]], [[1]], [[1]]
    
    if n == 1 and m == 1:
        # 1 x 1 x p
        # C_k = A_0 B_{0, k}
        rank = p
        U = [[1]] * rank
        V = [[1 if i == 0 else 0 for _ in range(rank)] for i in range(m * p)] # V is 1 x p
        # V row j*p+k. j=0. V[0*p+k] = 1 if k==c
        V = [[0]*rank for _ in range(m*p)]
        for k in range(p):
            V[k][k] = 1
        W = [[1 if i == 0 else 0 for _ in range(rank)] for i in range(p*n)] # W is p x 1
        # W row k*n+i. i=0. W[k*n+0] = 1 if k==c
        W = [[0]*rank for _ in range(p*n)]
        for k in range(p):
            W[k*n][k] = 1
        return U, V, W

    if n == 1 and p == 1:
        # 1 x m x 1
        # C_0 = sum A_j B_{j, 0}
        rank = m
        U = [[0]*rank for _ in range(n*m)]
        for j in range(m):
            U[j][j] = 1
        V = [[0]*rank for _ in range(m*p)]
        for j in range(m):
            V[j*p][j] = 1
        W = [[0]*rank for _ in range(p*n)]
        W[0] = [1]*rank
        return U, V, W

    if m == 1 and p == 1:
        # n x 1 x 1
        # C_0 = sum A_i B_{0, 0} ? No.
        # C_i = A_i B_{0, 0} ? No.
        # C is n x 1. C_i = A_{i, 0} B_{0, 0}.
        rank = n
        U = [[0]*rank for _ in range(n*m)]
        for i in range(n):
            U[i*m][i] = 1
        V = [[0]*rank for _ in range(m*p)]
        V[0] = [1]*rank
        W = [[0]*rank for _ in range(p*n)]
        for i in range(n):
            W[i*n][i] = 1
        return U, V, W

    if n == 2 and m == 2 and p == 2:
        return strassen_2x2x2()
        
    if n == 4 and m == 4 and p == 4:
        return strassen_4x4x4()

    # Fallback: Schoolbook
    rank = n * m * p
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j][col] = 1
                V[j*p+k][col] = 1
                W[k*n+i][col] = 1
                col += 1
    return U, V, W

def strassen_2x2x2():
    # Rank 7
    # M1 = (A00 + A11) * (B00 + B11)
    # M2 = (A10 + A11) * B00
    # M3 = A00 * (B01 - B11)
    # M4 = A11 * (B10 - B00)
    # M5 = (A00 + A01) * B11
    # M6 = (A10 - A00) * (B00 + B01)
    # M7 = (A01 - A11) * (B10 + B11)
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    rank = 7
    U = [[0]*rank for _ in range(4)]
    V = [[0]*rank for _ in range(4)]
    W = [[0]*rank for _ in range(4)]
    
    # U: A indices 00, 01, 10, 11 -> 0, 1, 2, 3
    # V: B indices 00, 01, 10, 11 -> 0, 1, 2, 3
    # W: C indices 00, 01, 10, 11 -> 0, 1, 2, 3
    
    # M1
    U[0][0] = 1; U[3][0] = 1
    V[0][0] = 1; V[3][0] = 1
    W[0][0] = 1; W[3][0] = 1
    
    # M2
    U[2][1] = 1; U[3][1] = 1
    V[0][1] = 1
    W[2][1] = 1
    
    # M3
    U[0][2] = 1
    V[1][2] = 1; V[3][2] = -1
    W[1][2] = 1
    
    # M4
    U[3][3] = 1
    V[2][3] = 1; V[0][3] = -1
    W[0][3] = 1; W[2][3] = 1
    
    # M5
    U[0][4] = 1; U[1][4] = 1
    V[3][4] = 1
    W[0][4] = -1; W[1][4] = 1
    
    # M6
    U[2][5] = 1; U[0][5] = -1
    V[0][5] = 1; V[1][5] = 1
    W[3][5] = 1
    
    # M7
    U[1][6] = 1; U[3][6] = -1
    V[2][6] = 1; V[3][6] = 1
    W[0][6] = 1; W[3][6] = -1
    
    return U, V, W

def strassen_4x4x4():
    # Use 8 Strassen blocks
    rank = 56
    U = [[0]*rank for _ in range(16)]
    V = [[0]*rank for _ in range(16)]
    W = [[0]*rank for _ in range(16)]
    
    # Blocks:
    # A11 A12
    # A21 A22
    # B11 B12
    # B21 B22
    # C11 = A11 B11 + A12 B21
    # C12 = A11 B12 + A12 B22
    # C21 = A21 B11 + A22 B21
    # C22 = A21 B12 + A22 B22
    
    # We compute 8 products:
    # P1 = A11 B11
    # P2 = A12 B21
    # P3 = A11 B12
    # P4 = A12 B22
    # P5 = A21 B11
    # P6 = A22 B21
    # P7 = A21 B12
    # P8 = A22 B22
    
    # Each Pi is 2x2x2 Strassen (Rank 7)
    # Total Rank 56
    
    # Indices in 4x4:
    # A: 0-3, 4-7, 8-11, 12-15
    # A11: 0,1,4,5
    # A12: 2,3,6,7
    # A21: 8,9,12,13
    # A22: 10,11,14,15
    # B: same
    # C: same
    
    def add_strassen_block(a_offset, b_offset, c_offset, rank_offset):
        Ub, Vb, Wb = strassen_2x2x2()
        for i in range(4):
            for c in range(7):
                U[a_offset + i][rank_offset + c] = Ub[i][c]
                V[b_offset + i][rank_offset + c] = Vb[i][c]
                W[c_offset + i][rank_offset + c] = Wb[i][c]
                
    # P1: A11(0) B11(0) -> C11(0)
    add_strassen_block(0, 0, 0, 0)
    # P2: A12(2) B21(8) -> C11(0)
    add_strassen_block(2, 8, 0, 7)
    # P3: A11(0) B12(2) -> C12(2)
    add_strassen_block(0, 2, 2, 14)
    # P4: A12(2) B22(10) -> C12(2)
    add_strassen_block(2, 10, 2, 21)
    # P5: A21(8) B11(0) -> C21(8)
    add_strassen_block(8, 0, 8, 28)
    # P6: A22(10) B21(8) -> C21(8)
    add_strassen_block(10, 8, 8, 35)
    # P7: A21(8) B12(2) -> C22(10)
    add_strassen_block(8, 2, 10, 42)
    # P8: A22(10) B22(10) -> C22(10)
    add_strassen_block(10, 10, 10, 49)
    
    return U, V, W

# EVOLVE-BLOCK-END
