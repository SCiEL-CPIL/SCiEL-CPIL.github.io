<<<<<<< SEARCH
import time
import random

def _schoolbook():
    n, m, p = 3, 5, 6
    rank = n * m * p
    # Initialize U, V, W as lists of lists of zeros
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # U[i*m + j, col] = 1
                U[i * m + j][col] = 1.0
                # V[j*p + k, col] = 1
                V[j * p + k][col] = 1.0
                # W[k*n + i, col] = 1
                W[k * n + i][col] = 1.0
                col += 1
                
    return {"U": U, "V": V, "W": W}
=======
import time
import random
import numpy as np

def _strassen_2x2x2():
    """Return rank-7 Strassen decomposition for 2x2x2 matrix multiplication.
    Returns U (4x7), V (4x7), W (4x7) where:
    U[i*2+j, r] for input A[i,j]
    V[j*2+k, r] for input B[j,k]
    W[k*2+i, r] for output C[i,k]
    """
    # Standard Strassen 7 products
    # S1 = (A00+A11)(B00+B11)
    # S2 = (A10+A11)B00
    # S3 = A00(B01-B11)
    # S4 = A11(B10-B00)
    # S5 = (A00+A01)B11
    # S6 = (A10-A00)(B00+B01)
    # S7 = (A01-A11)(B10+B11)
    
    # U: rows indexed by (i,j) -> i*2+j, columns by Strassen factors
    # A00=0, A01=1, A10=2, A11=3
    U = [[1, 0, 0, 0, 1, 0, 0],   # A00
         [0, 1, 1, 0, 1, 0, 0],   # A01
         [0, 0, 1, 0, 0, 1, 1],   # A10
         [1, 1, 0, 1, 0, 0, 0]]   # A11
    
    # V: rows indexed by (j,k) -> j*2+k, columns by Strassen factors
    # B00=0, B01=1, B10=2, B11=3
    V = [[1, 1, 0, 1, 0, 1, 1],   # B00
         [1, 0, 1, 0, 0, 0, 1],   # B01
         [0, 0, 0, 1, 1, 0, 0],   # B10
         [1, 1, 1, 0, 1, 1, 1]]   # B11
    
    # W: rows indexed by (k,i) -> k*2+i, columns by Strassen factors
    # C00=0, C01=1, C10=2, C11=3
    W = [[1, 0, 0, 0, 0, 0, 0],   # C00
         [0, 0, 1, 0, 1, 0, 0],   # C01
         [0, 1, 0, 1, 0, 1, 0],   # C10
         [1, 1, 0, 0, 0, 0, 1]]   # C11
    
    return U, V, W

def _schoolbook_block(n_block, m_block, p_block, offset_u, offset_v, offset_w, U, V, W, start_col):
    """Add schoolbook decomposition for a block to U, V, W.
    n_block x m_block x p_block
    offset_u: row offset in U (i*m + j indexing)
    offset_v: row offset in V (j*p + k indexing)  
    offset_w: row offset in W (k*n + i indexing)
    Returns new start_col.
    """
    col = start_col
    for i in range(n_block):
        for j in range(m_block):
            for k in range(p_block):
                U[offset_u + i * m_block + j][col] = 1.0
                V[offset_v + j * p_block + k][col] = 1.0
                W[offset_w + k * n_block + i][col] = 1.0
                col += 1
    return col

def _add_strassen_block(n_block, m_block, p_block, offset_u, offset_v, offset_w, U, V, W, start_col):
    """Add Strassen decomposition for a 2x2x2 block.
    Returns new start_col.
    """
    assert n_block == 2 and m_block == 2 and p_block == 2
    Su, Sv, Sw = _strassen_2x2x2()
    for r in range(7):
        for i in range(2):
            for j in range(2):
                U[offset_u + i * 2 + j][start_col + r] += Su[i * 2 + j][r]
        for j in range(2):
            for k in range(2):
                V[offset_v + j * 2 + k][start_col + r] += Sv[j * 2 + k][r]
        for k in range(2):
            for i in range(2):
                W[offset_w + k * 2 + i][start_col + r] += Sw[k * 2 + i][r]
    return start_col + 7

def _block_decomposition():
    """Construct rank-49 decomposition by partitioning 3x5x6 into blocks.
    
    Partition:
    - Rows 0-1 (i=0,1), Cols 0-1 (j=0,1), Outputs 0-1 (k=0,1): 2x2x2 -> Strassen (rank 7)
    - Rows 0-1 (i=0,1), Cols 0-1 (j=0,1), Outputs 2-5 (k=2,3,4,5): 2x2x4 -> schoolbook (rank 16)
    - Rows 0-1 (i=0,1), Cols 2-4 (j=2,3,4), Outputs 0-1 (k=0,1): 2x3x2 -> schoolbook (rank 12)
    - Row 2 (i=2), Cols 0-4 (j=0,1,2,3,4), Outputs 0-5 (k=0,1,2,3,4,5): 1x5x6 -> schoolbook (rank 30)
    
    Total: 7 + 16 + 12 + 30 = 65. Hmm, that's not 49.
    
    Let me reconsider the partition for rank 49:
    - 2x2x2: Strassen rank 7
    - 2x2x2: Strassen rank 7  
    - 1x2x2: schoolbook rank 4
    - 1x2x4: schoolbook rank 8
    - 1x5x6: schoolbook rank 30
    Total: 7+7+4+8+30 = 56. Still not 49.
    
    Let me try:
    - 2x2x2: Strassen rank 7
    - 2x3x2: schoolbook rank 12
    - 1x5x6: schoolbook rank 30
    Total: 7+12+30 = 49. Yes!
    
    Partition:
    - i=0,1; j=0,1; k=0,1: 2x2x2 -> Strassen (7)
    - i=0,1; j=2,3,4; k=0,1: 2x3x2 -> schoolbook (12)
    - i=2; j=0,1,2,3,4; k=0,1,2,3,4,5: 1x5x6 -> schoolbook (30)
    Total: 7+12+30 = 49
    """
    n, m, p = 3, 5, 6
    rank = 49
    
    U = [[0.0] * rank for _ in range(n * m)]  # 15 x 49
    V = [[0.0] * rank for _ in range(m * p)]  # 30 x 49
    W = [[0.0] * rank for _ in range(p * n)]  # 18 x 49
    
    col = 0
    
    # Block 1: i=0,1; j=0,1; k=0,1 -> 2x2x2 Strassen
    # offset_u: rows for i=0,1 and j=0,1 -> rows 0,1,2,3 in U (i*5+j)
    # offset_v: rows for j=0,1 and k=0,1 -> rows 0,1,6,7 in V (j*6+k)
    # offset_w: rows for k=0,1 and i=0,1 -> rows 0,1,3,4 in W (k*3+i)
    # Need to map the 2x2x2 Strassen to these non-contiguous rows
    
    # For Strassen, we need to handle the non-contiguous indexing carefully
    # U rows: i*5+j for i in {0,1}, j in {0,1} -> 0,1,5,6
    # V rows: j*6+k for j in {0,1}, k in {0,1} -> 0,1,6,7
    # W rows: k*3+i for k in {0,1}, i in {0,1} -> 0,1,3,4
    
    Su, Sv, Sw = _strassen_2x2x2()
    
    # Map local indices to global
    # Local (i,j) -> global U row: i*5 + j
    u_rows = [0, 1, 5, 6]  # (0,0), (0,1), (1,0), (1,1)
    # Local (j,k) -> global V row: j*6 + k
    v_rows = [0, 1, 6, 7]  # (0,0), (0,1), (1,0), (1,1)
    # Local (k,i) -> global W row: k*3 + i
    w_rows = [0, 1, 3, 4]  # (0,0), (0,1), (1,0), (1,1)
    
    for r in range(7):
        for li in range(2):
            for lj in range(2):
                local_u = li * 2 + lj
                global_u = u_rows[local_u]
                U[global_u][col + r] += Su[local_u][r]
        
        for lj in range(2):
            for lk in range(2):
                local_v = lj * 2 + lk
                global_v = v_rows[local_v]
                V[global_v][col + r] += Sv[local_v][r]
        
        for lk in range(2):
            for li in range(2):
                local_w = lk * 2 + li
                global_w = w_rows[local_w]
                W[global_w][col + r] += Sw[local_w][r]
    
    col += 7  # col is now 7
    
    # Block 2: i=0,1; j=2,3,4; k=0,1 -> 2x3x2 schoolbook (rank 12)
    # U rows: i*5+j for i in {0,1}, j in {2,3,4} -> 2,3,4, 7,8,9
    # V rows: j*6+k for j in {2,3,4}, k in {0,1} -> 12,13, 18,19, 24,25
    # W rows: k*3+i for k in {0,1}, i in {0,1} -> 0,1,3,4
    
    u_rows2 = [2, 3, 4, 7, 8, 9]  # (0,2),(0,3),(0,4),(1,2),(1,3),(1,4)
    v_rows2 = [12, 13, 18, 19, 24, 25]  # (2,0),(2,1),(3,0),(3,1),(4,0),(4,1)
    w_rows2 = [0, 1, 3, 4]  # (0,0),(0,1),(1,0),(1,1)
    
    # Schoolbook for 2x3x2
    for i in range(2):
        for j in range(3):
            for k in range(2):
                gu = u_rows2[i * 3 + j]
                gv = v_rows2[j * 2 + k]
                gw = w_rows2[k * 2 + i]
                U[gu][col] = 1.0
                V[gv][col] = 1.0
                W[gw][col] = 1.0
                col += 1
    
    # col is now 7 + 12 = 19
    
    # Block 3: i=2; j=0,1,2,3,4; k=0,1,2,3,4,5 -> 1x5x6 schoolbook (rank 30)
    # U rows: i*5+j for i=2, j in {0,1,2,3,4} -> 10,11,12,13,14
    # V rows: j*6+k for j in {0,1,2,3,4}, k in {0,1,2,3,4,5} -> 0-29
    # W rows: k*3+i for k in {0,1,2,3,4,5}, i=2 -> 2,5,8,11,14,17
    
    u_rows3 = [10, 11, 12, 13, 14]
    v_rows3 = list(range(30))  # 0 to 29
    w_rows3 = [2, 5, 8, 11, 14, 17]
    
    for j in range(5):
        for k in range(6):
            gu = u_rows3[j]
            gv = v_rows3[j * 6 + k]
            gw = w_rows3[k]
            U[gu][col] = 1.0
            V[gv][col] = 1.0
            W[gw][col] = 1.0
            col += 1
    
    # col should be 19 + 30 = 49 = rank
    
    assert col == rank, f"Column count {col} != rank {rank}"
    
    return {"U": U, "V": V, "W": W}
>>>>>>> REPLACE
