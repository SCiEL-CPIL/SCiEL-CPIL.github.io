import numpy as np
import time

DIMENSIONS = [3, 3, 7]

def _verify(U, V, W, n, m, p):
    """Verify that U, V, W give an exact tensor decomposition of the matrix multiplication tensor."""
    rank = len(U[0]) if U else 0
    if len(U) != n * m or len(V) != m * p or len(W) != p * n:
        return False
    for i in range(len(U)):
        if len(U[i]) != rank:
            return False
    for i in range(len(V)):
        if len(V[i]) != rank:
            return False
    for i in range(len(W)):
        if len(W[i]) != rank:
            return False
    
    U_arr = np.array(U, dtype=np.float64)
    V_arr = np.array(V, dtype=np.float64)
    W_arr = np.array(W, dtype=np.float64)
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = np.dot(U_arr[i*m+j], V_arr[j*p+k] * W_arr[k*n+i])
                if abs(s - 1.0) > 1e-10:
                    return False
    return True

def _schoolbook(n, m, p):
    """Return the schoolbook (rank n*m*p) decomposition."""
    rank = n * m * p
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

def _strassen_2x2():
    """
    Return the rank-7 Strassen decomposition for 2x2 * 2x2.
    U has 4 rows, V has 4 rows, W has 4 rows, each with 7 columns.
    Indexing: U[i*2+j], V[j*2+k], W[k*2+i]
    """
    # Strassen's 7 products
    # P1 = A00 * (B01 - B11)
    # P2 = (A00 + A01) * B11
    # P3 = (A10 + A11) * B00
    # P4 = A11 * (B10 - B00)
    # P5 = (A00 + A11) * (B00 + B11)
    # P6 = (A01 - A11) * (B10 + B11)
    # P7 = (A00 + A10) * (B00 - B10)
    
    # C00 = P5 + P4 - P2 + P6
    # C01 = P1 + P5
    # C10 = P3 + P7
    # C11 = P5 - P1 + P3 - P7
    
    U = [[0]*7 for _ in range(4)]
    V = [[0]*7 for _ in range(4)]
    W = [[0]*7 for _ in range(4)]
    
    # Term 1
    U[0][0] = 1 # A00
    V[1][0] = 1 # B01
    V[3][0] = -1 # B11
    W[1][0] = 1 # C01
    W[3][0] = -1 # C11
    
    # Term 2
    U[0][1] = 1 # A00
    U[1][1] = 1 # A01
    V[3][1] = 1 # B11
    W[0][1] = -1 # C00
    W[3][1] = 1 # C11
    
    # Term 3
    U[2][2] = 1 # A10
    U[3][2] = 1 # A11
    V[0][2] = 1 # B00
    W[2][2] = 1 # C10
    W[3][2] = 1 # C11
    
    # Term 4
    U[3][3] = 1 # A11
    V[2][3] = 1 # B10
    V[0][3] = -1 # B00
    W[0][3] = 1 # C00
    W[3][3] = -1 # C11
    
    # Term 5
    U[0][4] = 1 # A00
    U[3][4] = 1 # A11
    V[0][4] = 1 # B00
    V[3][4] = 1 # B11
    W[0][4] = 1 # C00
    W[1][4] = 1 # C01
    W[2][4] = 1 # C10
    W[3][4] = 1 # C11
    
    # Term 6
    U[1][5] = 1 # A01
    U[3][5] = -1 # A11
    V[2][5] = 1 # B10
    V[3][5] = 1 # B11
    W[0][5] = 1 # C00
    W[3][5] = -1 # C11
    
    # Term 7
    U[0][6] = 1 # A00
    U[2][6] = 1 # A10
    V[0][6] = 1 # B00
    V[2][6] = -1 # B10
    W[2][6] = 1 # C10
    W[3][6] = -1 # C11
    
    return U, V, W

def _construct_rank_62():
    """
    Construct a rank-62 decomposition for 3x3 * 3x7.
    Split B into B1 (3x3) and B2 (3x4).
    Use rank-26 for 3x3 * 3x3 (Strassen for 2x2 block + schoolbook for cross terms).
    Use schoolbook (rank 36) for 3x3 * 3x4.
    """
    n, m, p = 3, 3, 7
    U_total = [[0] * 62 for _ in range(n * m)]
    V_total = [[0] * 62 for _ in range(m * p)]
    W_total = [[0] * 62 for _ in range(p * n)]
    
    # Part 1: 3x3 * 3x3 (cols 0-2 of B)
    # Split A into A11 (2x2), A12 (2x1), A21 (1x2), A22 (1x1)
    # Split B1 into B11 (2x2), B12 (2x1), B21 (1x2), B22 (1x1)
    # Split C1 into C11 (2x2), C12 (2x1), C21 (1x2), C22 (1x1)
    
    # C11 = A11 B11 + A12 B21
    # C12 = A11 B12 + A12 B22
    # C21 = A21 B11 + A22 B21
    # C22 = A21 B12 + A22 B22
    
    col_offset = 0
    
    # 1. A11 B11 (2x2 * 2x2) -> Rank 7 (Strassen)
    U_s, V_s, W_s = _strassen_2x2()
    for i in range(4):
        for r in range(7):
            # Map 2x2 indices to 3x3
            # i is 0..3 for 2x2: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
            # 3x3 indices: 0:(0,0), 1:(0,1), 2:(0,2), 3:(1,0), 4:(1,1), 5:(1,2), 6:(2,0), 7:(2,1), 8:(2,2)
            # U index in 3x3:
            if i == 0: u_idx = 0 # A00
            elif i == 1: u_idx = 1 # A01
            elif i == 2: u_idx = 3 # A10
            elif i == 3: u_idx = 4 # A11
            
            # V index in 3x3 (for B1, cols 0-2):
            # V is 3x3. Indices: 0:(0,0), 1:(0,1), 2:(0,2), 3:(1,0), 4:(1,1), 5:(1,2), 6:(2,0), 7:(2,1), 8:(2,2)
            if i == 0: v_idx = 0 # B00
            elif i == 1: v_idx = 1 # B01
            elif i == 2: v_idx = 3 # B10
            elif i == 3: v_idx = 4 # B11
            
            # W index in 3x3 (for C1, cols 0-2):
            # W is 3x3. Indices: 0:(0,0), 1:(0,1), 2:(0,2), 3:(1,0), 4:(1,1), 5:(1,2), 6:(2,0), 7:(2,1), 8:(2,2)
            if i == 0: w_idx = 0 # C00
            elif i == 1: w_idx = 1 # C01
            elif i == 2: w_idx = 3 # C10
            elif i == 3: w_idx = 4 # C11
            
            U_total[u_idx][col_offset + r] = U_s[i][r]
            V_total[v_idx][col_offset + r] = V_s[i][r]
            W_total[w_idx][col_offset + r] = W_s[i][r]
    col_offset += 7
    
    # 2. A12 B21 (2x1 * 1x2) -> Rank 4 (Schoolbook)
    # A12 is rows 0,1 col 2. B21 is row 2 cols 0,1.
    # C12 is rows 0,1 cols 0,1? No.
    # A12 (2x1) * B21 (1x2) -> (2x2) contribution to C11?
    # Wait. C = A B.
    # C[i,k] = sum_j A[i,j] B[j,k].
    # A12 is A[0:2, 2]. B21 is B[2, 0:2].
    # Product A12 B21 contributes to C[0:2, 0:2] which is C11.
    # So this term contributes to C11.
    # My previous split:
    # C11 = A11 B11 + A12 B21. Correct.
    # C12 = A11 B12 + A12 B22.
    # C21 = A21 B11 + A22 B21.
    # C22 = A21 B12 + A22 B22.
    
    # So A12 B21 contributes to C11 (rows 0,1; cols 0,1).
    # A12 indices: 0:(0,2), 1:(1,2).
    # B21 indices: 0:(2,0), 1:(2,1).
    # C11 indices: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1).
    
    # Terms:
    # (0,2)*(2,0) -> (0,0)
    # (0,2)*(2,1) -> (0,1)
    # (1,2)*(2,0) -> (1,0)
    # (1,2)*(2,1) -> (1,1)
    
    terms_12_21 = [
        (2, 6, 0), # A02, B20, C00
        (2, 7, 1), # A02, B21, C01
        (5, 6, 3), # A12, B20, C10
        (5, 7, 4)  # A12, B21, C11
    ]
    for u_idx, v_idx, w_idx in terms_12_21:
        U_total[u_idx][col_offset] = 1
        V_total[v_idx][col_offset] = 1
        W_total[w_idx][col_offset] = 1
        col_offset += 1
    
    # 3. A11 B12 (2x2 * 2x1) -> Rank 4
    # A11 (2x2) * B12 (2x1) -> (2x1) contribution to C12.
    # A11 indices: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1).
    # B12 indices: 0:(0,2), 1:(1,2).
    # C12 indices: 0:(0,2), 1:(1,2).
    
    terms_11_12 = [
        (0, 2, 2), # A00, B02, C02
        (1, 4, 2), # A01, B12, C02
        (3, 2, 5), # A10, B02, C12
        (4, 4, 5)  # A11, B12, C12
    ]
    for u_idx, v_idx, w_idx in terms_11_12:
        U_total[u_idx][col_offset] = 1
        V_total[v_idx][col_offset] = 1
        W_total[w_idx][col_offset] = 1
        col_offset += 1
        
    # 4. A12 B22 (2x1 * 1x1) -> Rank 2
    # A12 (2x1) * B22 (1x1) -> (2x1) contribution to C12.
    # A12 indices: 0:(0,2), 1:(1,2).
    # B22 index: 0:(2,2).
    # C12 indices: 0:(0,2), 1:(1,2).
    
    terms_12_22 = [
        (2, 8, 2), # A02, B22, C02
        (5, 8, 5)  # A12, B22, C12
    ]
    for u_idx, v_idx, w_idx in terms_12_22:
        U_total[u_idx][col_offset] = 1
        V_total[v_idx][col_offset] = 1
        W_total[w_idx][col_offset] = 1
        col_offset += 1
        
    # 5. A21 B11 (1x2 * 2x2) -> Rank 4
    # A21 (1x2) * B11 (2x2) -> (1x2) contribution to C21.
    # A21 indices: 0:(2,0), 1:(2,1).
    # B11 indices: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1).
    # C21 indices: 0:(2,0), 1:(2,1).
    
    terms_21_11 = [
        (6, 0, 6), # A20, B00, C20
        (6, 1, 7), # A20, B01, C21
        (7, 0, 6), # A21, B00, C20
        (7, 1, 7)  # A21, B01, C21
    ]
    for u_idx, v_idx, w_idx in terms_21_11:
        U_total[u_idx][col_offset] = 1
        V_total[v_idx][col_offset] = 1
        W_total[w_idx][col_offset] = 1
        col_offset += 1
        
    # 6. A22 B21 (1x1 * 1x2) -> Rank 2
    # A22 (1x1) * B21 (1x2) -> (1x2) contribution to C21.
    # A22 index: 0:(2,2).
    # B21 indices: 0:(2,0), 1:(2,1).
    # C21 indices: 0:(2,0), 1:(2,1).
    
    terms_22_21 = [
        (8, 6, 6), # A22, B20, C20
        (8, 7, 7)  # A22, B21, C21
    ]
    for u_idx, v_idx, w_idx in terms_22_21:
        U_total[u_idx][col_offset] = 1
        V_total[v_idx][col_offset] = 1
        W_total[w_idx][col_offset] = 1
        col_offset += 1
        
    # 7. A21 B12 (1x2 * 2x1) -> Rank 2
    # A21 (1x2) * B12 (2x1) -> (1x1) contribution to C22.
    # A21 indices: 0:(2,0), 1:(2,1).
    # B12 indices: 0:(0,2), 1:(1,2).
    # C22 index: 0:(2,2).
    
    terms_21_12 = [
        (6, 2, 8), # A20, B02, C22
        (7, 4, 8)  # A21, B12, C22
    ]
    for u_idx, v_idx, w_idx in terms_21_12:
        U_total[u_idx][col_offset] = 1
        V_total[v_idx][col_offset] = 1
        W_total[w_idx][col_offset] = 1
        col_offset += 1
        
    # 8. A22 B22 (1x1 * 1x1) -> Rank 1
    # A22 (1x1) * B22 (1x1) -> (1x1) contribution to C22.
    
    terms_22_22 = [
        (8, 8, 8)  # A22, B22, C22
    ]
    for u_idx, v_idx, w_idx in terms_22_22:
        U_total[u_idx][col_offset] = 1
        V_total[v_idx][col_offset] = 1
        W_total[w_idx][col_offset] = 1
        col_offset += 1
        
    # Part 2: 3x3 * 3x4 (cols 3-6 of B)
    # Schoolbook for these 12 terms? No, 3x3 * 3x4 is 36 terms.
    # A is 3x3. B2 is 3x4.
    # For each i in 0..2, j in 0..2, k in 3..6:
    # Term: A[i,j] * B[j,k] -> C[i,k]
    
    for i in range(3):
        for j in range(3):
            for k in range(3, 7):
                u_idx = i * 3 + j
                v_idx = j * 7 + k
                w_idx = k * 3 + i
                U_total[u_idx][col_offset] = 1
                V_total[v_idx][col_offset] = 1
                W_total[w_idx][col_offset] = 1
                col_offset += 1
                
    return U_total, V_total, W_total

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140
    
    # Try rank 62
    U, V, W = _construct_rank_62()
    
    if _verify(U, V, W, n, m, p):
        return {"U": U, "V": V, "W": W}
    else:
        # Fallback to schoolbook
        U, V, W = _schoolbook(n, m, p)
        return {"U": U, "V": V, "W": W}
