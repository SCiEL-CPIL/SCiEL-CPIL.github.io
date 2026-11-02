import time
import numpy as np

DIMENSIONS = [4, 4, 9]

def _strassen_2x2():
    """Returns U, V, W for 2x2x2 Strassen decomposition (rank 7)."""
    U = [
        [1, 0, 0, 1],
        [0, 0, 1, 1],
        [1, 1, 0, 0],
        [1, 0, 0, 0],
        [0, 0, 0, 1],
        [1, 1, -1, -1],
        [-1, -1, 1, 1]
    ]
    V = [
        [1, 0, 0, 1],
        [0, 1, 0, 0],
        [0, 0, 1, 0],
        [1, 0, -1, 0],
        [0, 1, 0, -1],
        [0, 0, 0, 1],
        [1, 0, 0, 0]
    ]
    W = [
        [1, 0, 0, 1],
        [0, 0, 1, -1],
        [0, 0, 0, 1],
        [-1, 1, 0, 0],
        [1, 0, 1, 0],
        [0, 1, 0, 0],
        [-1, 0, 0, 1]
    ]
    # Note: U, V, W are 4x7 matrices (rows are variables, cols are products)
    # The lists above are 7x4. We need to transpose for standard matrix representation
    # where rows are indices and cols are rank.
    return [list(row) for row in zip(*U)], [list(row) for row in zip(*V)], [list(row) for row in zip(*W)]

def _kron_product(A, B):
    """Kronecker product of two matrices A (ra x ca) and B (rb x cb)."""
    ra, ca = len(A), len(A[0])
    rb, cb = len(B), len(B[0])
    result = [[0.0] * (ca * cb) for _ in range(ra * rb)]
    for i in range(ra):
        for j in range(ca):
            if A[i][j] != 0:
                for k in range(rb):
                    for l in range(cb):
                        result[i * rb + k][j * cb + l] = A[i][j] * B[k][l]
    return result

def _schoolbook_4x4x1():
    """Schoolbook decomposition for 4x4x1 (rank 16)."""
    n, m, p = 4, 4, 1
    rank = n * m * p
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1.0
                V[j * p + k][col] = 1.0
                W[k * n + i][col] = 1.0
                col += 1
    return U, V, W

def _verify_decomposition(U, V, W, n, m, p):
    """Verify that U, V, W exactly decompose the matrix multiplication tensor."""
    rank = len(U[0])
    for i in range(n):
        for j in range(m):
            for k in range(p):
                total = 0.0
                for c in range(rank):
                    total += U[i * m + j][c] * V[j * p + k][c] * W[k * n + i][c]
                expected = 1.0 if i == k else 0.0
                if abs(total - expected) > 1e-10:
                    return False
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 145.0
    
    # 1. Get Strassen 2x2x2 factors
    U2, V2, W2 = _strassen_2x2()
    
    # 2. Kronecker product for 4x4x4
    U4_49 = _kron_product(U2, U2)
    V4_49 = _kron_product(V2, V2)
    W4_49 = _kron_product(W2, W2)
    
    # 3. Schoolbook for 4x4x1
    U4_16, V4_16, W4_16 = _schoolbook_4x4x1()
    
    # 4. Assemble for 4x4x9
    # Rank = 49 + 49 + 16 = 114
    total_rank = 114
    
    U_full = [[0.0] * total_rank for _ in range(16)]
    V_full = [[0.0] * total_rank for _ in range(36)]
    W_full = [[0.0] * total_rank for _ in range(36)]
    
    # Part 1: 4x4x4 (cols 0-3 of B, rows 0-3 of C)
    for r in range(16):
        for c in range(49):
            U_full[r][c] = U4_49[r][c]
            V_full[r][c] = V4_49[r][c] # V rows 0-15 map to B cols 0-3
            W_full[r][c] = W4_49[r][c] # W rows 0-15 map to C rows 0-3
            
    # Part 2: 4x4x4 (cols 4-7 of B, rows 4-7 of C)
    offset = 49
    for r in range(16):
        for c in range(49):
            U_full[r][offset + c] = U4_49[r][c]
            # V index: j*p + k. Here j in 0..3, k in 4..7.
            # Local V index r' (0..15) corresponds to j, k_local.
            # Global V index = j*9 + k_global.
            # r' = j*4 + k_local.
            # k_global = k_local + 4.
            # So we need to map V4_49 rows to V_full rows.
            # V4_49 has 16 rows.
            # Row r' in V4_49 corresponds to j = r' // 4, k_local = r' % 4.
            # Global row = j * 9 + (k_local + 4).
            for c_local in range(49):
                val = V4_49[r][c_local]
                if val != 0:
                    j = r // 4
                    k_local = r % 4
                    k_global = k_local + 4
                    global_v_row = j * 9 + k_global
                    V_full[global_v_row][offset + c_local] = val
            
            # W index: k*n + i. Here k in 4..7, i in 0..3.
            # Local W index r' (0..15) corresponds to k_local, i.
            # r' = k_local * 4 + i.
            # Global W index = k_global * 4 + i.
            # k_global = k_local + 4.
            for c_local in range(49):
                val = W4_49[r][c_local]
                if val != 0:
                    k_local = r // 4
                    i = r % 4
                    k_global = k_local + 4
                    global_w_row = k_global * 4 + i
                    W_full[global_w_row][offset + c_local] = val

    # Part 3: 4x4x1 (col 8 of B, row 8 of C)
    offset = 98
    for r in range(16):
        for c in range(16):
            U_full[r][offset + c] = U4_16[r][c]
            # V4_16 has 4 rows (j*1+0 = j).
            # Global V row = j * 9 + 8.
            # r in V4_16 is 0..3, corresponding to j=0..3.
            # But U4_16 has 16 rows.
            # We need to map V4_16 rows (4) to V_full rows.
            # And W4_16 rows (4) to W_full rows.
            
            # It's easier to iterate over the schoolbook structure directly for the last part
            # or map the indices carefully.
            
            # Let's just fill V_full and W_full for the last part separately.
            pass
            
    # Re-do Part 3 mapping carefully
    # U4_16 is 16x16.
    # V4_16 is 4x16.
    # W4_16 is 4x16.
    
    for c in range(16):
        for r_u in range(16):
            U_full[r_u][offset + c] = U4_16[r_u][c]
            
        for r_v in range(4):
            # r_v is j (0..3)
            # Global V row = r_v * 9 + 8
            V_full[r_v * 9 + 8][offset + c] = V4_16[r_v][c]
            
        for r_w in range(4):
            # r_w is i (0..3)
            # Global W row = 8 * 4 + r_w
            W_full[8 * 4 + r_w][offset + c] = W4_16[r_w][c]

    # Verify
    if not _verify_decomposition(U_full, V_full, W_full, 4, 4, 9):
        # Fallback to schoolbook if verification fails
        return _schoolbook()
        
    return {"U": U_full, "V": V_full, "W": W_full}

def _schoolbook():
    """Schoolbook decomposition of rank n*m*p = 144."""
    n, m, p = DIMENSIONS
    rank = n * m * p
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1.0
                V[j * p + k][col] = 1.0
                W[k * n + i][col] = 1.0
                col += 1
    return {"U": U, "V": V, "W": W}
