import numpy as np
import time

DIMENSIONS = [5, 5, 5]

def _schoolbook_decomp(n, m, p):
    """
    Returns U, V, W for schoolbook multiplication of n x m by m x p.
    U: (n*m, n*m*p)
    V: (m*p, n*m*p)
    W: (p*n, n*m*p)
    """
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

def _strassen_decomp_2x2():
    """
    Returns U, V, W for Strassen multiplication of 2x2 by 2x2.
    Rank 7.
    U: (4, 7), V: (4, 7), W: (4, 7)
    """
    U = [
        [1, 0, 1, 0, 1, -1, 0], # a
        [0, 0, 0, 0, 1, 0, -1], # b
        [0, 1, 0, 0, 0, 1, 0], # c
        [1, 1, 0, 1, 0, 0, 0]  # d
    ]
    V = [
        [1, 1, 0, 0, 0, 1, 1], # e
        [0, 0, 1, -1, 1, 0, 0], # f
        [0, 1, 0, 1, 0, 1, 1], # g
        [1, 0, -1, -1, 1, 0, 0] # h
    ]
    W = [
        [1, 0, 0, -1, 1], # i (C00)
        [0, 0, 1, 0, 1], # j (C01)
        [0, 1, 0, 1, 0], # k (C10)
        [1, -1, 1, 0, 1]  # l (C11)
    ]
    return U, V, W

def _compose_decompositions(decomp1, decomp2, offset1, offset2, rank1, rank2, n1, m1, p1, n2, m2, p2):
    """
    Placeholder to keep signature.
    """
    pass

def _strassen_decomp_4x4():
    """
    Build 4x4 Strassen decomposition by composing 7 2x2 Strassen decompositions.
    Returns U, V, W of shape (16, 49).
    """
    U2, V2, W2 = _strassen_decomp_2x2()
    
    U = [[0] * 49 for _ in range(16)]
    V = [[0] * 49 for _ in range(16)]
    W = [[0] * 49 for _ in range(16)]
    
    # Block indices in 4x4 (row-major)
    # A11: 0, 1, 4, 5
    # A12: 2, 3, 6, 7
    # A21: 8, 9, 12, 13
    # A22: 10, 11, 14, 15
    
    A11 = [0, 1, 4, 5]
    A12 = [2, 3, 6, 7]
    A21 = [8, 9, 12, 13]
    A22 = [10, 11, 14, 15]
    
    B11 = [0, 1, 4, 5]
    B12 = [2, 3, 6, 7]
    B21 = [8, 9, 12, 13]
    B22 = [10, 11, 14, 15]
    
    C11 = [0, 1, 4, 5]
    C12 = [2, 3, 6, 7]
    C21 = [8, 9, 12, 13]
    C22 = [10, 11, 14, 15]
    
    # Map integer keys to C block index lists to avoid unhashable list keys
    C_map_keys = {
        0: C11,
        1: C12,
        2: C21,
        3: C22
    }

    def add_step_signed(A_blocks, B_blocks, C_map, col_off):
        # A_blocks: list of (indices, sign)
        # B_blocks: list of (indices, sign)
        # C_map: dict {key: sign} where key is int mapping to C block
        for r in range(4):
            for c in range(7):
                u_val = U2[r][c]
                v_val = V2[r][c]
                
                # Update U
                if u_val != 0:
                    for block, sign in A_blocks:
                        U[block[r]][col_off + c] += u_val * sign
                        
                # Update V
                if v_val != 0:
                    for block, sign in B_blocks:
                        V[block[r]][col_off + c] += v_val * sign
                        
                # Update W
                for c_key, sign in C_map.items():
                    w_val = W2[r][c] * sign
                    if w_val != 0:
                        c_indices = C_map_keys[c_key]
                        W[c_indices[r]][col_off + c] += w_val

    # S1: (A11+A22)(B11+B22) -> C11+=S1, C12-=S1, C21-=S1, C22+=S1
    add_step_signed([(A11, 1), (A22, 1)], [(B11, 1), (B22, 1)], {0: 1, 1: -1, 2: -1, 3: 1}, 0)
    
    # S2: (A21+A22)B11 -> C21+=S2, C22-=S2
    add_step_signed([(A21, 1), (A22, 1)], [(B11, 1)], {2: 1, 3: -1}, 7)
    
    # S3: A11(B12-B22) -> C12+=S3, C22+=S3
    add_step_signed([(A11, 1)], [(B12, 1), (B22, -1)], {1: 1, 3: 1}, 14)
    
    # S4: A22(B21-B11) -> C11+=S4, C21+=S4
    add_step_signed([(A22, 1)], [(B21, 1), (B11, -1)], {0: 1, 2: 1}, 21)
    
    # S5: (A11+A12)B22 -> C11-=S5, C12+=S5
    add_step_signed([(A11, 1), (A12, 1)], [(B22, 1)], {0: -1, 1: 1}, 28)
    
    # S6: (A21-A11)(B11+B12) -> C22+=S6
    add_step_signed([(A21, 1), (A11, -1)], [(B11, 1), (B12, 1)], {3: 1}, 35)
    
    # S7: (A12-A22)(B21+B22) -> C11+=S7
    add_step_signed([(A12, 1), (A22, -1)], [(B21, 1), (B22, 1)], {0: 1}, 42)
    
    return U, V, W

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    
    # Get 4x4 Strassen
    U4, V4, W4 = _strassen_decomp_4x4()
    
    # Total rank: 49 (Strassen 4x4) + 61 (Schoolbook cross terms) = 110
    total_rank = 110
    U = [[0] * total_rank for _ in range(25)]
    V = [[0] * total_rank for _ in range(25)]
    W = [[0] * total_rank for _ in range(25)]
    
    # 1. Add 4x4 Strassen for A11 B11 -> C11
    # Indices 0..15 in U,V,W correspond to the 4x4 blocks
    for i in range(16):
        for j in range(49):
            U[i][j] = U4[i][j]
            V[i][j] = V4[i][j]
            W[i][j] = W4[i][j]
            
    col = 49
    
    # Helper to check time
    def time_check():
        if time.time() - start_time > 100:
            return True
        return False

    # 2. A12 B21 -> C11 (Outer product 4x4)
    # A12 indices in 5x5 (row-major): 4, 9, 14, 19
    # B21 indices in 5x5 (row-major): 20, 21, 22, 23
    # C11 indices in 5x5 (row-major): 0, 1, 2, 3, 5, 6, 7, 8, 10, 11, 12, 13, 15, 16, 17, 18
    # C[i,j] += A[i,4] * B[4,j] for i,j in 0..3
    A12_5x5 = [4, 9, 14, 19]
    B21_5x5 = [20, 21, 22, 23]
    C11_5x5 = [i*5 + j for i in range(4) for j in range(4)]
    
    for i in range(4):
        for j in range(4):
            if time_check(): break
            U[A12_5x5[i]][col] = 1
            V[B21_5x5[j]][col] = 1
            W[C11_5x5[i*4 + j]][col] = 1
            col += 1
    if time_check():
        return {"U": U, "V": V, "W": W}

    # 3. A11 B12 -> C12
    # C[i,4] += sum_k A[i,k] B[k,4] for i in 0..3, k in 0..3
    # A11 indices: i*5 + k
    # B12 indices: k*5 + 4
    # C12 indices: i*5 + 4
    for i in range(4):
        for k in range(4):
            if time_check(): break
            U[i*5 + k][col] = 1
            V[k*5 + 4][col] = 1
            W[i*5 + 4][col] = 1
            col += 1
    if time_check():
        return {"U": U, "V": V, "W": W}

    # 4. A12 B22 -> C12
    # C[i,4] += A[i,4] B[4,4] for i in 0..3
    for i in range(4):
        if time_check(): break
        U[i*5 + 4][col] = 1
        V[24][col] = 1
        W[i*5 + 4][col] = 1
        col += 1
    if time_check():
        return {"U": U, "V": V, "W": W}

    # 5. A21 B11 -> C21
    # C[4,j] += sum_k A[4,k] B[k,j] for j in 0..3, k in 0..3
    for j in range(4):
        for k in range(4):
            if time_check(): break
            U[20 + k][col] = 1
            V[k*5 + j][col] = 1
            W[20 + j][col] = 1
            col += 1
    if time_check():
        return {"U": U, "V": V, "W": W}

    # 6. A22 B21 -> C21
    # C[4,j] += A[4,4] B[4,j] for j in 0..3
    for j in range(4):
        if time_check(): break
        U[24][col] = 1
        V[20 + j][col] = 1
        W[20 + j][col] = 1
        col += 1
    if time_check():
        return {"U": U, "V": V, "W": W}

    # 7. A21 B12 -> C22
    # C[4,4] += sum_k A[4,k] B[k,4] for k in 0..3
    for k in range(4):
        if time_check(): break
        U[20 + k][col] = 1
        V[k*5 + 4][col] = 1
        W[24][col] = 1
        col += 1
    if time_check():
        return {"U": U, "V": V, "W": W}

    # 8. A22 B22 -> C22
    U[24][col] = 1
    V[24][col] = 1
    W[24][col] = 1
    col += 1
    
    # col should be 110
    return {"U": U, "V": V, "W": W}
