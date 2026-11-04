import time
import sys

# EVOLVE-BLOCK-START

def _strassen_2x2_2x2_UVW():
    """
    Return Strassen's 7-term decomposition for 2x2 x 2x2 -> 2x2 matrix multiplication.
    U: shape (4, 7) for A[0,0], A[0,1], A[1,0], A[1,1]
    V: shape (4, 7) for B[0,0], B[0,1], B[1,0], B[1,1]
    W: shape (4, 7) for C[0,0], C[0,1], C[1,0], C[1,1]
    
    C[i,k] = sum_r U[i*2+j, r] * V[j*2+k, r] * W[k*2+i, r]
    Note: The W index in the seed is k*2+i for 2x2.
    """
    U = [
        [1, 1, 1, 0, 1, -1, 0],  # a00
        [0, 0, 0, 1, 0, 0, -1],  # a01
        [0, 1, 0, 0, 0, 1, 0],   # a10
        [1, 1, 0, 1, 0, 0, 1],   # a11
    ]
    V = [
        [1, 1, 0, -1, 0, 1, 0],  # b00
        [0, 0, 1, 0, 0, 1, 0],   # b01
        [1, 0, -1, 0, 1, 0, 1],  # b10
        [1, 0, 0, 0, 1, 0, 1],   # b11
    ]
    W = [
        [1, 0, 0, 1],   # C00 = M1 - M4 + M6
        [0, 1, 1, 0],   # C01 = M3 + M5
        [0, 1, 0, 1],   # C10 = M2 + M4
        [1, 0, -1, 1],  # C11 = M1 - M3 + M7
    ]
    return U, V, W

def _build_4x4x4_strassen_UVW():
    """
    Build rank-49 decomposition for 4x4 x 4x4 -> 4x4 using recursive Strassen.
    Returns U (16x49), V (16x49) [for 4x4 B], W (16x49) [for 4x4 C]
    """
    U2, V2, W2 = _strassen_2x2_2x2_UVW()
    
    rank = 49
    # U rows: i in 0..3, j in 0..3. Index i*4+j
    U = [[0]*rank for _ in range(16)]
    # V rows: j in 0..3, k in 0..3. Index j*4+k (local 4x4)
    V = [[0]*rank for _ in range(16)]
    # W rows: k in 0..3, i in 0..3. Index k*4+i (local 4x4)
    W = [[0]*rank for _ in range(16)]
    
    # Block definitions for 4x4 split into 2x2 blocks of 2x2 matrices
    # A = [[A00, A01], [A10, A11]]
    # B = [[B00, B01], [B10, B11]]
    # C = [[C00, C01], [C10, C11]]
    
    # Strassen block products
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    
    # C00 = M1 - M4 + M6
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M3 + M7
    
    # For each block product m_idx (0..6), we have 7 scalar products (s_idx 0..6)
    # Total rank index r = m_idx * 7 + s_idx
    
    # Define block contributions
    # Each block product involves sums/differences of A-blocks and B-blocks.
    # We need to map these to the 2x2 Strassen U, V, W coefficients.
    
    # A-blocks:
    # A00: rows 0-3 (i=0,1; j=0,1)
    # A01: rows 4-7 (i=0,1; j=2,3)
    # A10: rows 8-11 (i=2,3; j=0,1)
    # A11: rows 12-15 (i=2,3; j=2,3)
    
    # B-blocks:
    # B00: cols 0-3 (j=0,1; k=0,1)
    # B01: cols 4-7 (j=0,1; k=2,3)
    # B10: cols 8-11 (j=2,3; k=0,1)
    # B11: cols 12-15 (j=2,3; k=2,3)
    
    # C-blocks:
    # C00: outputs (i=0,1; k=0,1)
    # C01: outputs (i=0,1; k=2,3)
    # C10: outputs (i=2,3; k=0,1)
    # C11: outputs (i=2,3; k=2,3)
    
    # Helper to add contributions
    def add_block_contrib(m_idx, s_idx, a_blocks, b_blocks, c_blocks):
        r = m_idx * 7 + s_idx
        u2 = U2[s_idx] # [u00, u01, u10, u11]
        v2 = V2[s_idx] # [v00, v01, v10, v11]
        w2 = W2[s_idx] # [w00, w01, w10, w11]
        
        for a_blk, a_coeff in a_blocks:
            # a_blk is 0..3 corresponding to A00, A01, A10, A11
            # local i, j in 0..1
            for li in range(2):
                for lj in range(2):
                    # Global i, j
                    gi = (a_blk // 2) * 2 + li
                    gj = (a_blk % 2) * 2 + lj
                    u_idx = gi * 4 + gj
                    # Local Strassen index for A
                    lu_idx = li * 2 + lj
                    U[u_idx][r] += a_coeff * u2[lu_idx]
        
        for b_blk, b_coeff in b_blocks:
            for lj in range(2):
                for lk in range(2):
                    gj = (b_blk // 2) * 2 + lj
                    gk = (b_blk % 2) * 2 + lk
                    v_idx = gj * 4 + gk
                    lv_idx = lj * 2 + lk
                    V[v_idx][r] += b_coeff * v2[lv_idx]
        
        for c_blk, c_coeff in c_blocks:
            for li in range(2):
                for lk in range(2):
                    gi = (c_blk // 2) * 2 + li
                    gk = (c_blk % 2) * 2 + lk
                    w_idx = gk * 4 + gi # Note: W index is k*n+i
                    lw_idx = lk * 2 + li # Note: W2 index is k*2+i
                    W[w_idx][r] += c_coeff * w2[lw_idx]

    # M1 = (A00+A11)(B00+B11)
    # C00 += M1, C11 += M1
    for s in range(7):
        add_block_contrib(0, s, [(0, 1), (3, 1)], [(0, 1), (3, 1)], [(0, 1), (3, 1)])
        
    # M2 = (A10+A11)B00
    # C10 += M2
    for s in range(7):
        add_block_contrib(1, s, [(2, 1), (3, 1)], [(0, 1)], [(2, 1)])
        
    # M3 = A00(B01-B11)
    # C01 += M3, C11 -= M3
    for s in range(7):
        add_block_contrib(2, s, [(0, 1)], [(1, 1), (3, -1)], [(1, 1), (3, -1)])
        
    # M4 = A11(B10-B00)
    # C00 -= M4, C10 += M4
    for s in range(7):
        add_block_contrib(3, s, [(3, 1)], [(2, 1), (0, -1)], [(0, -1), (2, 1)])
        
    # M5 = (A00+A01)B11
    # C01 += M5
    for s in range(7):
        add_block_contrib(4, s, [(0, 1), (1, 1)], [(3, 1)], [(1, 1)])
        
    # M6 = (A10-A00)(B00+B01)
    # C00 += M6
    for s in range(7):
        add_block_contrib(5, s, [(2, 1), (0, -1)], [(0, 1), (1, 1)], [(0, 1)])
        
    # M7 = (A01-A11)(B10+B11)
    # C11 += M7
    for s in range(7):
        add_block_contrib(6, s, [(1, 1), (3, -1)], [(2, 1), (3, 1)], [(3, 1)])
        
    return U, V, W

def _trivial_4x4x3_UVW():
    """
    Build trivial rank-48 decomposition for 4x4 x 4x3 -> 4x3.
    Returns U (16x48), V (12x48), W (12x48)
    V is for B[:, 4:7] (4x3)
    W is for C[:, 4:7] (4x3) -> indices k in 4..6, i in 0..3
    """
    rank = 48
    U = [[0]*rank for _ in range(16)]
    V = [[0]*rank for _ in range(12)] # 4x3
    W = [[0]*rank for _ in range(12)] # 3x4 -> k in 0..2 (local), i in 0..3
    
    t = 0
    for i in range(4):
        for j in range(4):
            for k in range(3):
                u_idx = i * 4 + j
                v_idx = j * 3 + k
                w_idx = k * 4 + i
                U[u_idx][t] = 1
                V[v_idx][t] = 1
                W[w_idx][t] = 1
                t += 1
    return U, V, W

def construct():
    start_time = time.time()
    deadline = start_time + 110
    
    n, m, p = 4, 4, 7
    
    # Part 1: 4x4x4 using Strassen (Rank 49)
    U1, V1, W1 = _build_4x4x4_strassen_UVW()
    
    # Part 2: 4x4x3 using Trivial (Rank 48)
    U2, V2, W2 = _trivial_4x4x3_UVW()
    
    total_rank = 49 + 48
    
    # Full matrices
    U = [[0]*total_rank for _ in range(16)]
    V = [[0]*total_rank for _ in range(28)] # 4x7
    W = [[0]*total_rank for _ in range(28)] # 7x4
    
    # Copy Part 1
    for i in range(16):
        for r in range(49):
            U[i][r] = U1[i][r]
    
    # V1 is 4x4 (j in 0..3, k in 0..3). Full V is 4x7.
    for j in range(4):
        for k in range(4):
            v_idx = j * 7 + k
            for r in range(49):
                V[v_idx][r] = V1[j * 4 + k][r]
                
    # W1 is 4x4 (k in 0..3, i in 0..3). Full W is 7x4.
    for k in range(4):
        for i in range(4):
            w_idx = k * 4 + i
            for r in range(49):
                W[w_idx][r] = W1[k * 4 + i][r]
                
    # Copy Part 2
    offset = 49
    for i in range(16):
        for r in range(48):
            U[i][offset + r] = U2[i][r]
            
    # V2 is 4x3 (j in 0..3, k in 0..2). These correspond to B[:, 4:7].
    for j in range(4):
        for k in range(3):
            v_idx = j * 7 + (4 + k)
            for r in range(48):
                V[v_idx][offset + r] = V2[j * 3 + k][r]
                
    # W2 is 3x4 (k in 0..2, i in 0..3). These correspond to C[:, 4:7].
    for k in range(3):
        for i in range(4):
            w_idx = (4 + k) * 4 + i
            for r in range(48):
                W[w_idx][offset + r] = W2[k * 4 + i][r]
                
    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
