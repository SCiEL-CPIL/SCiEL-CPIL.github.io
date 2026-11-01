"""Matrix multiplication 2x3 by 3x8 tensor decomposition."""
DIMENSIONS = [2, 3, 8]

# EVOLVE-BLOCK-START
import time
import numpy as np

def _verify_decomposition(U, V, W, n, m, p):
    """
    Verify if U, V, W exactly decompose the matrix multiplication tensor.
    U: (n*m, rank)
    V: (m*p, rank)
    W: (p*n, rank)
    T[a, b, c] = sum_d U[a, d] * V[b, d] * W[c, d]
    where a = i*m + j, b = j*p + k, c = k*n + i
    """
    rank = U.shape[1]
    
    for a in range(n * m):
        i_a = a // m
        j_a = a % m
        for b in range(m * p):
            j_b = b // p
            k_b = b % p
            for c in range(p * n):
                k_c = c // n
                i_c = c % n
                val = 0.0
                for d in range(rank):
                    val += U[a, d] * V[b, d] * W[c, d]
                
                # Determine expected value
                if i_a == i_c and j_a == j_b and k_b == k_c:
                    if val != 1.0:
                        return False
                else:
                    if val != 0.0:
                        return False
    return True

def _construct_sub_block_rank11():
    """
    Construct a rank-11 decomposition for 2x3 * 3x2.
    Returns U (6x11), V (6x11), W (4x11).
    """
    n, m, p = 2, 3, 2
    rank = 11
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    # Indices for A (2x3): a = i*3 + j
    # Indices for B (3x2): b = j*2 + k
    # Indices for C (2x2): c = k*2 + i (Trace convention W[k*n+i])
    
    # Strassen terms for 2x2x2 part (using A[0:2, 0:2] and B[0:2, :])
    # A indices: 00(0), 01(1), 10(3), 11(4)
    # B indices: 00(0), 01(1), 10(2), 11(3)
    # C indices: 00(0), 01(1), 10(2), 11(3)
    
    # Term 1: p1 = (a00+a11)(b00+b11)
    U[0, 0] = 1; U[4, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    W[0, 0] = 1; W[1, 0] = 1; W[2, 0] = 1; W[3, 0] = 1
    
    # Term 2: p2 = (a10+a11)b00
    U[3, 1] = 1; U[4, 1] = 1
    V[0, 1] = 1
    W[2, 1] = 1; W[3, 1] = -1
    
    # Term 3: p3 = a00(b01-b11)
    U[0, 2] = 1
    V[1, 2] = 1; V[3, 2] = -1
    W[1, 2] = 1; W[3, 2] = 1
    
    # Term 4: p4 = a11(b10-b00)
    U[4, 3] = 1
    V[2, 3] = 1; V[0, 3] = -1
    W[0, 3] = 1; W[2, 3] = 1
    
    # Term 5: p5 = (a00+a01)b11
    U[0, 4] = 1; U[1, 4] = 1
    V[3, 4] = 1
    W[0, 4] = -1; W[1, 4] = 1
    
    # Term 6: p6 = (a10-a00)(b00+b01)
    U[3, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[1, 5] = 1
    W[3, 5] = 1
    
    # Term 7: p7 = (a01-a11)(b10+b11)
    U[1, 6] = 1; U[4, 6] = -1
    V[2, 6] = 1; V[3, 6] = 1
    W[0, 6] = 1
    
    # Cross terms for A[:, 2] and B[2, :]
    # A indices: 02(2), 12(5)
    # B indices: 20(4), 21(5)
    
    # Term 8: a02 * b20 -> c00
    U[2, 7] = 1
    V[4, 7] = 1
    W[0, 7] = 1
    
    # Term 9: a02 * b21 -> c01
    U[2, 8] = 1
    V[5, 8] = 1
    W[1, 8] = 1
    
    # Term 10: a12 * b20 -> c10
    U[5, 9] = 1
    V[4, 9] = 1
    W[2, 9] = 1
    
    # Term 11: a12 * b21 -> c11
    U[5, 10] = 1
    V[5, 10] = 1
    W[3, 10] = 1
    
    return U, V, W

def _construct_tiled_rank44():
    """
    Construct a rank-44 decomposition for 2x3 * 3x8 by tiling 4 rank-11 sub-blocks.
    """
    n, m, p = 2, 3, 8
    U_sub, V_sub, W_sub = _construct_sub_block_rank11()
    rank_sub = U_sub.shape[1] # 11
    num_blocks = p // 2 # 4
    rank = num_blocks * rank_sub # 44
    
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    for block in range(num_blocks):
        col_offset = block * rank_sub
        
        # U is the same for all blocks
        U[:, col_offset:col_offset+rank_sub] = U_sub
        
        # V: Map sub-block B (3x2) to full B (3x8)
        # Sub-block k indices: 0, 1
        # Full B k indices: 2*block, 2*block+1
        # V index: j * p + k
        for j in range(m):
            for k_sub in range(2):
                k_full = 2 * block + k_sub
                v_idx_full = j * p + k_full
                v_idx_sub = j * 2 + k_sub
                V[v_idx_full, col_offset:col_offset+rank_sub] = V_sub[v_idx_sub, :]
        
        # W: Map sub-block C (2x2) to full C (2x8)
        # Sub-block k indices: 0, 1
        # Full C k indices: 2*block, 2*block+1
        # W index: k * n + i
        for k_sub in range(2):
            k_full = 2 * block + k_sub
            for i in range(n):
                w_idx_full = k_full * n + i
                w_idx_sub = k_sub * n + i
                W[w_idx_full, col_offset:col_offset+rank_sub] = W_sub[w_idx_sub, :]
    
    return U, V, W

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin
    
    n, m, p = DIMENSIONS  # 2, 3, 8
    
    # 1. Construct the deterministic rank-44 solution
    U_tiled, V_tiled, W_tiled = _construct_tiled_rank44()
    rank_tiled = U_tiled.shape[1]
    
    # Verify tiled decomposition
    if _verify_decomposition(U_tiled, V_tiled, W_tiled, n, m, p):
        return {"U": U_tiled.tolist(), "V": V_tiled.tolist(), "W": W_tiled.tolist()}
    
    # Fallback: Schoolbook
    rank_school = n * m * p
    U_school = np.zeros((n * m, rank_school))
    V_school = np.zeros((m * p, rank_school))
    W_school = np.zeros((p * n, rank_school))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j, col] = 1
                V_school[j * p + k, col] = 1
                W_school[k * n + i, col] = 1
                col += 1
    
    return {"U": U_school.tolist(), "V": V_school.tolist(), "W": W_school.tolist()}

# EVOLVE-BLOCK-END
