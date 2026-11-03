import numpy as np
import time
import itertools

DIMENSIONS = [3, 5, 6]

# EVOLVE-BLOCK-START

def _strassen_2x2():
    """
    Return the Strassen decomposition for 2x2 * 2x2 -> 2x2 multiplication.
    U: (4, 7), V: (4, 7), W: (4, 7)
    Convention: U[i*2+j, c], V[j*2+k, c], W[k*2+i, c]
    """
    U = np.zeros((4, 7))
    V = np.zeros((4, 7))
    W = np.zeros((4, 7))
    
    # Strassen's 7 products:
    # M1 = (a00+a11)(b00+b11)
    # M2 = (a10+a11)b00
    # M3 = a00(b10-b11)
    # M4 = a11(b01-b00)
    # M5 = (a00+a01)b11
    # M6 = (a10-a00)(b00+b10)
    # M7 = (a01-a11)(b01+b11)
    #
    # C00 = M1 - M4 + M5 - M7
    # C01 = M3 + M4
    # C10 = M2 + M5
    # C11 = M1 - M3 + M6 - M7
    
    # Index mapping: A is 2x2, indices: (0,0)=0, (0,1)=1, (1,0)=2, (1,1)=3
    # B is 2x2, indices: (0,0)=0, (0,1)=1, (1,0)=2, (1,1)=3
    # C is 2x2, indices: (0,0)=0, (0,1)=1, (1,0)=2, (1,1)=3
    # W index: k*2+i. So C00->0, C01->1, C10->2, C11->3
    
    # M1
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    W[0, 0] = 1; W[3, 0] = 1  # C00, C11
    
    # M2
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    W[2, 1] = 1  # C10
    
    # M3
    U[0, 2] = 1
    V[2, 2] = 1; V[3, 2] = -1
    W[1, 2] = 1; W[3, 2] = -1  # C01, C11
    
    # M4
    U[3, 3] = 1
    V[1, 3] = 1; V[0, 3] = -1
    W[0, 3] = -1; W[1, 3] = 1  # C00, C01
    
    # M5
    U[0, 4] = 1; U[1, 4] = 1
    V[3, 4] = 1
    W[0, 4] = 1; W[2, 4] = 1  # C00, C10
    
    # M6
    U[2, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[2, 5] = 1
    W[3, 5] = 1  # C11
    
    # M7
    U[1, 6] = 1; U[3, 6] = -1
    V[1, 6] = 1; V[3, 6] = 1
    W[0, 6] = -1; W[3, 6] = -1  # C00, C11
    
    return U, V, W

def _embed_block_2x5x2_to_2x2(U_block, V_block, W_block, 
                               a_rows, a_cols, 
                               b_rows, b_cols, 
                               c_rows, c_cols,
                               U_out, V_out, W_out, col_offset):
    """
    Embed a 2x2x2 Strassen decomposition into a larger 3x5x6 problem.
    a_rows: list of 2 row indices in A (0-2)
    a_cols: list of 2 col indices in A (0-4)
    b_rows: list of 2 row indices in B (0-4) -- must match a_cols order
    b_cols: list of 2 col indices in B (0-5)
    c_rows: list of 2 row indices in C (0-2) -- output rows
    c_cols: list of 2 col indices in C (0-5) -- output cols
    
    U_out: (15, R)
    V_out: (30, R)
    W_out: (18, R)
    """
    n, m, p = 3, 5, 6
    for c in range(7):
        u_col = col_offset + c
        # U entries
        for ui, val in enumerate(U_block[:, c]):
            if val != 0:
                i_local = ui // 2
                j_local = ui % 2
                i = a_rows[i_local]
                j = a_cols[j_local]
                U_out[i * m + j, u_col] = val
        
        # V entries
        for vi, val in enumerate(V_block[:, c]):
            if val != 0:
                j_local = vi // 2
                k_local = vi % 2
                j = b_rows[j_local]
                k = b_cols[k_local]
                V_out[j * p + k, u_col] = val
        
        # W entries
        for wi, val in enumerate(W_block[:, c]):
            if val != 0:
                k_local = wi // 2
                i_local = wi % 2
                k = c_cols[k_local]
                i = c_rows[i_local]
                W_out[k * n + i, u_col] = val
    return col_offset + 7

def _embed_block_1x5x2_to_1x1(U_block, V_block, W_block,
                               a_row, a_cols,
                               b_rows, b_cols,
                               c_row, c_col,
                               U_out, V_out, W_out, col_offset):
    """
    Embed a 1x1x1 multiplication (scalar) into the larger problem.
    This is just the schoolbook for 1x1x1.
    """
    n, m, p = 3, 5, 6
    for c in range(4):  # Actually 1x5x2 has rank 4? No, 1*5*2 = 10 schoolbook.
        pass
    
    # Actually, for 1x5 * 5x2, the schoolbook rank is 10.
    # But we can do better? No, rank of 1x5x2 is 5? 
    # C_k = sum_j A_j B_jk. This is 5 independent products per output column.
    # So rank is 5 * 2 = 10.
    
    # Wait, the previous logic said rank 4 for 1x2x2.
    # 1x2 * 2x2 -> 1x2. C0 = A0 B00 + A1 B10. C1 = A0 B01 + A1 B11.
    # This is 4 products. Rank 4.
    
    # So 1x5x2 is composed of 5 such 1x1x2? No.
    # A is 1x5, B is 5x2.
    # C0 = sum_{j=0}^4 A_j B_j0.
    # C1 = sum_{j=0}^4 A_j B_j1.
    # This is 10 multiplications.
    
    # Can we do better? 
    # If we group A and B, maybe.
    # But for now, let's use schoolbook for the 1x5x2 blocks.
    # Rank = 10.
    
    # Wait, the previous estimate was 33 total.
    # 3 * 7 (Strassen for 2x2) + 3 * 10 (Schoolbook for 1x5x2? No, 1x2x2 is 4, so 1x5x2 is 5*4=20?)
    # Let's re-evaluate.
    # If we split 5 into 2+2+1.
    # Top row (2x5) * 5x2 blocks.
    # The 2x5 * 5x2 is not a single 2x2x2. It's a bilinear map.
    # If we split the 5 dimension into 2+2+1, we get:
    # (A00 + A01 + A02) where A00 is 2x2, A01 is 2x2, A02 is 2x1.
    # This is getting complicated.
    
    # Let's stick to the simplest valid decomposition first: Schoolbook Rank 90.
    # Then try to improve.
    
    return col_offset

def _generate_schoolbook(n, m, p):
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    return U, V, W

def _verify_decomp(n, m, p, U, V, W):
    """
    Verify if the factors U, V, W correctly decompose the matrix multiplication tensor.
    """
    rank = U.shape[1]
    np.random.seed(12345)
    A = np.random.randint(-10, 10, (n, m))
    B = np.random.randint(-10, 10, (m, p))
    C_expected = A @ B
    
    C_computed = np.zeros((n, p))
    
    for c in range(rank):
        u_c = U[:, c].reshape(n, m)
        v_c = V[:, c].reshape(m, p)
        w_c = W[:, c].reshape(p, n).T # (n, p)
        
        X = np.sum(A * u_c, axis=1) # shape (n,)
        Y = np.sum(B * v_c, axis=0) # shape (p,)
        C_computed += w_c * np.outer(X, Y)
        
    if not np.allclose(C_computed, C_expected, atol=1e-9):
        return False
    return True

def _try_rank_84(n, m, p):
    """
    Attempt to construct a rank-84 decomposition.
    Strategy:
    1. Split A (3x5) into A_top (2x5) and A_bot (1x5).
    2. Split B (5x6) into three 5x2 blocks: B0, B1, B2.
    3. The product C = A B has blocks:
       C_top = A_top B (2x6)
       C_bot = A_bot B (1x6)
    
    4. For C_top (2x6), we can view it as three 2x2 blocks: C_top0, C_top1, C_top2.
       Each C_top_j = A_top B_j.
       A_top is 2x5, B_j is 5x2.
       This is a 2x5x2 multiplication.
       Rank of 2x5x2?
       We can decompose 2x5x2 into five 2x1x2 multiplications.
       Each 2x1x2 is rank 4 (schoolbook).
       Total rank = 5 * 4 = 20?
       Or can we use Strassen?
       If we pad 2x5 to 2x8? No.
       
       Actually, a known bound for 2x5x2 is 14? Or 15?
       If we use 3*7 = 21 for the top row, and 3*10 = 30 for the bottom row?
       21 + 30 = 51?
       
       Wait, the middle dimension is shared.
       The Strassen algorithm on 2x2 blocks requires the input blocks to be independent.
       If we split the 5 dimension into 2+2+1, the blocks are:
       A_top0 (2x2), A_top1 (2x2), A_top2 (2x1)
       B0 (5x2) -> B00 (2x2), B01 (2x2), B02 (1x2)
       
       C_top0 = A_top0 B00 + A_top1 B01 + A_top2 B02
       This is a sum of three terms.
       This is not a simple block multiplication.
       
       Therefore, simple block Strassen does not directly apply.
       
       Let's try a different approach:
       Use the known rank-84 decomposition from literature.
       Since I don't have it hardcoded, I will return None.
    """
    return None

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110  # 10s buffer
    
    # Generate Schoolbook (Rank 90) as baseline
    U_school, V_school, W_school = _generate_schoolbook(n, m, p)
    if _verify_decomp(n, m, p, U_school, V_school, W_school):
        best_result = {"U": U_school.tolist(), "V": V_school.tolist(), "W": W_school.tolist()}
        best_rank = n * m * p
    else:
        return None

    # Attempt Rank-84 Construction
    result_84 = _try_rank_84(n, m, p)
    if result_84 is not None:
        U_84, V_84, W_84 = result_84
        if _verify_decomp(n, m, p, U_84, V_84, W_84):
            return {"U": U_84.tolist(), "V": V_84.tolist(), "W": W_84.tolist()}

    # Fallback to Schoolbook
    return best_result

# EVOLVE-BLOCK-END
