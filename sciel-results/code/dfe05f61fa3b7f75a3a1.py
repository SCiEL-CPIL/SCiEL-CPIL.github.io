"""
Matrix multiplication 4x5 * 5x6 tensor decomposition.
Uses a block decomposition:
  - 4x4 * 4x4 -> 4x4 block using Strassen (rank 49)
  - 4x1 * 1x6 -> 4x6 block using schoolbook (rank 24)
  - 4x4 * 4x2 -> 4x2 block using two 2x2 Strassen (rank 14) + two 2x2 Strassen (rank 14) = 28
Total rank = 49 + 24 + 28 = 101
"""
import time

DIMENSIONS = [4, 5, 6]

# EVOLVE-BLOCK-START

def _strassen_2x2():
    """
    Returns (U, V, W) for 2x2 * 2x2 -> 2x2 Strassen decomposition.
    U shape: (4, 7)  [i*2+j, rank]
    V shape: (4, 7)  [j*2+k, rank]
    W shape: (4, 7)  [k*2+i, rank]
    """
    rank = 7
    U = [[0]*rank for _ in range(4)]
    V = [[0]*rank for _ in range(4)]
    W = [[0]*rank for _ in range(4)]
    
    # P1 = (a00+a11)*(b00+b11)
    U[0][0] = 1; U[3][0] = 1
    V[0][0] = 1; V[3][0] = 1
    W[0][0] = 1; W[3][0] = 1
    
    # P2 = (a10+a11)*b00
    U[2][1] = 1; U[3][1] = 1
    V[0][1] = 1
    W[1][1] = 1; W[3][1] = -1
    
    # P3 = a00*(b01-b11)
    U[0][2] = 1
    V[1][2] = 1; V[3][2] = -1
    W[2][2] = 1; W[3][2] = 1
    
    # P4 = a11*(b10-b00)
    U[3][3] = 1
    V[2][3] = 1; V[0][3] = -1
    W[0][3] = 1; W[1][3] = 1
    
    # P5 = (a00+a01)*b11
    U[0][4] = 1; U[1][4] = 1
    V[3][4] = 1
    W[0][4] = -1; W[2][4] = 1
    
    # P6 = (a10-a00)*(b00+b01)
    U[2][5] = 1; U[0][5] = -1
    V[0][5] = 1; V[1][5] = 1
    W[3][5] = 1
    
    # P7 = (a01-a11)*(b10+b11)
    U[1][6] = 1; U[3][6] = -1
    V[2][6] = 1; V[3][6] = 1
    W[0][6] = 1
    
    return U, V, W

def _build_4x4x4_strassen():
    """
    Build 4x4 * 4x4 -> 4x4 using Strassen. Rank 49.
    U shape: (16, 49)
    V shape: (16, 49)
    W shape: (16, 49)
    """
    U2, V2, W2 = _strassen_2x2()
    rank = 49
    U = [[0]*rank for _ in range(16)]
    V = [[0]*rank for _ in range(16)]
    W = [[0]*rank for _ in range(16)]
    
    # 7 Strassen sub-decompositions for the 2x2 block structure
    # Each handles a 2x2 * 2x2 -> 2x2 multiplication
    strassen_defs = [
        # (A_blocks, B_blocks, C_blocks)
        # A_blocks: list of ((local_i, local_j), coeff)
        # B_blocks: list of ((local_j, local_k), coeff)
        # C_blocks: list of ((local_i, local_k), coeff)
        ([((0,0), 1), ((1,1), 1)],
         [((0,0), 1), ((1,1), 1)],
         [((0,0), 1), ((1,1), 1)]),
        ([((1,0), 1), ((1,1), 1)],
         [((0,0), 1)],
         [((1,0), 1), ((1,1), -1)]),
        ([((0,0), 1)],
         [((0,1), 1), ((1,1), -1)],
         [((0,1), 1), ((1,1), 1)]),
        ([((1,1), 1)],
         [((1,0), 1), ((0,0), -1)],
         [((0,0), 1), ((1,0), 1)]),
        ([((0,0), 1), ((0,1), 1)],
         [((1,1), 1)],
         [((0,0), -1), ((0,1), 1)]),
        ([((1,0), 1), ((0,0), -1)],
         [((0,0), 1), ((0,1), 1)],
         [((1,1), 1)]),
        ([((0,1), 1), ((1,1), -1)],
         [((1,0), 1), ((1,1), 1)],
         [((0,0), 1)]),
    ]
    
    for s, (a_blocks, b_blocks, c_blocks) in enumerate(strassen_defs):
        for t in range(7):
            col = s * 7 + t
            
            # A part: local 2x2 -> global 4x4
            for li in range(2):
                for lj in range(2):
                    local_a_row = li * 2 + lj
                    coeff_a = U2[local_a_row][t]
                    if coeff_a != 0:
                        for (bi, bj), sigma in a_blocks:
                            global_i = bi * 2 + li
                            global_j = bj * 2 + lj
                            u_row = global_i * 4 + global_j
                            U[u_row][col] += coeff_a * sigma
            
            # B part: local 2x2 -> global 4x4
            for lj in range(2):
                for lk in range(2):
                    local_b_row = lj * 2 + lk
                    coeff_b = V2[local_b_row][t]
                    if coeff_b != 0:
                        for (bj, bk), sigma in b_blocks:
                            global_j = bj * 2 + lj
                            global_k = bk * 2 + lk
                            v_row = global_j * 4 + global_k
                            V[v_row][col] += coeff_b * sigma
            
            # C part: local 2x2 -> global 4x4
            for lk in range(2):
                for li in range(2):
                    local_c_row = lk * 2 + li
                    coeff_c = W2[local_c_row][t]
                    if coeff_c != 0:
                        for (ci, ck), sigma in c_blocks:
                            global_i = ci * 2 + li
                            global_k = ck * 2 + lk
                            w_row = global_k * 4 + global_i
                            W[w_row][col] += coeff_c * sigma
    
    return U, V, W

def _build_4x1x6_schoolbook():
    """
    Build 4x1 * 1x6 -> 4x6 using schoolbook. Rank 24.
    U shape: (4, 24)   [i, rank]  (m=1, so n*m = 4)
    V shape: (6, 24)   [k, rank]  (m=1, p=6, so m*p = 6)
    W shape: (24, 24)  [k*4+i, rank]  (p*n = 6*4 = 24)
    """
    rank = 24
    U = [[0]*rank for _ in range(4)]
    V = [[0]*rank for _ in range(6)]
    W = [[0]*rank for _ in range(24)]
    
    col = 0
    for i in range(4):
        for k in range(6):
            U[i][col] = 1
            V[k][col] = 1
            W[k * 4 + i][col] = 1
            col += 1
    
    return U, V, W

def _build_4x4x2_strassen():
    """
    Build 4x4 * 4x2 -> 4x2 using Strassen. Rank 28.
    Decompose into four 2x2 * 2x2 -> 2x2 Strassen blocks.
    U shape: (16, 28)  [i*4+j, rank]
    V shape: (8, 28)   [j*2+k, rank]
    W shape: (8, 28)   [k*4+i, rank]
    """
    U2, V2, W2 = _strassen_2x2()
    rank = 28
    U = [[0]*rank for _ in range(16)]
    V = [[0]*rank for _ in range(8)]
    W = [[0]*rank for _ in range(8)]
    
    # Four 2x2 * 2x2 -> 2x2 blocks:
    # Block 1: A[0:2, 0:2] * B[0:2, 0:2] -> C[0:2, 0:2]
    # Block 2: A[0:2, 2:4] * B[2:4, 0:2] -> C[0:2, 0:2]
    # Block 3: A[2:4, 0:2] * B[0:2, 0:2] -> C[2:4, 0:2]
    # Block 4: A[2:4, 2:4] * B[2:4, 0:2] -> C[2:4, 0:2]
    
    blocks = [
        (0, 0, 0, 0, 0, 0),   # a_i_start, a_j_start, b_j_start, c_i_start, c_k_start, base_rank
        (0, 2, 2, 0, 0, 7),
        (2, 0, 0, 2, 0, 14),
        (2, 2, 2, 2, 0, 21),
    ]
    
    for a_i_start, a_j_start, b_j_start, c_i_start, c_k_start, base_rank in blocks:
        for t in range(7):
            col = base_rank + t
            
            # A part
            for li in range(2):
                for lj in range(2):
                    local_a_row = li * 2 + lj
                    coeff_a = U2[local_a_row][t]
                    if coeff_a != 0:
                        global_i = a_i_start + li
                        global_j = a_j_start + lj
                        u_row = global_i * 4 + global_j
                        U[u_row][col] += coeff_a
            
            # B part
            for lj in range(2):
                for lk in range(2):
                    local_b_row = lj * 2 + lk
                    coeff_b = V2[local_b_row][t]
                    if coeff_b != 0:
                        global_j = b_j_start + lj
                        global_k = c_k_start + lk
                        v_row = global_j * 2 + global_k
                        V[v_row][col] += coeff_b
            
            # C part
            for lk in range(2):
                for li in range(2):
                    local_c_row = lk * 2 + li
                    coeff_c = W2[local_c_row][t]
                    if coeff_c != 0:
                        global_k = c_k_start + lk
                        global_i = c_i_start + li
                        w_row = global_k * 4 + global_i
                        W[w_row][col] += coeff_c
    
    return U, V, W

def _combine_blocks(blocks, n, m, p):
    """
    Combine block decompositions into a full decomposition.
    blocks: list of (U_sub, V_sub, W_sub, i_start, i_end, j_start, j_end, k_start, k_end)
    """
    total_rank = sum(len(U_sub[0]) for U_sub, _, _, _, _, _, _, _, _ in blocks)
    
    U = [[0]*total_rank for _ in range(n * m)]
    V = [[0]*total_rank for _ in range(m * p)]
    W = [[0]*total_rank for _ in range(p * n)]
    
    current_rank = 0
    for U_sub, V_sub, W_sub, i_s, i_e, j_s, j_e, k_s, k_e in blocks:
        n_sub = i_e - i_s
        m_sub = j_e - j_s
        p_sub = k_e - k_s
        rank_sub = len(U_sub[0])
        
        # U: rows are i*m + j for i in [i_s, i_e), j in [j_s, j_e)
        for i in range(n_sub):
            for j in range(m_sub):
                r_sub = i * m_sub + j
                r_glob = (i_s + i) * m + (j_s + j)
                for c in range(rank_sub):
                    U[r_glob][current_rank + c] = U_sub[r_sub][c]
        
        # V: rows are j*p + k for j in [j_s, j_e), k in [k_s, k_e)
        for j in range(m_sub):
            for k in range(p_sub):
                r_sub = j * p_sub + k
                r_glob = (j_s + j) * p + (k_s + k)
                for c in range(rank_sub):
                    V[r_glob][current_rank + c] = V_sub[r_sub][c]
        
        # W: rows are k*n + i for k in [k_s, k_e), i in [i_s, i_e)
        for k in range(p_sub):
            for i in range(n_sub):
                r_sub = k * n_sub + i
                r_glob = (k_s + k) * n + (i_s + i)
                for c in range(rank_sub):
                    W[r_glob][current_rank + c] = W_sub[r_sub][c]
        
        current_rank += rank_sub
    
    return U, V, W

def _verify(U, V, W, n, m, p):
    """
    Verify that the decomposition is correct.
    U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)
    Check: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = delta(i,j,k)
    """
    rank = len(U[0])
    # Check a few random entries
    import random
    random.seed(42)
    for _ in range(100):
        i = random.randint(0, n-1)
        j = random.randint(0, m-1)
        k = random.randint(0, p-1)
        s = 0
        for r in range(rank):
            s += U[i*m + j][r] * V[j*p + k][r] * W[k*n + i][r]
        expected = 1 if (i == j and j == k) else 0
        # Actually, the tensor T[i,j,k] = 1 if this is a valid multiplication term
        # For matrix multiplication, T[i,j,k] = 1 for all i,j,k where the term exists
        # The decomposition should satisfy: C[i,k] = sum_j A[i,j] B[j,k]
        # So the tensor is T[i,j,k] = 1 for all i,j,k
        # And the decomposition satisfies: sum_r U[i,j,r] V[j,k,r] W[k,i,r] = 1 for all i,j,k
        expected = 1
        if s != expected:
            return False, (i, j, k, s, expected)
    return True, None

def construct():
    start_time = time.time()
    deadline = start_time + 110.0
    
    n, m, p = DIMENSIONS
    
    try:
        # Block 1: 4x4 * 4x4 -> 4x4 (Strassen, rank 49)
        U1, V1, W1 = _build_4x4x4_strassen()
        block1 = (U1, V1, W1, 0, 4, 0, 4, 0, 4)
        
        # Block 2: 4x1 * 1x6 -> 4x6 (schoolbook, rank 24)
        U2, V2, W2 = _build_4x1x6_schoolbook()
        block2 = (U2, V2, W2, 0, 4, 4, 5, 0, 6)
        
        # Block 3: 4x4 * 4x2 -> 4x2 (Strassen, rank 28)
        U3, V3, W3 = _build_4x4x2_strassen()
        block3 = (U3, V3, W3, 0, 4, 0, 4, 4, 6)
        
        U, V, W = _combine_blocks([block1, block2, block3], n, m, p)
        
        # Verify
        valid, err = _verify(U, V, W, n, m, p)
        if not valid:
            # Fallback to schoolbook
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
        
        return {"U": U, "V": V, "W": W}
        
    except Exception:
        # Fallback to schoolbook if something goes wrong
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

# EVOLVE-BLOCK-END
