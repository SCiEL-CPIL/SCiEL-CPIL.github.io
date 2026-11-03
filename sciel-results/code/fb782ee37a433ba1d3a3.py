"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 7]

# EVOLVE-BLOCK-START
import time
_start_time = time.time()

def construct():
    n, m, p = DIMENSIONS  # 4, 4, 7
    
    # Part 1: Schoolbook for 4x4x3 (columns 0-2 of B), rank 48
    rank1 = n * m * 3
    U1 = [[0] * rank1 for _ in range(n * m)]
    V1 = [[0] * rank1 for _ in range(m * 3)]
    W1 = [[0] * rank1 for _ in range(3 * n)]
    c = 0
    for i in range(n):
        for j in range(m):
            for k in range(3):
                U1[i * m + j][c] = 1
                V1[j * 3 + k][c] = 1
                W1[k * n + i][c] = 1
                c += 1
    
    # Part 2: 4x4x4 Strassen-based, rank 28 (columns 3-6 of B)
    rank2 = 28
    U2 = [[0] * rank2 for _ in range(n * m)]
    V2 = [[0] * rank2 for _ in range(m * 4)]
    W2 = [[0] * rank2 for _ in range(4 * n)]
    
    br = [(0,2), (0,2), (2,4), (2,4)]
    bc = [(0,2), (2,4), (0,2), (2,4)]
    
    terms = [
        (0, 0, 0, 1, 1), (0, 0, 3, 1, 1), (0, 3, 0, 1, 1), (0, 3, 3, 1, 1),
        (1, 2, 0, 1, 1), (1, 3, 0, 1, 1),
        (2, 0, 1, 1, 1), (2, 0, 3, 1, -1),
        (3, 3, 2, 1, 1), (3, 3, 0, 1, -1),
        (4, 0, 3, 1, 1), (4, 1, 3, 1, 1),
        (5, 2, 0, 1, 1), (5, 2, 1, 1, 1), (5, 0, 0, 1, -1), (5, 0, 1, 1, -1),
        (6, 1, 2, 1, 1), (6, 1, 3, 1, 1), (6, 3, 2, 1, -1), (6, 3, 3, 1, -1),
    ]
    
    c_recon = [
        {0: 1, 3: 1, 4: -1, 6: 1},
        {2: 1, 4: 1},
        {1: 1, 3: 1},
        {0: 1, 1: -1, 2: 1, 5: 1},
    ]
    
    def get_c_block(a_blk, b_blk):
        if bc[a_blk] != br[b_blk]:
            return -1
        cr, cc = br[a_blk], bc[b_blk]
        if cr == (0,2) and cc == (0,2): return 0
        if cr == (0,2) and cc == (2,4): return 1
        if cr == (2,4) and cc == (0,2): return 2
        if cr == (2,4) and cc == (2,4): return 3
        return -1
    
    col2 = 0
    for (m_idx, a_blk, b_blk, a_coeff, b_coeff) in terms:
        c_blk = get_c_block(a_blk, b_blk)
        if c_blk < 0:
            continue
        w_coeff = c_recon[c_blk].get(m_idx, 0)
        if w_coeff == 0:
            continue
        for i_loc in range(2):
            for j_loc in range(2):
                for k_loc in range(2):
                    a_row = br[a_blk][0] + i_loc
                    a_col = bc[a_blk][0] + j_loc
                    b_row = br[b_blk][0] + j_loc
                    b_col = bc[b_blk][0] + k_loc
                    c_row = br[a_blk][0] + i_loc
                    c_col = bc[b_blk][0] + k_loc
                    if col2 < rank2:
                        U2[a_row * m + a_col][col2] = a_coeff
                        V2[b_row * 4 + b_col][col2] = b_coeff
                        W2[c_col * 4 + c_row][col2] = w_coeff
                        col2 += 1
    
    total_rank = rank1 + col2
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    
    for i in range(n * m):
        for c in range(rank1):
            U[i][c] = U1[i][c]
    for i in range(m * 3):
        for c in range(rank1):
            V[i][c] = V1[i][c]
    for i in range(3 * n):
        for c in range(rank1):
            W[i][c] = W1[i][c]
    
    for i in range(n * m):
        for c in range(col2):
            U[i][rank1 + c] = U2[i][c]
    for j in range(m):
        for k_loc in range(4):
            v2_row = j * 4 + k_loc
            v_full_row = j * p + (3 + k_loc)
            for c in range(col2):
                V[v_full_row][rank1 + c] = V2[v2_row][c]
    for k_loc in range(4):
        for i in range(n):
            w2_row = k_loc * n + i
            w_full_row = (3 + k_loc) * n + i
            for c in range(col2):
                W[w_full_row][rank1 + c] = W2[w2_row][c]
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
