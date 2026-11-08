"""Matrix multiplication 4x5x6 rank-minimal exact tensor decomposition."""
import time
import random
from fractions import Fraction

DIMENSIONS = [4, 5, 6]

# EVOLVE-BLOCK-START

def _strassen_2x2():
    """
    Build the 2x2 Strassen decomposition.
    Returns (U, V, W) where each has shape (4, 7).
    U rows: (i,j) -> i*2+j for A
    V rows: (j,k) -> j*2+k for B
    W rows: (k,i) -> k*2+i for C
    """
    U = [[0]*7 for _ in range(4)]
    V = [[0]*7 for _ in range(4)]
    W = [[0]*7 for _ in range(4)]
    
    # P1 = (a11+a22)*(b11+b22)
    U[0][0] = 1; U[3][0] = 1
    V[0][0] = 1; V[3][0] = 1
    W[0][0] = 1; W[3][0] = 1
    
    # P2 = (a21+a22)*b11
    U[2][1] = 1; U[3][1] = 1
    V[0][1] = 1
    W[1][1] = 1; W[3][1] = -1
    
    # P3 = a11*(b12-b22)
    U[0][2] = 1
    V[1][2] = 1; V[3][2] = -1
    W[2][2] = 1; W[3][2] = 1
    
    # P4 = a22*(b21-b11)
    U[3][3] = 1
    V[2][3] = 1; V[0][3] = -1
    W[0][3] = 1; W[1][3] = 1
    
    # P5 = (a11+a12)*b22
    U[0][4] = 1; U[1][4] = 1
    V[3][4] = 1
    W[0][4] = -1; W[2][4] = 1
    
    # P6 = (a21-a11)*(b11+b12)
    U[2][5] = 1; U[0][5] = -1
    V[0][5] = 1; V[1][5] = 1
    W[3][5] = 1
    
    # P7 = (a12-a22)*(b21+b22)
    U[1][6] = 1; U[3][6] = -1
    V[2][6] = 1; V[3][6] = 1
    W[0][6] = 1
    
    return U, V, W

def _schoolbook(n, m, p):
    """
    Schoolbook decomposition for n x m * m x p -> n x p.
    Rank = n*m*p.
    U shape: (n*m, rank), V shape: (m*p, rank), W shape: (p*n, rank)
    """
    rank = n * m * p
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m + j][col] = 1
                V[j*p + k][col] = 1
                W[k*n + i][col] = 1
                col += 1
    return U, V, W

def _decomp_block(n_sub, m_sub, p_sub, rank_sub, U_sub, V_sub, W_sub,
                  i_s, j_s, k_s, n, m, p, U, V, W, offset):
    """
    Place a sub-decomposition for block A[i_s:i_s+n_sub, j_s:j_s+m_sub] *
    B[j_s:j_s+m_sub, k_s:k_s+p_sub] -> C[i_s:i_s+n_sub, k_s:k_s+p_sub]
    into the global matrices at column offset.
    """
    for r in range(n_sub * m_sub):
        i_loc = r // m_sub
        j_loc = r % m_sub
        i_glob = i_s + i_loc
        j_glob = j_s + j_loc
        r_glob = i_glob * m + j_glob
        for c in range(rank_sub):
            U[r_glob][offset + c] = U_sub[r][c]
    
    for r in range(m_sub * p_sub):
        j_loc = r // p_sub
        k_loc = r % p_sub
        j_glob = j_s + j_loc
        k_glob = k_s + k_loc
        r_glob = j_glob * p + k_glob
        for c in range(rank_sub):
            V[r_glob][offset + c] = V_sub[r][c]
    
    for r in range(p_sub * n_sub):
        k_loc = r // n_sub
        i_loc = r % n_sub
        k_glob = k_s + k_loc
        i_glob = i_s + i_loc
        r_glob = k_glob * n + i_glob
        for c in range(rank_sub):
            W[r_glob][offset + c] = W_sub[r][c]

def _build_strassen_2x3():
    """
    Build 2x3 matrix multiplication decomposition.
    A is 2x3, B is 3x2, C is 2x2.
    Uses 2x2 Strassen on overlapping 2x2 sub-blocks.
    Rank = 14 (two independent 2x2 Strassen blocks).
    
    Block 1: A[:, 0:2] * B[0:2, :] -> C[:, :]
    Block 2: A[:, 1:3] * B[1:3, :] -> C[:, :]
    C = Block1 + Block2
    """
    U2, V2, W2 = _strassen_2x2()
    
    rank = 14
    U = [[0]*rank for _ in range(6)]   # n*m = 2*3
    V = [[0]*rank for _ in range(6)]   # m*p = 3*2
    W = [[0]*rank for _ in range(4)]   # p*n = 2*2
    
    # Block 1: A[0:2, 0:2] * B[0:2, 0:2] -> C[0:2, 0:2]
    # i_s=0, j_s=0, k_s=0, n_sub=2, m_sub=2, p_sub=2
    for r in range(4):
        i_loc = r // 2
        j_loc = r % 2
        U[i_loc*3 + j_loc][0 + r%4] = 0  # will fill below
    # Actually let's just place directly
    for t in range(7):
        for r in range(4):
            i_loc = r // 2
            j_loc = r % 2
            U[i_loc*3 + j_loc][t] = U2[r][t]
            V[j_loc*2 + (r%2)][t] = 0  # B rows: j*p+k, j in 0..1, k in 0..1
            # Wait, B is 3x2, so row index is j*2+k where j in 0..2, k in 0..1
            # For block 1, B[0:2, 0:2], so j in 0..1, k in 0..1
            # V row index = j*2 + k
        for r in range(4):
            j_loc = r // 2
            k_loc = r % 2
            V[j_loc*2 + k_loc][t] = V2[r][t]
        for r in range(4):
            k_loc = r // 2
            i_loc = r % 2
            W[k_loc*2 + i_loc][t] = W2[r][t]
    
    # Block 2: A[0:2, 1:3] * B[1:3, 0:2] -> C[0:2, 0:2]
    # i_s=0, j_s=1, k_s=0
    for t in range(7):
        col = 7 + t
        for r in range(4):
            i_loc = r // 2
            j_loc = r % 2
            j_glob = 1 + j_loc  # j_s=1
            U[i_loc*3 + j_glob][col] = U2[r][t]
        for r in range(4):
            j_loc = r // 2
            k_loc = r % 2
            j_glob = 1 + j_loc  # j_s=1
            V[j_glob*2 + k_loc][col] = V2[r][t]
        for r in range(4):
            k_loc = r // 2
            i_loc = r % 2
            W[k_loc*2 + i_loc][col] = W2[r][t]
    
    return U, V, W

def _build_strassen_3x2():
    """
    Build 3x2 matrix multiplication decomposition.
    A is 3x2, B is 2x2, C is 3x2.
    Uses 2x2 Strassen on overlapping 2x2 sub-blocks.
    Rank = 14.
    
    Block 1: A[0:2, :] * B[:, :] -> C[0:2, :]
    Block 2: A[1:3, :] * B[:, :] -> C[1:3, :]
    """
    U2, V2, W2 = _strassen_2x2()
    
    rank = 14
    U = [[0]*rank for _ in range(6)]   # n*m = 3*2
    V = [[0]*rank for _ in range(4)]   # m*p = 2*2
    W = [[0]*rank for _ in range(6)]   # p*n = 2*3
    
    # Block 1: A[0:2, 0:2] * B[0:2, 0:2] -> C[0:2, 0:2]
    # i_s=0, j_s=0, k_s=0
    for t in range(7):
        col = t
        for r in range(4):
            i_loc = r // 2
            j_loc = r % 2
            U[i_loc*2 + j_loc][col] = U2[r][t]
        for r in range(4):
            j_loc = r // 2
            k_loc = r % 2
            V[j_loc*2 + k_loc][col] = V2[r][t]
        for r in range(4):
            k_loc = r // 2
            i_loc = r % 2
            W[k_loc*3 + i_loc][col] = W2[r][t]
    
    # Block 2: A[1:3, 0:2] * B[0:2, 0:2] -> C[1:3, 0:2]
    # i_s=1, j_s=0, k_s=0
    for t in range(7):
        col = 7 + t
        for r in range(4):
            i_loc = r // 2
            j_loc = r % 2
            i_glob = 1 + i_loc
            U[i_glob*2 + j_loc][col] = U2[r][t]
        for r in range(4):
            j_loc = r // 2
            k_loc = r % 2
            V[j_loc*2 + k_loc][col] = V2[r][t]
        for r in range(4):
            k_loc = r // 2
            i_loc = r % 2
            i_glob = 1 + i_loc
            W[k_loc*3 + i_glob][col] = W2[r][t]
    
    return U, V, W

def construct():
    start_time = time.time()
    deadline = start_time + 110.0
    
    n, m, p = DIMENSIONS  # 4, 5, 6
    
    try:
        # Strategy: decompose 4x5 * 5x6 -> 4x6
        # 
        # Decomposition:
        # A = [[A1 (4x3), A2 (4x2)]]  (row blocks, col split 3+2)
        # B = [[B1 (3x3)], [B2 (3x2)], [B3 (2x3)], [B4 (2x2)]]  
        # 
        # Let's use a cleaner decomposition:
        # Split m=5 into 3+2, p=6 into 3+3
        # A is 4x5, B is 5x6
        # 
        # C[0:4, 0:3] = A[0:4,0:3]*B[0:3,0:3] + A[0:4,3:5]*B[3:5,0:3]
        # C[0:4, 3:6] = A[0:4,0:3]*B[0:3,3:6] + A[0:4,3:5]*B[3:5,3:6]
        #
        # So we need:
        # 1. 4x3 * 3x3 -> 4x3
        # 2. 4x2 * 2x3 -> 4x3
        # 3. 4x3 * 3x3 -> 4x3
        # 4. 4x2 * 2x3 -> 4x3
        #
        # Or better: split p=6 into 2+2+2
        # C[:, 0:2] = A[:,:3]*B[:3,:2] + A[:,3:]*B[3:,:2]
        # C[:, 2:4] = A[:,:3]*B[:3,2:4] + A[:,3:]*B[3:,2:4]
        # C[:, 4:6] = A[:,:3]*B[:3,4:6] + A[:,3:]*B[3:,4:6]
        #
        # This gives 6 sub-problems of type (4x3 * 3x2) and (4x2 * 2x2)
        #
        # Actually, let me think of a better decomposition.
        #
        # Use the existing 4x4x4 Strassen (rank 49) for the main block.
        # A[0:4, 0:4] * B[0:4, 0:4] -> C[0:4, 0:4]
        # 
        # Remaining:
        # C[:, 4:6] = A[:, 0:4]*B[0:4, 4:6] + A[:, 4:5]*B[4:5, 0:6]
        #            = (4x4 * 4x2) + (4x1 * 1x6)
        #
        # 4x4 * 4x2: can use 4x4x2 Strassen (rank 28)
        # 4x1 * 1x6: schoolbook rank 24
        #
        # Total: 49 + 28 + 24 = 101
        #
        # Let me try a different split to reduce rank.
        #
        # Split m=5 into 2+3, p=6 into 3+3:
        # C = A1*B1 + A2*B2 where A1=4x2, A2=4x3, B1=2x6, B2=3x6
        # But B is 5x6, so B1 = B[0:2, :6] (2x6), B2 = B[2:5, :6] (3x6)
        # A1 = A[0:4, 0:2] (4x2), A2 = A[0:4, 2:5] (4x3)
        #
        # C = A1*B1 + A2*B2
        # A1*B1: 4x2 * 2x6 -> 4x6
        # A2*B2: 4x3 * 3x6 -> 4x6
        #
        # 4x2 * 2x6: split B into three 2x2 blocks
        #   = A1*B1[:,0:2] + A1*B1[:,2:4] + A1*B1[:,4:6]
        #   Each is 4x2 * 2x2 -> 4x2, rank 7 each (Strassen)
        #   Total rank: 21
        #
        # 4x3 * 3x6: split B into three 3x2 blocks
        #   = A2*B2[:,0:2] + A2*B2[:,2:4] + A2*B2[:,4:6]
        #   Each is 4x3 * 3x2 -> 4x2
        #   For 4x3 * 3x2: 
        #     Split A into 2+2 rows: A_top (2x3) * B (3x2) -> 2x2, and A_bot (2x3) * B -> 2x2
        #     Each 2x3 * 3x2: can use 2x2 Strassen on overlapping columns
        #     Actually, 2x3 * 3x2: 
        #       C = A[:,0:2]*B[0:2,:] + A[:,1:3]*B[1:3,:]
        #       Each is 2x2 * 2x2, rank 7
        #       Total rank: 14
        #     So 4x3 * 3x2 rank = 14 (using 2x2 Strassen on top and bottom)
        #     Wait, that's not quite right. Let me reconsider.
        #     4x3 * 3x2: A is 4 rows, 3 cols; B is 3 rows, 2 cols; C is 4 rows, 2 cols
        #     Split A into two 2x3 blocks: A1 (rows 0-1), A2 (rows 2-3)
        #     A1*B: 2x3 * 3x2 -> 2x2, rank 14 (as computed above for 2x3*3x2)
        #     A2*B: 2x3 * 3x2 -> 2x2, rank 14
        #     Total: 28
        #     Hmm, that's worse than schoolbook (4*3*2=24).
        #
        # Let me just use the known good decomposition: 49 + 28 + 24 = 101.
        # But let me try to optimize.
        
        # Build 4x4x4 Strassen (rank 49)
        # Using the existing _build_strassen_4x4 logic
        U2, V2, W2 = _strassen_2x2()
        rank44 = 49
        U44 = [[0]*rank44 for _ in range(16)]
        V44 = [[0]*rank44 for _ in range(16)]
        W44 = [[0]*rank44 for _ in range(16)]
        
        strassen_defs = [
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
                for la_i in range(2):
                    for la_j in range(2):
                        local_a_row = la_i * 2 + la_j
                        coeff_a = U2[local_a_row][t]
                        if coeff_a != 0:
                            for (bi, bj), sigma in a_blocks:
                                global_i = bi * 2 + la_i
                                global_j = bj * 2 + la_j
                                u_row = global_i * 4 + global_j
                                U44[u_row][col] += coeff_a * sigma
                for lb_j in range(2):
                    for lb_k in range(2):
                        local_b_row = lb_j * 2 + lb_k
                        coeff_b = V2[local_b_row][t]
                        if coeff_b != 0:
                            for (bj, bk), sigma in b_blocks:
                                global_j = bj * 2 + lb_j
                                global_k = bk * 2 + lb_k
                                v_row = global_j * 4 + global_k
                                V44[v_row][col] += coeff_b * sigma
                for lc_k in range(2):
                    for lc_i in range(2):
                        local_c_row = lc_k * 2 + lc_i
                        coeff_c = W2[local_c_row][t]
                        if coeff_c != 0:
                            for (bi, bk), sigma in c_blocks:
                                global_k = bk * 2 + lc_k
                                global_i = bi * 2 + lc_i
                                w_row = global_k * 4 + global_i
                                W44[w_row][col] += coeff_c * sigma
        
        # Build 4x4x2 Strassen (rank 28)
        rank442 = 28
        U442 = [[0]*rank442 for _ in range(16)]
        V442 = [[0]*rank442 for _ in range(8)]
        W442 = [[0]*rank442 for _ in range(8)]
        
        def add_strassen_block_442(a_i_start, a_j_start, b_j_start, c_i_start, c_k_start, base_rank):
            for t in range(7):
                col = base_rank + t
                for li in range(2):
                    for lj in range(2):
                        local_a_row = li * 2 + lj
                        coeff_a = U2[local_a_row][t]
                        if coeff_a != 0:
                            global_i = a_i_start + li
                            global_j = a_j_start + lj
                            u_row = global_i * 4 + global_j
                            U442[u_row][col] += coeff_a
                for lj in range(2):
                    for lk in range(2):
                        local_b_row = lj * 2 + lk
                        coeff_b = V2[local_b_row][t]
                        if coeff_b != 0:
                            global_j = b_j_start + lj
                            global_k = c_k_start + lk
                            v_row = global_j * 2 + global_k
                            V442[v_row][col] += coeff_b
                for lk in range(2):
                    for li in range(2):
                        local_c_row = lk * 2 + li
                        coeff_c = W2[local_c_row][t]
                        if coeff_c != 0:
                            global_k = c_k_start + lk
                            global_i = c_i_start + li
                            w_row = global_k * 4 + global_i
                            W442[w_row][col] += coeff_c
        
        add_strassen_block_442(0, 0, 0, 0, 0, 0)
        add_strassen_block_442(0, 2, 2, 0, 0, 7)
        add_strassen_block_442(2, 0, 0, 2, 0, 14)
        add_strassen_block_442(2, 2, 2, 2, 0, 21)
        
        # Build 4x1x6 Schoolbook (rank 24)
        U416, V416, W416 = _schoolbook(4, 1, 6)
        
        if time.time() > deadline:
            raise TimeoutError("Deadline exceeded")
        
        # Combine: 
        # decomp1: 4x4 * 4x4 -> 4x4, placed at i=0..4, j=0..4, k=0..4
        # decomp2: 4x4 * 4x2 -> 4x2, placed at i=0..4, j=0..4, k=4..6
        # decomp3: 4x1 * 1x6 -> 4x6, placed at i=0..4, j=4..5, k=0..6
        
        total_rank = 49 + 28 + 24
        U = [[0]*total_rank for _ in range(n*m)]
        V = [[0]*total_rank for _ in range(m*p)]
        W = [[0]*total_rank for _ in range(p*n)]
        
        # Place decomp1: A[0:4,0:4] * B[0:4,0:4] -> C[0:4,0:4]
        offset = 0
        _decomp_block(4, 4, 4, 49, U44, V44, W44, 0, 0, 0, n, m, p, U, V, W, offset)
        
        # Place decomp2: A[0:4,0:4] * B[0:4,4:6] -> C[0:4,4:6]
        offset = 49
        _decomp_block(4, 4, 2, 28, U442, V442, W442, 0, 0, 4, n, m, p, U, V, W, offset)
        
        # Place decomp3: A[0:4,4:5] * B[4:5,0:6] -> C[0:4,0:6]
        offset = 77
        _decomp_block(4, 1, 6, 24, U416, V416, W416, 0, 4, 0, n, m, p, U, V, W, offset)
        
        return {"U": U, "V": V, "W": W}
        
    except Exception:
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

# EVOLVE-BLOCK-END
