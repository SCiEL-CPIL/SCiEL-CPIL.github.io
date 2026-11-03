"""
Matrix multiplication 3x4 by 4x7 with rank-minimal tensor decomposition.
Uses block decomposition with Strassen's algorithm for 2x2 blocks.
"""
import time
import numpy as np

DIMENSIONS = [3, 4, 7]

# EVOLVE-BLOCK-START
def construct():
    """
    Returns a low-rank tensor decomposition for 3x4 * 4x7 matrix multiplication.
    Uses block decomposition:
      - Split m=4 into 2+2
      - For each m-block, split p=7 into 4+3
      - For p=4 blocks, split n=3 into 2+1, p=4 into 2+2
      - For p=3 blocks, split n=3 into 2+1, p=3 into 2+1
    Uses Strassen for 2x2x2, and schoolbook for smaller blocks.
    """
    start_time = time.time()
    deadline = 140.0  # seconds, safety margin before 150s kill

    n, m, p = DIMENSIONS  # 3, 4, 7

    def strassen_222():
        """
        Strassen decomposition for 2x2 * 2x2.
        Returns (U, V, W) each of shape (4, 7).
        Indexing: U[i*2+j], V[j*2+k], W[k*2+i]
        """
        U = [[0]*7 for _ in range(4)]
        V = [[0]*7 for _ in range(4)]
        W = [[0]*7 for _ in range(4)]

        # M1: (A11+A22)*(B11+B22)
        U[0][0] = 1; U[3][0] = 1
        V[0][0] = 1; V[3][0] = 1
        W[0][0] = 1
        W[1][1] = 1

        # M2: (A21+A22)*B11
        U[2][1] = 1; U[3][1] = 1
        V[0][1] = 1
        W[0][1] = 1
        W[1][1] = -1

        # M3: A11*(B12-B22)
        U[0][2] = 1
        V[1][2] = 1; V[3][2] = -1
        W[1][0] = 1
        W[1][1] = 1

        # M4: A22*(B21-B11)
        U[3][3] = 1
        V[2][3] = 1; V[0][3] = -1
        W[0][0] += 1
        W[0][1] += 1

        # M5: (A11+A12)*B22
        U[0][4] = 1; U[1][4] = 1
        V[3][4] = 1
        W[0][0] -= 1
        W[1][0] += 1

        # M6: (A21-A11)*(B11+B12)
        U[2][5] = 1; U[0][5] = -1
        V[0][5] = 1; V[1][5] = 1
        W[1][1] += 1

        # M7: (A12-A22)*(B21+B22)
        U[1][6] = 1; U[3][6] = -1
        V[2][6] = 1; V[3][6] = 1
        W[0][0] += 1

        return U, V, W

    def decomp_1x2x2():
        """
        Rank 2 decomposition for 1x2 * 2x2.
        U: (2, 2), V: (4, 2), W: (2, 2)
        """
        U = [[0]*2 for _ in range(2)]
        V = [[0]*2 for _ in range(4)]
        W = [[0]*2 for _ in range(2)]
        
        # Term 1: j=0
        U[0][0] = 1
        V[0][0] = 1
        V[1][0] = 1
        W[0][0] = 1
        W[1][0] = 1
        
        # Term 2: j=1
        U[1][1] = 1
        V[2][1] = 1
        V[3][1] = 1
        W[0][1] = 1
        W[1][1] = 1
        
        return U, V, W

    def decomp_2x2x1():
        """
        Rank 2 decomposition for 2x2 * 2x1.
        U: (4, 2), V: (2, 2), W: (2, 2)
        """
        U = [[0]*2 for _ in range(4)]
        V = [[0]*2 for _ in range(2)]
        W = [[0]*2 for _ in range(2)]
        
        # Term 1: j=0
        U[0][0] = 1
        U[2][0] = 1
        V[0][0] = 1
        W[0][0] = 1
        W[1][0] = 1
        
        # Term 2: j=1
        U[1][1] = 1
        U[3][1] = 1
        V[1][1] = 1
        W[0][1] = 1
        W[1][1] = 1
        
        return U, V, W

    def decomp_1x2x1():
        """
        Rank 2 decomposition for 1x2 * 2x1.
        U: (2, 2), V: (2, 2), W: (1, 2)
        """
        U = [[0]*2 for _ in range(2)]
        V = [[0]*2 for _ in range(2)]
        W = [[0]*2 for _ in range(1)]
        
        # Term 1: j=0
        U[0][0] = 1
        V[0][0] = 1
        W[0][0] = 1
        
        # Term 2: j=1
        U[1][1] = 1
        V[1][1] = 1
        W[0][1] = 1
        
        return U, V, W

    def combine_decomps(decomps, n_global, m_global, p_global, 
                        n_offsets, m_offsets, p_offsets):
        """
        Combine multiple sub-decompositions into a global decomposition.
        """
        total_rank = sum(len(U[0]) if U else 0 for U, V, W in decomps)
        
        U_global = [[0]*total_rank for _ in range(n_global * m_global)]
        V_global = [[0]*total_rank for _ in range(m_global * p_global)]
        W_global = [[0]*total_rank for _ in range(p_global * n_global)]
        
        rank_offset = 0
        for idx, (U_sub, V_sub, W_sub) in enumerate(decomps):
            n_start, n_len = n_offsets[idx]
            m_start, m_len = m_offsets[idx]
            p_start, p_len = p_offsets[idx]
            
            sub_rank = len(U_sub[0]) if U_sub else 0
            
            for i_sub in range(n_len):
                for j_sub in range(m_len):
                    global_i = n_start + i_sub
                    global_j = m_start + j_sub
                    global_row = global_i * m_global + global_j
                    sub_row = i_sub * m_len + j_sub
                    for c in range(sub_rank):
                        U_global[global_row][rank_offset + c] = U_sub[sub_row][c]
            
            for j_sub in range(m_len):
                for k_sub in range(p_len):
                    global_j = m_start + j_sub
                    global_k = p_start + k_sub
                    global_row = global_j * p_global + global_k
                    sub_row = j_sub * p_len + k_sub
                    for c in range(sub_rank):
                        V_global[global_row][rank_offset + c] = V_sub[sub_row][c]
            
            for k_sub in range(p_len):
                for i_sub in range(n_len):
                    global_k = p_start + k_sub
                    global_i = n_start + i_sub
                    global_row = global_k * n_global + global_i
                    sub_row = k_sub * n_len + i_sub
                    for c in range(sub_rank):
                        W_global[global_row][rank_offset + c] = W_sub[sub_row][c]
            
            rank_offset += sub_rank
        
        return U_global, V_global, W_global

    # Build M_{2,2,2} (Strassen)
    S_222 = strassen_222()
    S_221 = decomp_2x2x1()
    S_122 = decomp_1x2x2()
    S_121 = decomp_1x2x1()

    # Build M_{3,2,4}
    decomps_324 = [
        S_222, S_222, S_122, S_122
    ]
    n_offsets_324 = [(0, 2), (0, 2), (2, 1), (2, 1)]
    m_offsets_324 = [(0, 2), (0, 2), (0, 2), (0, 2)]
    p_offsets_324 = [(0, 2), (2, 2), (0, 2), (2, 2)]

    U_324, V_324, W_324 = combine_decomps(
        decomps_324, 3, 2, 4,
        n_offsets_324, m_offsets_324, p_offsets_324
    )

    # Build M_{3,2,3}
    decomps_323 = [
        S_222, S_221, S_122, S_121
    ]
    n_offsets_323 = [(0, 2), (0, 2), (2, 1), (2, 1)]
    m_offsets_323 = [(0, 2), (0, 2), (0, 2), (0, 2)]
    p_offsets_323 = [(0, 2), (2, 1), (0, 2), (2, 1)]

    U_323, V_323, W_323 = combine_decomps(
        decomps_323, 3, 2, 3,
        n_offsets_323, m_offsets_323, p_offsets_323
    )

    # Build M_{3,2,7}
    decomps_327 = [
        (U_324, V_324, W_324),
        (U_323, V_323, W_323)
    ]
    n_offsets_327 = [(0, 3), (0, 3)]
    m_offsets_327 = [(0, 2), (0, 2)]
    p_offsets_327 = [(0, 4), (4, 3)]

    U_327, V_327, W_327 = combine_decomps(
        decomps_327, 3, 2, 7,
        n_offsets_327, m_offsets_327, p_offsets_327
    )

    # Build M_{3,4,7}
    decomps_347 = [
        (U_327, V_327, W_327),
        (U_327, V_327, W_327)
    ]
    n_offsets_347 = [(0, 3), (0, 3)]
    m_offsets_347 = [(0, 2), (2, 2)]
    p_offsets_347 = [(0, 7), (0, 7)]

    U_final, V_final, W_final = combine_decomps(
        decomps_347, 3, 4, 7,
        n_offsets_347, m_offsets_347, p_offsets_347
    )

    rank = len(U_final[0]) if U_final else 0

    # Verify correctness
    A = np.random.randn(3, 4)
    B = np.random.randn(4, 7)
    C_expected = A @ B

    U_arr = np.array(U_final, dtype=float)
    V_arr = np.array(V_final, dtype=float)
    W_arr = np.array(W_final, dtype=float)

    C_computed = np.zeros((3, 7))
    for i in range(3):
        for k in range(7):
            val = 0.0
            for j in range(4):
                for r in range(rank):
                    val += A[i, j] * B[j, k] * U_arr[i*4+j, r] * V_arr[j*7+k, r] * W_arr[k*3+i, r]
            C_computed[i, k] = val

    error = np.max(np.abs(C_computed - C_expected))
    if error > 1e-10:
        rank_school = n * m * p
        U_school = [[0] * rank_school for _ in range(n * m)]
        V_school = [[0] * rank_school for _ in range(m * p)]
        W_school = [[0] * rank_school for _ in range(p * n)]
        
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_school[i * m + j][col] = 1
                    V_school[j * p + k][col] = 1
                    W_school[k * n + i][col] = 1
                    col += 1
        
        return {"U": U_school, "V": V_school, "W": W_school}

    return {"U": U_final, "V": V_final, "W": W_final}
# EVOLVE-BLOCK-END
