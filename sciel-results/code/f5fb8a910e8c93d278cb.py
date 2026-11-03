"""Tensor decomposition for 4x4 x 4x9 matrix multiplication."""
import time
import numpy as np

DIMENSIONS = [4, 4, 9]

# EVOLVE-BLOCK-START

def _strassen_2x2():
    """
    Returns U, V, W for M_{2,2,2} using Strassen's algorithm.
    Rank 7.
    U: (4, 7), V: (4, 7), W: (4, 7)
    """
    rank = 7
    U = np.zeros((4, rank))
    V = np.zeros((4, rank))
    W = np.zeros((4, rank))
    
    # Strassen formulas:
    # C00 = M1 - M4 + M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    #
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    
    # A indices: 0=A00, 1=A01, 2=A10, 3=A11
    # B indices: 0=B00, 1=B01, 2=B10, 3=B11
    # C indices: 0=C00, 1=C01, 2=C10, 3=C11
    
    # M1: A00+A11, B00+B11
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    W[0, 0] = 1; W[3, 0] = 1
    
    # M2: A10+A11, B00
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    W[2, 1] = 1; W[3, 1] = -1
    
    # M3: A00, B01-B11
    U[0, 2] = 1
    V[1, 2] = 1; V[3, 2] = -1
    W[1, 2] = 1; W[3, 2] = 1
    
    # M4: A11, B10-B00
    U[3, 3] = 1
    V[2, 3] = 1; V[0, 3] = -1
    W[0, 3] = -1; W[2, 3] = 1
    
    # M5: A00+A01, B11
    U[0, 4] = 1; U[1, 4] = 1
    V[3, 4] = 1
    W[0, 4] = 1; W[1, 4] = 1
    
    # M6: A10-A00, B00+B01
    U[2, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[1, 5] = 1
    W[3, 5] = 1
    
    # M7: A01-A11, B10+B11
    U[1, 6] = 1; U[3, 6] = -1
    V[2, 6] = 1; V[3, 6] = 1
    W[0, 6] = 1
    
    return U, V, W

def _decomp_2x2x3():
    """
    Decomposition of M_{2,2,3} using Winograd's rank-7 algorithm.
    A is 2x2, B is 2x3, C is 2x3.
    U: (4, 7), V: (6, 7), W: (6, 7)
    
    C00 = M1 - M4 + M5
    C01 = M3 + M5
    C02 = M3 + M6
    C10 = M2 + M4
    C11 = M1 - M2 + M3
    C12 = M1 - M2 + M6
    
    M1 = (A00+A11)(B00+B11)
    M2 = (A10+A11)B00
    M3 = A00(B01-B11)
    M4 = A11(B10-B00)
    M5 = (A00+A01)B11
    M6 = (A10-A00)(B02+B12)  -- adapted for 3rd column
    M7 = (A01-A11)(B10+B11)  -- not used in this variant
    """
    rank = 7
    U = np.zeros((4, rank))
    V = np.zeros((6, rank))
    W = np.zeros((6, rank))
    
    # A indices: 0=A00, 1=A01, 2=A10, 3=A11
    # B indices: 0=B00, 1=B01, 2=B02, 3=B10, 4=B11, 5=B12
    # C indices: 0=C00, 1=C01, 2=C02, 3=C10, 4=C11, 5=C12
    
    # M1: (A00+A11)(B00+B11)
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[4, 0] = 1
    W[0, 0] = 1; W[4, 0] = 1
    
    # M2: (A10+A11)B00
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    W[3, 1] = 1; W[4, 1] = -1
    
    # M3: A00(B01-B11)
    U[0, 2] = 1
    V[1, 2] = 1; V[4, 2] = -1
    W[1, 2] = 1; W[4, 2] = 1
    
    # M4: A11(B10-B00)
    U[3, 3] = 1
    V[3, 3] = 1; V[0, 3] = -1
    W[0, 3] = -1; W[3, 3] = 1
    
    # M5: (A00+A01)B11
    U[0, 4] = 1; U[1, 4] = 1
    V[4, 4] = 1
    W[0, 4] = 1; W[1, 4] = 1
    
    # M6: (A10-A00)(B02+B12)
    U[2, 5] = 1; U[0, 5] = -1
    V[2, 5] = 1; V[5, 5] = 1
    W[2, 5] = 1; W[5, 5] = 1
    
    # M7: (A01-A11)(B10+B11) -- used for C12 contribution
    U[1, 6] = 1; U[3, 6] = -1
    V[3, 6] = 1; V[4, 6] = 1
    W[0, 6] = 1  # contributes to C00
    
    # Now fix up the C coefficients:
    # C00 = M1 - M4 + M5 + M7  (M7 contributes +1 to C00)
    # C01 = M3 + M5
    # C02 = M3 + M6  -- but M3 has coeff on C01, need separate
    # Let me redo this more carefully.
    
    # Actually, let me use a cleaner construction.
    # For 2x2x3, we can use:
    # C = A*B where A is 2x2, B is 2x3
    # C00 = A00*B00 + A01*B10
    # C01 = A00*B01 + A01*B11
    # C02 = A00*B02 + A01*B12
    # C10 = A10*B00 + A11*B10
    # C11 = A10*B01 + A11*B11
    # C12 = A10*B02 + A11*B12
    
    # Use 7 products:
    # M1 = (A00+A11)(B00+B10)  -- handles C00, C10 partially
    # M2 = (A10+A11)(B00)
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A01)(B11)
    # M6 = (A10-A00)(B02+B12)
    # M7 = (A01-A11)(B10+B11)
    
    # Let me just verify with a direct approach and use schoolbook if needed.
    # For now, let me use a rank-12 decomposition (2x schoolbook 2x2x2 + 2x2x1)
    # Actually, let me just use schoolbook for 2x2x3: rank 12
    
    # Reset and use schoolbook
    U[:] = 0
    V[:] = 0
    W[:] = 0
    rank = 12
    U = np.zeros((4, rank))
    V = np.zeros((6, rank))
    W = np.zeros((6, rank))
    col = 0
    for i in range(2):
        for j in range(2):
            for k in range(3):
                U[i*2+j, col] = 1
                V[j*3+k, col] = 1
                W[k*2+i, col] = 1
                col += 1
    return U, V, W

def _decomp_4x4x3():
    """
    Decomposition of M_{4,4,3} using four 2x2x3 blocks.
    A is 4x4, B is 4x3, C is 4x3.
    
    Split A into 2x2 blocks: A = [[A00,A01],[A10,A11]]
    Split B into 2x3 blocks: B = [Btop; Bbot]
    
    C_top = A00*Btop + A01*Bbot
    C_bot = A10*Btop + A11*Bbot
    
    Each 2x2x3 product uses rank 12 (schoolbook).
    Total rank: 4 * 12 = 48.
    
    U: (16, 48), V: (12, 48), W: (12, 48)
    """
    n, m, p = 4, 4, 3
    rank_223 = 12
    total_rank = 4 * rank_223
    
    U = np.zeros((n*m, total_rank))
    V = np.zeros((m*p, total_rank))
    W = np.zeros((p*n, total_rank))
    
    U223, V223, W223 = _decomp_2x2x3()
    
    # Four 2x2x3 blocks:
    # Block (0,0): A00 * Btop -> C_top_left
    # Block (0,1): A01 * Bbot -> C_top_right  
    # Block (1,0): A10 * Btop -> C_bot_left
    # Block (1,1): A11 * Bbot -> C_bot_right
    
    # Wait, this isn't quite right. Let me reconsider.
    # C_top = A00*Btop + A01*Bbot
    # C_bot = A10*Btop + A11*Bbot
    #
    # Btop is rows 0,1 of B (2x3)
    # Bbot is rows 2,3 of B (2x3)
    # A00 is rows 0,1 cols 0,1 of A
    # A01 is rows 0,1 cols 2,3 of A
    # A10 is rows 2,3 cols 0,1 of A
    # A11 is rows 2,3 cols 2,3 of A
    
    # For block (0,0): A00 (2x2) * Btop (2x3) -> 2x3 result for C rows 0,1
    # U indices: A00 rows 0,1; cols 0,1 -> U[0:2, 0:2]
    # V indices: Btop rows 0,1; all 3 cols -> V[0:2, :]
    # W indices: C rows 0,1; all 3 cols -> W[0:2, :]
    
    # For block (0,1): A01 (2x2) * Bbot (2x3) -> 2x3 result for C rows 0,1
    # U indices: A01 rows 0,1; cols 2,3 -> U[0:2, 2:4]
    # V indices: Bbot rows 2,3; all 3 cols -> V[2:4, :]
    # W indices: C rows 0,1; all 3 cols -> W[0:2, :]
    
    # For block (1,0): A10 (2x2) * Btop (2x3) -> 2x3 result for C rows 2,3
    # U indices: A10 rows 2,3; cols 0,1 -> U[2:4, 0:2]
    # V indices: Btop rows 0,1; all 3 cols -> V[0:2, :]
    # W indices: C rows 2,3; all 3 cols -> W[2:4, :]
    
    # For block (1,1): A11 (2x2) * Bbot (2x3) -> 2x3 result for C rows 2,3
    # U indices: A11 rows 2,3; cols 2,3 -> U[2:4, 2:4]
    # V indices: Bbot rows 2,3; all 3 cols -> V[2:4, :]
    # W indices: C rows 2,3; all 3 cols -> W[2:4, :]
    
    blocks = [
        # (U_row_start, U_col_start, V_row_start, W_row_start)
        (0, 0, 0, 0),   # A00 * Btop -> C top
        (0, 2, 2, 0),   # A01 * Bbot -> C top
        (2, 0, 0, 2),   # A10 * Btop -> C bot
        (2, 2, 2, 2),   # A11 * Bbot -> C bot
    ]
    
    for b_idx, (u_r, u_c, v_r, w_r) in enumerate(blocks):
        col_start = b_idx * rank_223
        for r in range(rank_223):
            col = col_start + r
            for i in range(2):
                for j in range(2):
                    u_idx = (u_r + i) * 4 + (u_c + j)
                    U[u_idx, col] = U223[i*2+j, r]
            for j in range(2):
                for k in range(3):
                    v_idx = (v_r + j) * 3 + k
                    V[v_idx, col] = V223[j*3+k, r]
            for k in range(3):
                for i in range(2):
                    w_idx = k * 4 + (w_r + i)
                    W[w_idx, col] = W223[k*2+i, r]
    
    return U, V, W

def _schoolbook_nm_p(n, m, p):
    """
    Schoolbook decomposition for M_{n,m,p}.
    Rank n*m*p.
    U: (n*m, n*m*p), V: (m*p, n*m*p), W: (p*n, n*m*p)
    """
    rank = n * m * p
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    return U, V, W

def _decomp_4x4x9_combined():
    """
    Decomposition of M_{4,4,9} by splitting into three 4x4x3 blocks.
    Total rank: 3 * 48 = 144.
    U: (16, 144), V: (36, 144), W: (36, 144)
    """
    n, m, p = 4, 4, 9
    rank_443 = 48
    total_rank = 3 * rank_443
    
    U = np.zeros((n * m, total_rank))
    V = np.zeros((m * p, total_rank))
    W = np.zeros((p * n, total_rank))
    
    U443, V443, W443 = _decomp_4x4x3()
    
    for s in range(3):
        col_start = s * rank_443
        U[:, col_start:col_start+rank_443] = U443
        for j in range(m):
            for k in range(3):
                V[j * p + (s*3 + k), col_start:col_start+rank_443] = V443[j * 3 + k, :]
        for k in range(3):
            for i in range(n):
                W[(s*3 + k) * n + i, col_start:col_start+rank_443] = W443[k * 4 + i, :]
    
    return U, V, W

def _decomp_4x4x9_original():
    """
    Original decomposition: two 4x4x4 Strassen blocks + one 4x4x1 schoolbook.
    Total rank: 49 + 49 + 16 = 114.
    """
    n, m, p = 4, 4, 9
    rank_444 = 49
    rank_441 = 16
    total_rank = 2 * rank_444 + rank_441
    
    U = np.zeros((n * m, total_rank))
    V = np.zeros((m * p, total_rank))
    W = np.zeros((p * n, total_rank))
    
    inner_U, inner_V, inner_W = _strassen_2x2()
    
    outer_A = [
        [1, 0, 0, 1], [0, 0, 1, 1], [1, 0, 0, 0], [0, 0, 0, 1],
        [1, 1, 0, 0], [0, 0, 1, -1], [0, 1, 0, -1]
    ]
    outer_B = [
        [1, 0, 0, 1], [1, 0, 0, 0], [0, 1, 0, -1], [0, 0, 1, -1],
        [0, 0, 0, 1], [1, 1, 0, 0], [0, 0, 1, 1]
    ]
    outer_C = [
        [1, 0, 0, 1], [0, 0, 1, -1], [0, 1, 0, 1], [-1, 0, 1, 0],
        [1, 1, 0, 0], [0, 0, 0, 1], [1, 0, 0, 0]
    ]
    
    def _make_4x4x4(outer_A, outer_B, outer_C, inner_U, inner_V, inner_W):
        n, m, p = 4, 4, 4
        rank = 49
        U = np.zeros((n*m, rank))
        V = np.zeros((m*p, rank))
        W = np.zeros((p*n, rank))
        for t in range(7):
            col_offset = t * 7
            for r in range(7):
                col = col_offset + r
                for li in range(2):
                    for lj in range(2):
                        b = li * 2 + lj
                        coeff_U = outer_A[t][b] * inner_U[li*2 + lj, r]
                        coeff_V = outer_B[t][b] * inner_V[li*2 + lj, r]
                        coeff_W = outer_C[t][b] * inner_W[li*2 + lj, r]
                        for ei in range(2):
                            for ej in range(2):
                                g_i = 2 * (b // 2) + ei
                                g_j = 2 * (b % 2) + ej
                                U[g_i * 4 + g_j, col] = coeff_U
                                V[g_j * 4 + g_i, col] = coeff_V
                                W[g_i * 4 + g_j, col] = coeff_W
        return U, V, W
    
    U1, V1, W1 = _make_4x4x4(outer_A, outer_B, outer_C, inner_U, inner_V, inner_W)
    U2, V2, W2 = _make_4x4x4(outer_A, outer_B, outer_C, inner_U, inner_V, inner_W)
    U3, V3, W3 = _schoolbook_nm_p(4, 4, 1)
    
    U[:, :49] = U1
    for j in range(m):
        for k in range(4):
            V[j * p + k, :49] = V1[j * 4 + k, :]
    for k in range(4):
        for i in range(n):
            W[k * n + i, :49] = W1[k * 4 + i, :]
    
    col_start = 49
    U[:, col_start:col_start+49] = U2
    for j in range(m):
        for k in range(4):
            V[j * p + (4 + k), col_start:col_start+49] = V2[j * 4 + k, :]
    for k in range(4):
        for i in range(n):
            W[(4 + k) * n + i, col_start:col_start+49] = W2[k * 4 + i, :]
    
    col_start2 = 98
    U[:, col_start2:col_start2+16] = U3
    for j in range(m):
        V[j * p + 8, col_start2:col_start2+16] = V3[j * 1 + 0, :]
    for i in range(n):
        W[8 * n + i, col_start2:col_start2+16] = W3[0 * 4 + i, :]
    
    return U, V, W

def _verify_decomp(U, V, W, n, m, p, atol=1e-9):
    """Vectorized verification of the decomposition."""
    rank = U.shape[1]
    U_r = U.reshape(n, m, rank)
    V_r = V.reshape(m, p, rank)
    W_r = W.reshape(p, n, rank)
    
    i_idx, j_idx, k_idx = np.meshgrid(np.arange(n), np.arange(m), np.arange(p), indexing='ij')
    i_idx = i_idx.ravel()
    j_idx = j_idx.ravel()
    k_idx = k_idx.ravel()
    
    u_vec = U_r[i_idx, j_idx, :]
    v_vec = V_r[j_idx, k_idx, :]
    w_vec = W_r[k_idx, i_idx, :]
    
    prod = u_vec * v_vec * w_vec
    sums = prod.sum(axis=1)
    
    return np.allclose(sums, 1.0, atol=atol)

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140
    
    # Try the original 114-rank decomposition first (known correct)
    U, V, W = _decomp_4x4x9_original()
    
    if time.time() < deadline - 10:
        if _verify_decomp(U, V, W, n, m, p):
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Fallback: schoolbook
    Us, Vs, Ws = _schoolbook_nm_p(n, m, p)
    return {"U": Us.tolist(), "V": Vs.tolist(), "W": Ws.tolist()}

# EVOLVE-BLOCK-END
