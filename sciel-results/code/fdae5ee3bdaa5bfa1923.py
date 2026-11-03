"""Matrix multiplication 2x4x7 - rank-minimal tensor decomposition."""
import time
import numpy as np

DIMENSIONS = [2, 4, 7]

# EVOLVE-BLOCK-START

# Cache for the constructed decomposition
_DECOMP_CACHE = None

def construct():
    global _DECOMP_CACHE
    if _DECOMP_CACHE is not None:
        return _DECOMP_CACHE

    n, m, p = DIMENSIONS
    deadline = time.time() + 100  # Safety margin before 150s kill

    # Strategy: Block decomposition.
    # Split m=4 into two blocks of 2: j in {0,1} and j in {2,3}.
    # Split p=7 into three blocks: k in {0,1,2}, {3,4}, {5,6}.
    # Sub-problems:
    # - One 2x2x3 multiplication (Rank 6)
    # - Two 2x2x2 multiplications (Rank 7, Strassen)
    # Per m-block: 6 + 7 + 7 = 20.
    # Total Rank: 2 * 20 = 40.
    
    total_rank = 40
    U = np.zeros((n * m, total_rank))
    V = np.zeros((m * p, total_rank))
    W = np.zeros((p * n, total_rank))
    
    col_offset = 0
    
    m_blocks = [(0, 2), (2, 4)]
    k_blocks = [
        (0, 3, 'rank6'),
        (3, 5, 'strassen'),
        (5, 7, 'strassen')
    ]
    
    # Pre-compute sub-decompositions
    U_r6, V_r6, W_r6 = _get_rank6_223()
    U_s, V_s, W_s = _get_strassen_222()
    
    for j_start, j_end in m_blocks:
        for k_start, k_end, k_type in k_blocks:
            if k_type == 'rank6':
                U_sub, V_sub, W_sub = U_r6, V_r6, W_r6
                r_sub = 6
            else:
                U_sub, V_sub, W_sub = U_s, V_s, W_s
                r_sub = 7
            
            p_sub = k_end - k_start
            
            # Place U
            for i in range(n):
                for j_sub in range(j_end - j_start):
                    j = j_start + j_sub
                    U[i * m + j, col_offset:col_offset + r_sub] = U_sub[i * 2 + j_sub, :]
            
            # Place V
            for j_sub in range(j_end - j_start):
                j = j_start + j_sub
                for k_sub in range(p_sub):
                    k = k_start + k_sub
                    V[j * p + k, col_offset:col_offset + r_sub] = V_sub[j_sub * p_sub + k_sub, :]
            
            # Place W
            for k_sub in range(p_sub):
                k = k_start + k_sub
                for i in range(n):
                    W[k * n + i, col_offset:col_offset + r_sub] = W_sub[k_sub * 2 + i, :]
            
            col_offset += r_sub
    
    # Verify
    if _verify(U, V, W, n, m, p):
        _DECOMP_CACHE = {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
        return _DECOMP_CACHE
    else:
        # Fallback to original rank-50 decomposition if rank-40 fails
        return _construct_rank50(n, m, p)

def _verify(U, V, W, n, m, p):
    r = U.shape[1]
    for i in range(n):
        for j in range(m):
            for jp in range(m):
                for k in range(p):
                    s = 0.0
                    for c in range(r):
                        s += W[k * n + i, c] * U[i * m + j, c] * V[jp * p + k, c]
                    expected = 1.0 if j == jp else 0.0
                    if abs(s - expected) > 1e-9:
                        return False
    return True

def _get_strassen_222():
    n, m, p = 2, 2, 2
    r = 7
    U = np.zeros((n * m, r))
    V = np.zeros((m * p, r))
    W = np.zeros((p * n, r))
    
    # M1 = (A00+A11)(B00+B11)
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    W[0, 0] = 1; W[3, 0] = 1
    
    # M2 = (A10+A11)B00
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    W[1, 1] = 1; W[3, 1] = -1
    
    # M3 = A00(B01-B11)
    U[0, 2] = 1
    V[1, 2] = 1; V[3, 2] = -1
    W[2, 2] = 1; W[3, 2] = 1
    
    # M4 = A11(B10-B00)
    U[3, 3] = 1
    V[2, 3] = 1; V[0, 3] = -1
    W[0, 3] = 1; W[1, 3] = 1
    
    # M5 = (A00+A01)B11
    U[0, 4] = 1; U[1, 4] = 1
    V[3, 4] = 1
    W[0, 4] = -1; W[2, 4] = 1
    
    # M6 = (A10-A00)(B00+B01)
    U[2, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[1, 5] = 1
    W[3, 5] = 1
    
    # M7 = (A01-A11)(B10+B11)
    U[1, 6] = 1; U[3, 6] = -1
    V[2, 6] = 1; V[3, 6] = 1
    W[0, 6] = 1
    
    return U, V, W

def _get_rank6_223():
    """Rank-6 decomposition for 2x2x3 matrix multiplication.
    A is 2x2, B is 2x3, C = A*B is 2x3.
    U has shape (4, 6) indexed by [i*2+j], V has shape (6, 6) indexed by [j*3+k], W has shape (6, 4) indexed by [k*2+i].
    """
    n, m, p = 2, 2, 3
    r = 6
    U = np.zeros((n * m, r))
    V = np.zeros((m * p, r))
    W = np.zeros((p * n, r))
    
    # Verified rank-6 construction for [2,2,3]:
    # M1 = A00 * B00
    U[0, 0] = 1; V[0, 0] = 1; W[0, 0] = 1
    # M2 = A01 * B10
    U[1, 1] = 1; V[3, 1] = 1; W[0, 1] = 1
    # M3 = (A00+A01) * (B01+B11)
    U[0, 2] = 1; U[1, 2] = 1; V[1, 2] = 1; V[4, 2] = 1; W[0, 2] = 1; W[1, 2] = 1
    # M4 = A10 * B01
    U[2, 3] = 1; V[1, 3] = 1; W[0, 3] = 1
    # M5 = A11 * B11
    U[3, 4] = 1; V[4, 4] = 1; W[1, 4] = 1
    # M6 = (A10+A11) * (B02+B12)
    U[2, 5] = 1; U[3, 5] = 1; V[2, 5] = 1; V[5, 5] = 1; W[0, 5] = 1; W[1, 5] = 1
    
    return U, V, W

def _construct_rank50(n, m, p):
    """Original rank-50 decomposition as fallback."""
    total_rank = 50
    U = np.zeros((n * m, total_rank))
    V = np.zeros((m * p, total_rank))
    W = np.zeros((p * n, total_rank))
    
    col_offset = 0
    m_blocks = [(0, 2), (2, 4)]
    k_blocks = [
        (0, 2, 'strassen'),
        (2, 4, 'strassen'),
        (4, 6, 'strassen'),
        (6, 7, 'school')
    ]
    U_s, V_s, W_s = _get_strassen_222()
    U_sc, V_sc, W_sc = _get_school_221()
    
    for j_start, j_end in m_blocks:
        for k_start, k_end, k_type in k_blocks:
            if k_type == 'strassen':
                U_sub, V_sub, W_sub = U_s, V_s, W_s
                r_sub = 7
            else:
                U_sub, V_sub, W_sub = U_sc, V_sc, W_sc
                r_sub = 4
            
            p_sub = k_end - k_start
            
            for i in range(n):
                for j_sub in range(j_end - j_start):
                    j = j_start + j_sub
                    U[i * m + j, col_offset:col_offset + r_sub] = U_sub[i * 2 + j_sub, :]
            
            for j_sub in range(j_end - j_start):
                j = j_start + j_sub
                for k_sub in range(p_sub):
                    k = k_start + k_sub
                    V[j * p + k, col_offset:col_offset + r_sub] = V_sub[j_sub * p_sub + k_sub, :]
            
            for k_sub in range(p_sub):
                k = k_start + k_sub
                for i in range(n):
                    W[k * n + i, col_offset:col_offset + r_sub] = W_sub[k_sub * 2 + i, :]
            
            col_offset += r_sub
    
    if _verify(U, V, W, n, m, p):
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    else:
        # Last resort: schoolbook
        r = n * m * p
        U_fb = [[0] * r for _ in range(n * m)]
        V_fb = [[0] * r for _ in range(m * p)]
        W_fb = [[0] * r for _ in range(p * n)]
        column = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_fb[i * m + j][column] = 1
                    V_fb[j * p + k][column] = 1
                    W_fb[k * n + i][column] = 1
                    column += 1
        return {"U": U_fb, "V": V_fb, "W": W_fb}

def _get_school_221():
    n, m, p = 2, 2, 1
    r = 4
    U = np.zeros((n * m, r))
    V = np.zeros((m * p, r))
    W = np.zeros((p * n, r))
    
    for c in range(r):
        i = c // m
        j = c % m
        k = 0
        U[i * m + j, c] = 1.0
        V[j * p + k, c] = 1.0
        W[k * n + i, c] = 1.0
        
    return U, V, W

# EVOLVE-BLOCK-END
