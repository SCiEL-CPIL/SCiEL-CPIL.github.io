"""Rewritten program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
from fractions import Fraction
import numpy as np

DIMENSIONS = [4, 5, 6]

# EVOLVE-BLOCK-START

def _build_strassen_2x2():
    """
    Build the 2x2 Strassen decomposition.
    Returns (U, V, W) where:
      U has shape (4, 7), V has shape (4, 7), W has shape (4, 7)
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

def _build_strassen_4x4():
    """
    Build 4x4 Strassen decomposition. Rank 49.
    """
    U2, V2, W2 = _build_strassen_2x2()
    
    rank = 49
    U = [[0]*rank for _ in range(16)]
    V = [[0]*rank for _ in range(16)]
    W = [[0]*rank for _ in range(16)]
    
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
                            U[u_row][col] += coeff_a * sigma
            
            for lb_j in range(2):
                for lb_k in range(2):
                    local_b_row = lb_j * 2 + lb_k
                    coeff_b = V2[local_b_row][t]
                    if coeff_b != 0:
                        for (bj, bk), sigma in b_blocks:
                            global_j = bj * 2 + lb_j
                            global_k = bk * 2 + lb_k
                            v_row = global_j * 4 + global_k
                            V[v_row][col] += coeff_b * sigma
            
            for lc_k in range(2):
                for lc_i in range(2):
                    local_c_row = lc_k * 2 + lc_i
                    coeff_c = W2[local_c_row][t]
                    if coeff_c != 0:
                        for (bi, bk), sigma in c_blocks:
                            global_k = bk * 2 + lc_k
                            global_i = bi * 2 + lc_i
                            w_row = global_k * 4 + global_i
                            W[w_row][col] += coeff_c * sigma
    
    return U, V, W

def _build_strassen_4x4x2():
    """
    Build 4x4x2 decomposition using 4 Strassen 2x2x2 blocks.
    A is 4x4, B is 4x2, C is 4x2.
    Rank = 28.
    """
    U2, V2, W2 = _build_strassen_2x2()
    
    rank = 28
    U = [[0]*rank for _ in range(16)] # 4x4 = 16 rows
    V = [[0]*rank for _ in range(8)]  # 4x2 = 8 rows
    W = [[0]*rank for _ in range(8)]  # 4x2 = 8 rows (p*n = 2*4)
    
    def add_strassen_block(a_i_start, a_j_start, b_j_start, c_i_start, c_k_start, base_rank):
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
                        u_row = global_i * 4 + global_j # m=4
                        U[u_row][col] += coeff_a
                        
            # B part
            for lj in range(2):
                for lk in range(2):
                    local_b_row = lj * 2 + lk
                    coeff_b = V2[local_b_row][t]
                    if coeff_b != 0:
                        global_j = b_j_start + lj
                        global_k = c_k_start + lk # B's col is C's col
                        v_row = global_j * 2 + global_k # p=2
                        V[v_row][col] += coeff_b
                        
            # C part
            for lk in range(2):
                for li in range(2):
                    local_c_row = lk * 2 + li
                    coeff_c = W2[local_c_row][t]
                    if coeff_c != 0:
                        global_k = c_k_start + lk
                        global_i = c_i_start + li
                        w_row = global_k * 4 + global_i # n=4
                        W[w_row][col] += coeff_c

    # 1. A11 * B11 -> C11
    add_strassen_block(0, 0, 0, 0, 0, 0)
    
    # 2. A12 * B21 -> C11
    add_strassen_block(0, 2, 2, 0, 0, 7)
    
    # 3. A21 * B11 -> C21
    add_strassen_block(2, 0, 0, 2, 0, 14)
    
    # 4. A22 * B21 -> C21
    add_strassen_block(2, 2, 2, 2, 0, 21)
    
    return U, V, W

def _build_schoolbook(n, m, p):
    """
    Build schoolbook decomposition for n x m * m x p -> n x p.
    Returns U, V, W with rank n*m*p.
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

def _build_strassen_4x2x2():
    """
    Build 4x2x2 decomposition.
    A is 4x2, B is 2x2, C is 4x2.
    Rank = 14 (2 * 7).
    """
    U2, V2, W2 = _build_strassen_2x2()
    
    rank = 14
    U = [[0]*rank for _ in range(8)]  # 4x2 = 8 rows
    V = [[0]*rank for _ in range(4)]  # 2x2 = 4 rows
    W = [[0]*rank for _ in range(8)]  # 2x4 = 8 rows (p*n = 2*4)
    
    # Block 1: A11 (2x2) * B11 (2x2) -> C11 (2x2)
    for t in range(7):
        col = t
        for li in range(2):
            for lj in range(2):
                local_a_row = li * 2 + lj
                coeff_a = U2[local_a_row][t]
                if coeff_a != 0:
                    global_i = li
                    global_j = lj
                    u_row = global_i * 2 + global_j # m=2
                    U[u_row][col] += coeff_a
        for lj in range(2):
            for lk in range(2):
                local_b_row = lj * 2 + lk
                coeff_b = V2[local_b_row][t]
                if coeff_b != 0:
                    global_j = lj
                    global_k = lk
                    v_row = global_j * 2 + global_k # p=2
                    V[v_row][col] += coeff_b
        for lk in range(2):
            for li in range(2):
                local_c_row = lk * 2 + li
                coeff_c = W2[local_c_row][t]
                if coeff_c != 0:
                    global_k = lk
                    global_i = li
                    w_row = global_k * 4 + global_i # n=4
                    W[w_row][col] += coeff_c
                    
    # Block 2: A21 (2x2) * B11 (2x2) -> C21 (2x2)
    for t in range(7):
        col = 7 + t
        for li in range(2):
            for lj in range(2):
                local_a_row = li * 2 + lj
                coeff_a = U2[local_a_row][t]
                if coeff_a != 0:
                    global_i = 2 + li
                    global_j = lj
                    u_row = global_i * 2 + global_j # m=2
                    U[u_row][col] += coeff_a
        for lj in range(2):
            for lk in range(2):
                local_b_row = lj * 2 + lk
                coeff_b = V2[local_b_row][t]
                if coeff_b != 0:
                    global_j = lj
                    global_k = lk
                    v_row = global_j * 2 + global_k # p=2
                    V[v_row][col] += coeff_b
        for lk in range(2):
            for li in range(2):
                local_c_row = lk * 2 + li
                coeff_c = W2[local_c_row][t]
                if coeff_c != 0:
                    global_k = lk
                    global_i = 2 + li
                    w_row = global_k * 4 + global_i # n=4
                    W[w_row][col] += coeff_c
                    
    return U, V, W

def _combine_decompositions(decomps, n, m, p):
    """
    Combine a list of decompositions into one.
    """
    total_rank = sum(len(d[0][0]) for d in decomps)
    
    U = [[0]*total_rank for _ in range(n*m)]
    V = [[0]*total_rank for _ in range(m*p)]
    W = [[0]*total_rank for _ in range(p*n)]
    
    current_rank = 0
    for U_sub, V_sub, W_sub, (i_s, i_e), (j_s, j_e), (k_s, k_e) in decomps:
        n_sub = i_e - i_s
        m_sub = j_e - j_s
        p_sub = k_e - k_s
        rank_sub = len(U_sub[0])
        
        for r_sub in range(n_sub * m_sub):
            i_loc = r_sub // m_sub
            j_loc = r_sub % m_sub
            i_glob = i_s + i_loc
            j_glob = j_s + j_loc
            r_glob = i_glob * m + j_glob
            
            for c_sub in range(rank_sub):
                U[r_glob][current_rank + c_sub] = U_sub[r_sub][c_sub]
        
        for r_sub in range(m_sub * p_sub):
            j_loc = r_sub // p_sub
            k_loc = r_sub % p_sub
            j_glob = j_s + j_loc
            k_glob = k_s + k_loc
            r_glob = j_glob * p + k_glob
            
            for c_sub in range(rank_sub):
                V[r_glob][current_rank + c_sub] = V_sub[r_sub][c_sub]
        
        for r_sub in range(p_sub * n_sub):
            k_loc = r_sub // n_sub
            i_loc = r_sub % n_sub
            k_glob = k_s + k_loc
            i_glob = i_s + i_loc
            r_glob = k_glob * n + i_glob
            
            for c_sub in range(rank_sub):
                W[r_glob][current_rank + c_sub] = W_sub[r_sub][c_sub]
        
        current_rank += rank_sub
    
    return U, V, W

def _verify_decomposition(U, V, W, n, m, p):
    """
    Verify that U, V, W correctly decompose the matrix multiplication tensor.
    Returns True if valid, False otherwise.
    """
    try:
        rank = len(U[0])
        # Convert to numpy arrays for efficient tensor contraction
        U_np = np.array(U, dtype=float)
        V_np = np.array(V, dtype=float)
        W_np = np.array(W, dtype=float)
        
        # Reshape to 3D tensors
        # U: (n*m, rank) -> (n, m, rank)
        U_3d = U_np.reshape(n, m, rank)
        # V: (m*p, rank) -> (m, p, rank)
        V_3d = V_np.reshape(m, p, rank)
        # W: (p*n, rank) -> (p, n, rank)
        W_3d = W_np.reshape(p, n, rank)
        
        # Compute tensor T[i,j,k] = sum_r U[i,j,r] * V[j,k,r] * W[k,i,r]
        # For matrix multiplication, T[i,j,k] should be 1 for all i,j,k
        
        if n <= 10 and m <= 10 and p <= 10:
            T = np.zeros((n, m, p), dtype=float)
            for r in range(rank):
                for i in range(n):
                    for j in range(m):
                        u_val = U[i*m + j][r]
                        if u_val == 0:
                            continue
                        for k in range(p):
                            v_val = V[j*p + k][r]
                            if v_val == 0:
                                continue
                            w_val = W[k*n + i][r]
                            if w_val == 0:
                                continue
                            T[i, j, k] += u_val * v_val * w_val
            
            expected = np.ones((n, m, p), dtype=float)
            if not np.allclose(T, expected, atol=1e-9):
                return False
        else:
            for _ in range(100):
                i = random.randint(0, n-1)
                j = random.randint(0, m-1)
                k = random.randint(0, p-1)
                t_val = 0.0
                for r in range(rank):
                    u_val = U[i*m + j][r]
                    v_val = V[j*p + k][r]
                    w_val = W[k*n + i][r]
                    t_val += u_val * v_val * w_val
                if abs(t_val - 1.0) > 1e-9:
                    return False
        
        return True
    except Exception:
        return False

def _try_optimize_blocks(n, m, p, deadline):
    """
    Try to find a better block decomposition than the default.
    Default: 4+1 split in m, 4+2 split in p.
    """
    start_time = time.time()
    
    # Current best rank: 49 + 24 + 28 = 101
    best_rank = 101
    best_result = None
    
    # Try different splits
    # m can be split as: 1+4, 2+3, 3+2, 4+1
    # p can be split as: 1+5, 2+4, 3+3, 4+2, 5+1
    
    for m1 in range(1, m):
        m2 = m - m1
        if m1 <= 0 or m2 <= 0:
            continue
            
        for p1 in range(1, p):
            p2 = p - p1
            if p1 <= 0 or p2 <= 0:
                continue
                
            # Check time
            if time.time() > deadline - 30:
                break
                
            try:
                # Block 1: n x m1 x p1
                if m1 == 1:
                    rank1 = n * m1 * p1  # schoolbook
                    U1, V1, W1 = _build_schoolbook(n, m1, p1)
                else:
                    # For m1 >= 2, use Strassen if possible
                    # Strassen 4x4x4 = 49, but we might not have exactly 4x4
                    # For now, use schoolbook for non-4x4 blocks
                    # Or try to pad
                    if m1 <= 4 and p1 <= 4:
                        # Can use Strassen if both are 2x2 or 4x4
                        if m1 == 2 and p1 == 2:
                            # Use 2x2 Strassen = 7
                            U2, V2, W2 = _build_strassen_2x2()
                            U1, V1, W1 = U2, V2, W2
                            rank1 = 7
                        elif m1 == 4 and p1 == 4:
                            U1, V1, W1 = _build_strassen_4x4()
                            rank1 = 49
                        elif m1 == 2 and p1 == 4:
                            # 2x4x4: use 2 blocks of 2x2x4
                            # 2x2x4 = 2 * 7 = 14
                            U1, V1, W1 = _build_schoolbook(n, m1, p1)
                            rank1 = n * m1 * p1
                        elif m1 == 4 and p1 == 2:
                            U1, V1, W1 = _build_strassen_4x4x2()
                            rank1 = 28
                        elif m1 == 2 and p1 == 2:
                            U1, V1, W1 = _build_strassen_2x2()
                            rank1 = 7
                        else:
                            U1, V1, W1 = _build_schoolbook(n, m1, p1)
                            rank1 = n * m1 * p1
                    else:
                        U1, V1, W1 = _build_schoolbook(n, m1, p1)
                        rank1 = n * m1 * p1
                
                # Block 2: n x m2 x p2
                if m2 == 1:
                    rank2 = n * m2 * p2
                    U2, V2, W2 = _build_schoolbook(n, m2, p2)
                else:
                    if m2 <= 4 and p2 <= 4:
                        if m2 == 2 and p2 == 2:
                            U2, V2, W2 = _build_strassen_2x2()
                            rank2 = 7
                        elif m2 == 4 and p2 == 4:
                            U2, V2, W2 = _build_strassen_4x4()
                            rank2 = 49
                        elif m2 == 2 and p2 == 4:
                            U2, V2, W2 = _build_schoolbook(n, m2, p2)
                            rank2 = n * m2 * p2
                        elif m2 == 4 and p2 == 2:
                            U2, V2, W2 = _build_strassen_4x4x2()
                            rank2 = 28
                        else:
                            U2, V2, W2 = _build_schoolbook(n, m2, p2)
                            rank2 = n * m2 * p2
                    else:
                        U2, V2, W2 = _build_schoolbook(n, m2, p2)
                        rank2 = n * m2 * p2
                
                total_rank = rank1 + rank2
                
                if total_rank < best_rank:
                    # Combine the two blocks
                    decomp1 = (U1, V1, W1, (0, n), (0, m1), (0, p1))
                    decomp2 = (U2, V2, W2, (0, n), (m1, m), (p1, p))
                    
                    U, V, W = _combine_decompositions([decomp1, decomp2], n, m, p)
                    
                    if _verify_decomposition(U, V, W, n, m, p):
                        best_rank = total_rank
                        best_result = {"U": U, "V": V, "W": W}
                        
            except Exception:
                pass
    
    return best_result, best_rank

def _build_strassen_3x3():
    """
    Build 3x3 Strassen decomposition. Rank 23.
    This is a known optimal decomposition for 3x3 matrix multiplication.
    """
    # This is a complex decomposition. For now, return schoolbook as placeholder.
    # A proper 3x3 Strassen requires 23 multiplications.
    # Since we don't have the explicit coefficients memorized, we fall back to schoolbook.
    # However, 3x3 schoolbook is 27.
    # If we can find a 23-rank decomposition, we could use it.
    # For safety, we return schoolbook.
    return _build_schoolbook(3, 3, 3)

def construct():
    start_time = time.time()
    deadline = start_time + 110.0
    
    n, m, p = DIMENSIONS
    
    try:
        # First, try to optimize the block decomposition
        # But this might take too long, so let's just use the known good decomposition
        
        # 1. Build 4x4x4 Strassen (Rank 49)
        U1, V1, W1 = _build_strassen_4x4()
        decomp1 = (U1, V1, W1, (0, 4), (0, 4), (0, 4))
        
        # 2. Build 4x1x6 Schoolbook (Rank 24)
        U2, V2, W2 = _build_schoolbook(4, 1, 6)
        decomp2 = (U2, V2, W2, (0, 4), (4, 5), (0, 6))
        
        # 3. Build 4x4x2 Strassen (Rank 28)
        U3, V3, W3 = _build_strassen_4x4x2()
        decomp3 = (U3, V3, W3, (0, 4), (0, 4), (4, 6))
        
        U, V, W = _combine_decompositions([decomp1, decomp2, decomp3], n, m, p)
        
        # Verify the decomposition
        if _verify_decomposition(U, V, W, n, m, p):
            return {"U": U, "V": V, "W": W}
        else:
            # If verification fails, fall back to schoolbook
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
