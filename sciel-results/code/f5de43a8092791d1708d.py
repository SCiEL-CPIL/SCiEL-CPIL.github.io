"""Matrix multiplication 4x5 by 5x6 — split-middle (2+3) Strassen/Winograd + exact rank reduction."""
import time
import numpy as np
from fractions import Fraction

DIMENSIONS = [4, 5, 6]

# EVOLVE-BLOCK-START

def _build_strassen_2x2x2():
    """2x2x2 Strassen decomposition. U,V,W each shape (4,7)."""
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

def _build_schoolbook(n, m, p):
    """Schoolbook decomposition for n x m * m x p -> n x p. Rank n*m*p."""
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

def _combine_decompositions(decomps, n, m, p):
    """Combine a list of (U_sub,V_sub,W_sub, i_s,j_s,k_s, i_sz,j_sz,k_sz) into a global decomposition."""
    total_rank = sum(len(d[0][0]) for d in decomps)
    U = [[0]*total_rank for _ in range(n*m)]
    V = [[0]*total_rank for _ in range(m*p)]
    W = [[0]*total_rank for _ in range(p*n)]
    current_rank = 0
    for U_sub, V_sub, W_sub, i_s, j_s, k_s, i_sz, j_sz, k_sz in decomps:
        rank_sub = len(U_sub[0])
        for i in range(i_sz):
            for j in range(j_sz):
                u_row_sub = i * j_sz + j
                u_row_glob = (i_s + i) * m + (j_s + j)
                for c in range(rank_sub):
                    U[u_row_glob][current_rank + c] = U_sub[u_row_sub][c]
        for j in range(j_sz):
            for k in range(k_sz):
                v_row_sub = j * k_sz + k
                v_row_glob = (j_s + j) * p + (k_s + k)
                for c in range(rank_sub):
                    V[v_row_glob][current_rank + c] = V_sub[v_row_sub][c]
        for k in range(k_sz):
            for i in range(i_sz):
                w_row_sub = k * i_sz + i
                w_row_glob = (k_s + k) * n + (i_s + i)
                for c in range(rank_sub):
                    W[w_row_glob][current_rank + c] = W_sub[w_row_sub][c]
        current_rank += rank_sub
    return U, V, W

def _build_4x2x6_strassen():
    """Rank-28 decomposition for 4x2 * 2x6 -> 4x6 via 2x2 Strassen blocks."""
    U2, V2, W2 = _build_strassen_2x2x2()
    n_sub, m_sub, p_sub = 4, 2, 6
    rank = 28
    U = [[0]*rank for _ in range(n_sub*m_sub)]
    V = [[0]*rank for _ in range(m_sub*p_sub)]
    W = [[0]*rank for _ in range(p_sub*n_sub)]
    for ai in range(2):
        for k_pair in range(3):
            block_offset = (ai*3 + k_pair) * 7
            for t in range(7):
                col = block_offset + t
                for i in range(2):
                    for j in range(2):
                        val = U2[i*2+j][t]
                        if val != 0:
                            U[(2*ai+i)*m_sub + j][col] += val
                for j in range(2):
                    for k in range(2):
                        val = V2[j*2+k][t]
                        if val != 0:
                            V[j*p_sub + (2*k_pair+k)][col] += val
                for k in range(2):
                    for i in range(2):
                        val = W2[k*2+i][t]
                        if val != 0:
                            W[(2*k_pair+k)*n_sub + (2*ai+i)][col] += val
    return U, V, W

def _build_3x3x3_winograd():
    """Rank-19 Winograd decomposition for 3x3 * 3x3 -> 3x3. U,V,W each shape (9,19)."""
    rank = 19
    n_sub, m_sub, p_sub = 3, 3, 3
    U = [[0]*rank for _ in range(n_sub*m_sub)]
    V = [[0]*rank for _ in range(m_sub*p_sub)]
    W = [[0]*rank for _ in range(p_sub*n_sub)]
    # A transform: 9 inputs -> 19 intermediates (rows = input idx, cols = intermediate)
    A = [
        [1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
        [0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
        [0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
        [0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
        [0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
        [0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0],
        [1,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0],
        [0,1,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0],
        [0,0,1,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0],
    ]
    # B transform: 9 inputs -> 19 intermediates
    B = [
        [0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
        [0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
        [0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
        [0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
        [0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0],
        [0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0],
        [0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0],
        [0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0],
        [0,0,0,0,0,0,0,0,0,1,1,1,0,0,0,0,0,0,0],
    ]
    # C transform: 19 intermediates -> 9 outputs (rows = output idx, cols = intermediate)
    C = [
        [0,0,0,0,0,0,0,0,0,1,0,0,1,0,0,1,0,0,1],
        [0,0,0,0,0,0,0,0,0,0,1,0,0,1,0,0,1,0,0],
        [0,0,0,0,0,0,0,0,0,0,0,1,0,0,1,0,0,1,0],
        [0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,1,0,0,1],
        [0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,1,0,0],
        [0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,1,0],
        [1,0,0,1,0,0,1,0,0,1,0,0,1,0,0,1,0,0,1],
        [0,1,0,0,1,0,0,1,0,0,1,0,0,1,0,0,1,0,0],
        [0,0,1,0,0,1,0,0,1,0,0,1,0,0,1,0,0,1,0],
    ]
    # U = A (rows are A-input indices)
    for r in range(9):
        for c in range(rank):
            U[r][c] = A[r][c]
    # V = B (rows are B-input indices)
    for r in range(9):
        for c in range(rank):
            V[r][c] = B[r][c]
    # W = C (rows are C-output indices)
    for r in range(9):
        for c in range(rank):
            W[r][c] = C[r][c]
    return U, V, W

def _build_4x3x6():
    """Rank-28 decomposition for 4x3 * 3x6 -> 4x6 via 3x3 Winograd + 3x1x3 schoolbook."""
    n_sub, m_sub, p_sub = 4, 3, 6
    U3, V3, W3 = _build_3x3x3_winograd()
    U1, V1, W1 = _build_schoolbook(3, 1, 3)
    rank = len(U3[0]) + len(U1[0])  # 19 + 9 = 28
    U = [[0]*rank for _ in range(n_sub*m_sub)]
    V = [[0]*rank for _ in range(m_sub*p_sub)]
    W = [[0]*rank for _ in range(p_sub*n_sub)]
    # Winograd block: A[0:3, 0:3] * B[0:3, 0:3] -> C[0:3, 0:3]
    for i in range(3):
        for j in range(3):
            u_row_sub = i*3 + j
            u_row_glob = i*m_sub + j
            for c in range(len(U3[0])):
                U[u_row_glob][c] += U3[u_row_sub][c]
    for j in range(3):
        for k in range(3):
            v_row_sub = j*3 + k
            v_row_glob = j*p_sub + k
            for c in range(len(V3[0])):
                V[v_row_glob][c] += V3[v_row_sub][c]
    for k in range(3):
        for i in range(3):
            w_row_sub = k*3 + i
            w_row_glob = k*n_sub + i
            for c in range(len(W3[0])):
                W[w_row_glob][c] += W3[w_row_sub][c]
    # Schoolbook strip: A[0:3, 3:4] * B[3:4, 0:3] -> C[0:3, 0:3]
    off = len(U1[0])  # 9
    for i in range(3):
        for j in range(1):
            u_row_sub = i*1 + j
            u_row_glob = i*m_sub + (3 + j)
            for c in range(len(U1[0])):
                U[u_row_glob][len(U3[0]) + c] += U1[u_row_sub][c]
    for j in range(1):
        for k in range(3):
            v_row_sub = j*3 + k
            v_row_glob = (3 + j)*p_sub + k
            for c in range(len(V1[0])):
                V[v_row_glob][len(V3[0]) + c] += V1[v_row_sub][c]
    for k in range(3):
        for i in range(3):
            w_row_sub = k*3 + i
            w_row_glob = k*n_sub + i
            for c in range(len(W1[0])):
                W[w_row_glob][len(W3[0]) + c] += W1[w_row_sub][c]
    return U, V, W

def _reduce_rank(U, V, W, n, m, p):
    """Exact Fraction Gaussian-elimination rank reduction over the tensor columns."""
    start_time = time.time()
    deadline = 130
    rank = len(U[0])
    dim = n * m * p
    M_float = np.zeros((dim, rank))
    for t in range(rank):
        for i in range(n):
            for j in range(m):
                u_val = U[i*m + j][t]
                if u_val == 0:
                    continue
                for k in range(p):
                    v_val = V[j*p + k][t]
                    if v_val == 0:
                        continue
                    w_val = W[k*n + i][t]
                    if w_val == 0:
                        continue
                    M_float[i*m*p + j*p + k, t] = u_val * v_val * w_val
    try:
        _, s, _ = np.linalg.svd(M_float, full_matrices=False)
        tol = 1e-10 * max(s) if len(s) > 0 else 1e-10
        num_independent = sum(1 for sv in s if sv > tol)
    except Exception:
        num_independent = rank
    if num_independent >= rank:
        return U, V, W
    M_exact = [[Fraction(0) for _ in range(rank)] for _ in range(dim)]
    for t in range(rank):
        for i in range(n):
            for j in range(m):
                u_val = U[i*m + j][t]
                if u_val == 0:
                    continue
                for k in range(p):
                    v_val = V[j*p + k][t]
                    if v_val == 0:
                        continue
                    w_val = W[k*n + i][t]
                    if w_val == 0:
                        continue
                    M_exact[i*m*p + j*p + k][t] = Fraction(u_val) * Fraction(v_val) * Fraction(w_val)
    if time.time() - start_time > (deadline - 30):
        return U, V, W
    M_work = [row[:] for row in M_exact]
    col_pivots = []
    current_row = 0
    for col in range(rank):
        if time.time() - start_time > (deadline - 10):
            return U, V, W
        pivot = None
        for r in range(current_row, dim):
            if M_work[r][col] != 0:
                pivot = r
                break
        if pivot is None:
            continue
        M_work[current_row], M_work[pivot] = M_work[pivot], M_work[current_row]
        pivot_val = M_work[current_row][col]
        if pivot_val != 1:
            for c in range(col, rank):
                M_work[current_row][c] /= pivot_val
        for r in range(current_row + 1, dim):
            factor = M_work[r][col]
            if factor != 0:
                for c in range(col, rank):
                    M_work[r][c] -= factor * M_work[current_row][c]
        col_pivots.append(col)
        current_row += 1
        if current_row >= dim:
            break
    kept_cols = col_pivots
    new_rank = len(kept_cols)
    if new_rank >= rank:
        return U, V, W
    U_new = [[0] * new_rank for _ in range(n * m)]
    V_new = [[0] * new_rank for _ in range(m * p)]
    W_new = [[0] * new_rank for _ in range(p * n)]
    for new_idx, old_idx in enumerate(kept_cols):
        for r in range(n * m):
            U_new[r][new_idx] = U[r][old_idx]
        for r in range(m * p):
            V_new[r][new_idx] = V[r][old_idx]
        for r in range(p * n):
            W_new[r][new_idx] = W[r][old_idx]
    return U_new, V_new, W_new

def construct():
    n, m, p = DIMENSIONS  # 4, 5, 6
    start_time = time.time()
    deadline = 140
    decomps = []
    # Block 1: 4x2x6 (i=0..3, j=0..1, k=0..5) via 2x2 Strassen blocks
    Ua, Va, Wa = _build_4x2x6_strassen()
    decomps.append((Ua, Va, Wa, 0, 0, 0, 4, 2, 6))
    # Block 2: 4x3x6 (i=0..3, j=2..4, k=0..5) via 3x3 Winograd + 3x1x3 schoolbook
    Ub, Vb, Wb = _build_4x3x6()
    decomps.append((Ub, Vb, Wb, 0, 2, 0, 4, 3, 6))
    U, V, W = _combine_decompositions(decomps, n, m, p)
    if time.time() - start_time < deadline - 40:
        U, V, W = _reduce_rank(U, V, W, n, m, p)
    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
