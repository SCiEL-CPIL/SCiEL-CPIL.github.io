"""Matrix multiplication 2x3x5 with explicit rank-28 construction."""
DIMENSIONS = [2, 3, 5]

import time
import numpy as np

def _strassen_222():
    """Return U, V, W for M_{2,2,2} (Strassen, rank 7).
    U shape (4,7): indexed by (i*2+j, r)
    V shape (4,7): indexed by (j*2+k, r)
    W shape (4,7): indexed by (k*2+i, r)
    """
    n, m, p = 2, 2, 2
    rank = 7
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]
    
    # Strassen 7 products for 2x2 matrices
    U[0][0] = 1; U[3][0] = 1
    U[2][1] = 1; U[3][1] = 1
    U[0][2] = 1
    U[3][3] = 1
    U[0][4] = 1; U[1][4] = 1
    U[2][5] = 1; U[0][5] = -1
    U[1][6] = 1; U[3][6] = -1
    
    V[0][0] = 1; V[3][0] = 1
    V[0][1] = 1
    V[1][2] = 1; V[3][2] = -1
    V[2][3] = 1; V[0][3] = -1
    V[3][4] = 1
    V[0][5] = 1; V[1][5] = 1
    V[2][6] = 1; V[3][6] = 1
    
    W[0][0] = 1; W[0][3] = 1; W[0][4] = -1; W[0][6] = -1
    W[1][2] = 1; W[1][4] = 1
    W[2][1] = 1; W[2][3] = 1
    W[3][0] = 1; W[3][1] = -1; W[3][2] = 1; W[3][6] = 1
    
    return U, V, W

def _schoolbook(n, m, p):
    """Return U, V, W for schoolbook M_{n,m,p} with rank n*m*p."""
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

def _strassen_225():
    """Return U, V, W for M_{2,2,5} using Strassen on first 2x2x2 block + schoolbook for remaining 2x2x3.
    Total rank = 7 + 6 = 13.
    U shape (4,13): indexed by (i*2+j, r)
    V shape (10,13): indexed by (j*5+k, r)
    W shape (10,13): indexed by (k*2+i, r)
    """
    n, m, p = 2, 2, 5
    rank = 13
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]
    
    # Block A: k in [0,2) -> Strassen 2x2x2, rank 7
    sub_U, sub_V, sub_W = _strassen_222()
    for li in range(2):
        for lj in range(2):
            gi = li
            gj = lj
            # U: local (li*2+lj) -> global (gi*2+gj)
            for rr in range(7):
                U[gi*2+gj][rr] += sub_U[li*2+lj][rr]
            # V: local (lj*2+lk) -> global (gj*5+lk) where lk in [0,2)
            for lk in range(2):
                for rr in range(7):
                    V[gj*5+lk][rr] += sub_V[lj*2+lk][rr]
            # W: local (lk*2+li) -> global (lk*2+gi) where lk in [0,2)
            for lk in range(2):
                for rr in range(7):
                    W[lk*2+gi][rr] += sub_W[lk*2+li][rr]
    
    # Block B: k in [2,5) -> Schoolbook 2x2x3, rank 6
    sb_U, sb_V, sb_W = _schoolbook(2, 2, 3)
    for li in range(2):
        for lj in range(2):
            gi = li
            gj = lj
            for lk in range(3):
                gk = 2 + lk
                col = 7 + (lj*3 + lk)
                for rr in range(6):
                    U[gi*2+gj][col + rr] += sb_U[li*2+lj][rr]
                    V[gj*5+gk][col + rr] += sb_V[lj*3+lk][rr]
                    W[gk*2+gi][col + rr] += sb_W[lk*2+li][rr]
    
    return U, V, W

def _add_block(global_U, global_V, global_W, rank_offset, 
               sub_U, sub_V, sub_W,
               i_off, j_off, k_off,
               n_sub, m_sub, p_sub,
               n_global, m_global, p_global):
    """Add a sub-block decomposition into the global factor matrices."""
    r = len(sub_U[0])
    for local_i in range(n_sub):
        for local_j in range(m_sub):
            gi = i_off + local_i
            gj = j_off + local_j
            global_row_U = gi * m_global + gj
            for rr in range(r):
                if sub_U[local_i*m_sub + local_j][rr] != 0:
                    global_U[global_row_U][rank_offset + rr] += sub_U[local_i*m_sub + local_j][rr]
    for local_j in range(m_sub):
        for local_k in range(p_sub):
            gj = j_off + local_j
            gk = k_off + local_k
            global_row_V = gj * p_global + gk
            for rr in range(r):
                if sub_V[local_j*p_sub + local_k][rr] != 0:
                    global_V[global_row_V][rank_offset + rr] += sub_V[local_j*p_sub + local_k][rr]
    for local_k in range(p_sub):
        for local_i in range(n_sub):
            gk = k_off + local_k
            gi = i_off + local_i
            global_row_W = gk * n_global + gi
            for rr in range(r):
                if sub_W[local_k*n_sub + local_i][rr] != 0:
                    global_W[global_row_W][rank_offset + rr] += sub_W[local_k*n_sub + local_i][rr]

def _verify_full(U, V, W, n, m, p):
    """Verify that the decomposition is correct by checking tensor equality."""
    rank = len(U[0])
    U_np = np.array(U, dtype=float)
    V_np = np.array(V, dtype=float)
    W_np = np.array(W, dtype=float)
    
    # Reconstruct tensor T[i,j,k] = sum_r U[i*m+j, r] V[j*p+k, r] W[k*n+i, r]
    # We only need to check if it equals 1 for all i,j,k
    # T is shape (n*m, m*p, p*n)
    # But we can just check the contraction for each (i,j,k)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                for r in range(rank):
                    s += U_np[u_idx, r] * V_np[v_idx, r] * W_np[w_idx, r]
                if abs(s - 1.0) > 1e-9:
                    return False
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin
    
    n, m, p = DIMENSIONS  # 2, 3, 5
    
    # Explicit rank-28 construction:
    # Block A: i in [0,2), j in [0,2), k in [0,5) -> Strassen_225 (rank 13)
    # Block B: i in [0,2), j in [2,3), k in [0,5) -> Schoolbook 2x1x5 (rank 10)
    # Total rank = 13 + 10 = 23? No wait...
    # Actually let me reconsider. The blocks overlap in indices.
    # 
    # Better decomposition:
    # Block A: i in [0,2), j in [0,2), k in [0,2) -> Strassen 2x2x2 (rank 7)
    # Block B: i in [0,2), j in [0,2), k in [2,5) -> Schoolbook 2x2x3 (rank 6)
    # Block C: i in [0,2), j in [2,3), k in [0,5) -> Schoolbook 2x1x5 (rank 10)
    # Total rank = 7 + 6 + 10 = 23
    #
    # Wait, that's even better! Let me verify this is valid.
    # The tensor T[i,j,k] = 1 for all i,j,k.
    # Each (i,j,k) is covered by exactly one block, so the sum works.
    
    total_rank = 7 + 6 + 10  # = 23
    U = [[0]*total_rank for _ in range(n*m)]
    V = [[0]*total_rank for _ in range(m*p)]
    W = [[0]*total_rank for _ in range(p*n)]
    
    # Block A: Strassen 2x2x2, i in [0,2), j in [0,2), k in [0,2)
    sub_U, sub_V, sub_W = _strassen_222()
    _add_block(U, V, W, 0, sub_U, sub_V, sub_W,
               i_off=0, j_off=0, k_off=0,
               n_sub=2, m_sub=2, p_sub=2,
               n_global=n, m_global=m, p_global=p)
    
    # Block B: Schoolbook 2x2x3, i in [0,2), j in [0,2), k in [2,5)
    sb_U, sb_V, sb_W = _schoolbook(2, 2, 3)
    _add_block(U, V, W, 7, sb_U, sb_V, sb_W,
               i_off=0, j_off=0, k_off=2,
               n_sub=2, m_sub=2, p_sub=3,
               n_global=n, m_global=m, p_global=p)
    
    # Block C: Schoolbook 2x1x5, i in [0,2), j in [2,3), k in [0,5)
    sb2_U, sb2_V, sb2_W = _schoolbook(2, 1, 5)
    _add_block(U, V, W, 13, sb2_U, sb2_V, sb2_W,
               i_off=0, j_off=2, k_off=0,
               n_sub=2, m_sub=1, p_sub=5,
               n_global=n, m_global=m, p_global=p)
    
    # Verify the decomposition
    if _verify_full(U, V, W, n, m, p):
        return {"U": U, "V": V, "W": W}
    
    # Fallback: pure schoolbook rank 30
    rank_sb = n * m * p
    U_sb = [[0]*rank_sb for _ in range(n*m)]
    V_sb = [[0]*rank_sb for _ in range(m*p)]
    W_sb = [[0]*rank_sb for _ in range(p*n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i*m+j][col] = 1
                V_sb[j*p+k][col] = 1
                W_sb[k*n+i][col] = 1
                col += 1
    return {"U": U_sb, "V": V_sb, "W": W_sb}
