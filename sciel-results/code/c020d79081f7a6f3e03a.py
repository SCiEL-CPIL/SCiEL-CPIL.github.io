"""
Rank-minimal exact tensor decomposition for matrix multiplication 4x4 * 4x5.
Uses Strassen block decomposition (rank 64 baseline) + ALS refinement.
"""
import time
import random
import numpy as np

DIMENSIONS = [4, 4, 5]
n, m, p = DIMENSIONS


def _verify_fast(U, V, W):
    rank = len(U[0])
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for r in range(rank):
                    s += U[i * m + j][r] * V[j * p + k][r] * W[k * n + i][r]
                if abs(s - 1.0) > 1e-9:
                    return False
    return True


def _round_matrix_to_half_int(M):
    return np.round(M * 2.0) / 2.0


def _compute_error(U, V, W):
    err = 0.0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = np.dot(U[i * m + j, :], V[j * p + k, :] * W[k * n + i, :])
                err += (val - 1.0) ** 2
    return err


def _als_decompose(U0, V0, W0, max_iters=300, tol=1e-12):
    U = U0.astype(np.float64).copy()
    V = V0.astype(np.float64).copy()
    W = W0.astype(np.float64).copy()
    rank = U.shape[1]
    
    for it in range(max_iters):
        for i in range(n):
            for j in range(m):
                M = np.zeros((p, rank))
                for k in range(p):
                    M[k, :] = V[j * p + k, :] * W[k * n + i, :]
                rhs = np.ones(p)
                try:
                    U[i * m + j, :] = np.linalg.lstsq(M, rhs, rcond=None)[0]
                except:
                    pass
        
        for j in range(m):
            for k in range(p):
                N = np.zeros((n, rank))
                for i in range(n):
                    N[i, :] = U[i * m + j, :] * W[k * n + i, :]
                rhs = np.ones(n)
                try:
                    V[j * p + k, :] = np.linalg.lstsq(N, rhs, rcond=None)[0]
                except:
                    pass
        
        for k in range(p):
            for i in range(n):
                P = np.zeros((m, rank))
                for j in range(m):
                    P[j, :] = U[i * m + j, :] * V[j * p + k, :]
                rhs = np.ones(m)
                try:
                    W[k * n + i, :] = np.linalg.lstsq(P, rhs, rcond=None)[0]
                except:
                    pass
        
        if it % 20 == 0:
            err = _compute_error(U, V, W)
            if err < tol:
                break
    
    return U, V, W


def _strassen_2x2x2_block(i_off, j_off, k_off, U, V, W, col_start):
    """
    Add Strassen 2x2x2 terms for a specific block.
    i_off: offset in i dimension (0 or 2)
    j_off: offset in j dimension (0 or 2)  
    k_off: offset in k dimension (0, 2, or 4)
    k_width: 2 for k_off in {0,2}, 1 for k_off=4
    Returns number of columns used.
    """
    # Strassen 7 terms for 2x2*2x2
    # A sub-block: A[i_off+i, j_off+j] for i,j in {0,1}
    # B sub-block: B[j_off+j, k_off+k] for j in {0,1}, k in {0,1}
    # C sub-block: C[i_off+i, k_off+k] for i in {0,1}, k in {0,1}
    #
    # U index: (i_off+i)*m + (j_off+j)
    # V index: (j_off+j)*p + (k_off+k)
    # W index: (k_off+k)*n + (i_off+i)  (trace convention)
    
    # Strassen terms (u, v, w) for local 2x2 indices:
    # Local A: [0,0],[0,1],[1,0],[1,1] -> indices 0,1,2,3
    # Local B: [0,0],[0,1],[1,0],[1,1] -> indices 0,1,2,3
    # Local C (trace): C[i,k] -> index k*2+i
    #   C[0,0]->0, C[1,0]->1, C[0,1]->2, C[1,1]->3
    
    strassen = [
        # (u_vec, v_vec, w_vec)
        ([1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1]),   # M1
        ([0, 0, 1, 1], [1, 0, 0, 0], [0, 1, 0, -1]),   # M2
        ([1, 0, 0, 0], [0, 0, 1, -1], [0, 0, 1, 1]),   # M3
        ([0, 0, 0, 1], [0, 1, 0, -1], [1, 1, 0, 0]),   # M4
        ([1, 1, 0, 0], [0, 0, 0, 1], [-1, 0, 1, 0]),   # M5
        ([-1, 0, 1, 0], [1, 0, 1, 0], [0, 0, 0, 1]),   # M6
        ([0, 1, 0, -1], [0, 1, 0, 1], [1, 0, 0, 0]),   # M7
    ]
    
    col = col_start
    for u_loc, v_loc, w_loc in strassen:
        for li in range(2):
            for lj in range(2):
                if u_loc[li * 2 + lj] != 0:
                    ui = i_off + li
                    uj = j_off + lj
                    U[ui * m + uj, col] = u_loc[li * 2 + lj]
        
        for lj in range(2):
            for lk in range(2):
                if v_loc[lj * 2 + lk] != 0:
                    vj = j_off + lj
                    vk = k_off + lk
                    V[vj * p + vk, col] = v_loc[lj * 2 + lk]
        
        for lk in range(2):
            for li in range(2):
                if w_loc[lk * 2 + li] != 0:
                    wk = k_off + lk
                    wi = i_off + li
                    W[wk * n + wi, col] = w_loc[lk * 2 + li]
        
        col += 1
    
    return col - col_start


def _strassen_block_construct():
    """
    Build a rank-64 Strassen block decomposition for 4x4 * 4x5.
    
    Blocks:
    - i-groups: {0,1}, {2,3}
    - j-groups: {0,1}, {2,3}
    - k-groups: {0,1}, {2,3}, {4}
    
    For each (i-group, j-group) pair, we have a 2x2 * 2x5 sub-problem.
    Split k into {0,1}, {2,3}, {4}:
    - k in {0,1}: 7 Strassen terms
    - k in {2,3}: 7 Strassen terms
    - k in {4}: 2 schoolbook terms
    Total per (i-group, j-group): 16 terms
    4 (i-group, j-group) pairs: 64 terms total
    """
    rank = 64
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    col = 0
    for i_off in [0, 2]:
        for j_off in [0, 2]:
            # k in {0,1}: Strassen 2x2x2
            col += _strassen_2x2x2_block(i_off, j_off, 0, U, V, W, col)
            # k in {2,3}: Strassen 2x2x2
            col += _strassen_2x2x2_block(i_off, j_off, 2, U, V, W, col)
            # k in {4}: schoolbook 2x2x1 (2 terms)
            for li in range(2):
                for lj in range(2):
                    ui = i_off + li
                    uj = j_off + lj
                    vj = j_off + lj
                    vk = 4
                    wi = i_off + li
                    wk = 4
                    U[ui * m + uj, col] = 1.0
                    V[vj * p + vk, col] = 1.0
                    W[wk * n + wi, col] = 1.0
                    col += 1
    
    return U, V, W, col


def _schoolbook():
    rank = n * m * p
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    c = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][c] = 1.0
                V[j * p + k][c] = 1.0
                W[k * n + i][c] = 1.0
                c += 1
    return {"U": U, "V": V, "W": W}


def construct():
    start_time = time.time()
    deadline = start_time + 130.0
    
    random.seed(42)
    np.random.seed(42)
    
    # Build Strassen block decomposition (rank 64)
    U64, V64, W64, actual_rank = _strassen_block_construct()
    
    # Verify
    U64_list = [[float(x) for x in row] for row in U64]
    V64_list = [[float(x) for x in row] for row in V64]
    W64_list = [[float(x) for x in row] for row in W64]
    
    if _verify_fast(U64_list, V64_list, W64_list):
        best_result = {"U": U64_list, "V": V64_list, "W": W64_list}
        best_rank = actual_rank
    else:
        # Fallback to schoolbook
        sb = _schoolbook()
        best_result = sb
        best_rank = 80
        U64, V64, W64 = np.array(sb["U"]), np.array(sb["V"]), np.array(sb["W"])
        actual_rank = 80
    
    # Try to reduce rank further using ALS
    for target in [60, 58, 56, 54, 52, 50, 49, 48]:
        if time.time() > deadline - 15:
            break
        if target >= best_rank:
            continue
        
        found = False
        for trial in range(2):
            if time.time() > deadline - 10:
                break
            
            # Initialize from current best by selecting 'target' columns
            cols = random.sample(range(best_rank), target)
            U0 = U64[:, cols].copy()
            V0 = V64[:, cols].copy()
            W0 = W64[:, cols].copy()
            
            # Perturb
            U0 += 0.02 * np.random.randn(*U0.shape)
            V0 += 0.02 * np.random.randn(*V0.shape)
            W0 += 0.02 * np.random.randn(*W0.shape)
            
            U_s, V_s, W_s = _als_decompose(U0, V0, W0, max_iters=150)
            err = _compute_error(U_s, V_s, W_s)
            
            if err < 1e-8:
                U_r = _round_matrix_to_half_int(U_s)
                V_r = _round_matrix_to_half_int(V_s)
                W_r = _round_matrix_to_half_int(W_s)
                err_r = _compute_error(U_r, V_r, W_r)
                
                if err_r < 1e-8:
                    U_l = [[float(x) for x in row] for row in U_r]
                    V_l = [[float(x) for x in row] for row in V_r]
                    W_l = [[float(x) for x in row] for row in W_r]
                    if _verify_fast(U_l, V_l, W_l):
                        best_result = {"U": U_l, "V": V_l, "W": W_l}
                        best_rank = target
                        U64, V64, W64 = U_r, V_r, W_r
                        found = True
                        break
                else:
                    U_l = [[float(x) for x in row] for row in U_s]
                    V_l = [[float(x) for x in row] for row in V_s]
                    W_l = [[float(x) for x in row] for row in W_s]
                    if _verify_fast(U_l, V_l, W_l):
                        best_result = {"U": U_l, "V": V_l, "W": W_l}
                        best_rank = target
                        U64, V64, W64 = U_s, V_s, W_s
                        found = True
                        break
        
        if found:
            break
    
    return best_result
