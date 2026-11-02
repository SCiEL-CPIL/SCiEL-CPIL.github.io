"""Program for 3x5x5 matrix multiplication tensor decomposition using Strassen sub-blocks."""
DIMENSIONS = [3, 5, 5]

import time
import numpy as np
import random

# EVOLVE-BLOCK-START
def _strassen_2x2_2x2_2x2():
    """
    Return the 7 rank terms for 2x2x2 matrix multiplication using Strassen.
    Each term is (u_vec, v_vec, w_vec) where:
      u_vec: length 4, indexed by (i,j) for i in {0,1}, j in {0,1}
      v_vec: length 4, indexed by (j,k) for j in {0,1}, k in {0,1}
      w_vec: length 4, indexed by (k,i) for k in {0,1}, i in {0,1}
    """
    terms = []
    # M1 = (a00+a11)*(b00+b11)
    terms.append(([1,0,0,1], [1,0,0,1], [1,0,0,1]))
    # M2 = (a10+a11)*b00
    terms.append(([0,0,1,1], [1,0,0,0], [0,0,1,-1]))
    # M3 = a00*(b01-b11)
    terms.append(([1,0,0,0], [0,1,0,-1], [0,1,0,1]))
    # M4 = a11*(b10-b00)
    terms.append(([0,0,0,1], [-1,0,1,0], [1,0,1,0]))
    # M5 = (a00+a01)*b11
    terms.append(([1,1,0,0], [0,0,0,1], [-1,1,0,0]))
    # M6 = (a10-a00)*(b00+b01)
    terms.append([[-1,0,1,0], [1,1,0,0], [0,0,0,1]])
    # M7 = (a01-a11)*(b10+b11)
    terms.append(([0,1,0,-1], [0,0,1,1], [1,0,0,0]))
    
    return terms

def _build_decomposition(n, m, p):
    """
    Build decomposition for n x m x p matrix multiplication.
    Uses Strassen for 2x2x2 sub-blocks where possible.
    Returns (U, V, W) as numpy arrays.
    """
    max_rank = n * m * p
    U = np.zeros((n * m, max_rank))
    V = np.zeros((m * p, max_rank))
    W = np.zeros((p * n, max_rank))
    rank = 0
    
    strassen_terms = _strassen_2x2_2x2_2x2()
    
    # Partition to maximize 2x2x2 blocks
    i_parts = [[0, 1], [2]]
    j_parts = [[0, 1], [2, 3], [4]]
    k_parts = [[0, 1], [2, 3], [4]]
    
    for i_range in i_parts:
        for j_range in j_parts:
            for k_range in k_parts:
                ni = len(i_range)
                nj = len(j_range)
                nk = len(k_range)
                
                if ni == 2 and nj == 2 and nk == 2:
                    i0, i1 = i_range[0], i_range[1]
                    j0, j1 = j_range[0], j_range[1]
                    k0, k1 = k_range[0], k_range[1]
                    
                    for (u_vec, v_vec, w_vec) in strassen_terms:
                        for li in range(2):
                            for lj in range(2):
                                u_val = u_vec[li * 2 + lj]
                                if u_val != 0:
                                    i_idx = i_range[li]
                                    j_idx = j_range[lj]
                                    U[i_idx * m + j_idx, rank] = u_val
                        
                        for lj in range(2):
                            for lk in range(2):
                                v_val = v_vec[lj * 2 + lk]
                                if v_val != 0:
                                    j_idx = j_range[lj]
                                    k_idx = k_range[lk]
                                    V[j_idx * p + k_idx, rank] = v_val
                        
                        for lk in range(2):
                            for li in range(2):
                                w_val = w_vec[lk * 2 + li]
                                if w_val != 0:
                                    k_idx = k_range[lk]
                                    i_idx = i_range[li]
                                    W[k_idx * n + i_idx, rank] = w_val
                        
                        rank += 1
                else:
                    for i_idx in i_range:
                        for j_idx in j_range:
                            for k_idx in k_range:
                                U[i_idx * m + j_idx, rank] = 1
                                V[j_idx * p + k_idx, rank] = 1
                                W[k_idx * n + i_idx, rank] = 1
                                rank += 1
    
    U = U[:, :rank]
    V = V[:, :rank]
    W = W[:, :rank]
    
    return U, V, W, rank

def _verify_fast(U, V, W, n, m, p):
    """Fast vectorized verification."""
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = np.dot(U[i * m + j], V[j * p + k] * W[k * n + i])
                if abs(s - 1.0) > 1e-9:
                    return False
    return True

def _try_remove_column(U, V, W, n, m, p, col_idx):
    """Try removing a column and verify."""
    rank = U.shape[1]
    if rank <= 1:
        return None, None, None
    
    U_new = np.delete(U, col_idx, axis=1)
    V_new = np.delete(V, col_idx, axis=1)
    W_new = np.delete(W, col_idx, axis=1)
    
    if _verify_fast(U_new, V_new, W_new, n, m, p):
        return U_new, V_new, W_new
    return None, None, None

def _greedy_column_removal(U, V, W, n, m, p, deadline):
    """Greedily remove columns while maintaining validity."""
    start_time = time.time()
    improved = True
    while improved and (time.time() - start_time) < deadline:
        improved = False
        rank = U.shape[1]
        # Shuffle columns to escape local minima
        col_order = list(range(rank))
        random.shuffle(col_order)
        
        for col_idx in col_order:
            if (time.time() - start_time) > deadline:
                break
            U_new, V_new, W_new = _try_remove_column(U, V, W, n, m, p, col_idx)
            if U_new is not None:
                U, V, W = U_new, V_new, W_new
                improved = True
                break
    return U, V, W

def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    
    n, m, p = DIMENSIONS
    
    # Build initial decomposition with Strassen sub-blocks
    U, V, W, rank = _build_decomposition(n, m, p)
    
    # Verify
    if not _verify_fast(U, V, W, n, m, p):
        # Fallback to schoolbook
        rank = n * m * p
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
        if not _verify_fast(U, V, W, n, m, p):
            raise ValueError("Verification failed")
    
    # Greedy column removal with multiple restarts
    best_U, best_V, best_W = U, V, W
    
    # Run greedy removal for a fixed amount of time
    phase1_deadline = min(deadline, start_time + 40.0)
    U, V, W = _greedy_column_removal(U, V, W, n, m, p, phase1_deadline)
    
    if U.shape[1] < best_U.shape[1]:
        best_U, best_V, best_W = U, V, W
    
    # Try another round with different shuffle
    if time.time() < deadline - 20.0:
        phase2_deadline = min(deadline, start_time + 80.0)
        U2, V2, W2 = _greedy_column_removal(best_U, best_V, best_W, n, m, p, phase2_deadline)
        if U2.shape[1] < best_U.shape[1]:
            best_U, best_V, best_W = U2, V2, W2

    # Final round
    if time.time() < deadline - 10.0:
        U3, V3, W3 = _greedy_column_removal(best_U, best_V, best_W, n, m, p, deadline)
        if U3.shape[1] < best_U.shape[1]:
            best_U, best_V, best_W = U3, V3, W3

    U, V, W = best_U, best_V, best_W
    
    # Final verification
    if not _verify_fast(U, V, W, n, m, p):
        U, V, W, rank = _build_decomposition(n, m, p)
    
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()
    
    for mat in [U_list, V_list, W_list]:
        for row in mat:
            for idx in range(len(row)):
                if abs(row[idx]) < 1e-10:
                    row[idx] = 0.0
                elif abs(row[idx] - round(row[idx])) < 1e-10:
                    row[idx] = float(round(row[idx]))
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
