import time
import numpy as np

DIMENSIONS = [3, 4, 6]
_START_TIME = time.time()
_DEADLINE = 130

def construct():
    n, m, p = DIMENSIONS
    
    def check_deadline():
        return (time.time() - _START_TIME) < _DEADLINE
    
    # Build initial rank-66 decomposition
    rank = 66
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    current_rank = [0]
    
    def add_rank1(u_indices, u_vals, v_indices, v_vals, w_indices, w_vals):
        if current_rank[0] >= rank or not check_deadline():
            return
        r = current_rank[0]
        for idx, val in zip(u_indices, u_vals):
            U[idx, r] += val
        for idx, val in zip(v_indices, v_vals):
            V[idx, r] += val
        for idx, val in zip(w_indices, w_vals):
            W[idx, r] += val
        current_rank[0] += 1
    
    def add_1_4_2(i_off, j_off, k_off):
        for j in range(4):
            for k in range(2):
                u_idx = i_off * m + (j_off + j)
                v_idx = (j_off + j) * p + (k_off + k)
                w_idx = (k_off + k) * n + i_off
                add_rank1([u_idx], [1], [v_idx], [1], [w_idx], [1])
    
    def add_2_2_2(i_off, j_off, k_off):
        def u(i, j): return (i_off + i) * m + (j_off + j)
        def v(j, k): return (j_off + j) * p + (k_off + k)
        def w(i, k): return (k_off + k) * n + (i_off + i)
        
        add_rank1([u(0,0), u(1,1)], [1, 1], [v(0,0), v(1,1)], [1, 1], [w(0,0), w(1,1)], [1, 1])
        add_rank1([u(1,0), u(1,1)], [1, 1], [v(0,0)], [1], [w(1,0), w(1,1)], [1, -1])
        add_rank1([u(0,0)], [1], [v(0,1), v(1,1)], [1, -1], [w(0,1), w(1,1)], [1, 1])
        add_rank1([u(1,1)], [1], [v(1,0), v(0,0)], [1, -1], [w(0,0), w(1,0)], [1, 1])
        add_rank1([u(0,0), u(0,1)], [1, 1], [v(1,1)], [1], [w(0,0), w(0,1)], [-1, 1])
        add_rank1([u(1,0), u(0,0)], [1, -1], [v(0,0), v(0,1)], [1, 1], [w(1,1)], [1])
        add_rank1([u(0,1), u(1,1)], [1, -1], [v(1,0), v(1,1)], [1, 1], [w(0,0)], [1])
    
    def add_2_4_2(i_off, j_off, k_off):
        add_2_2_2(i_off, j_off, k_off)
        add_2_2_2(i_off, j_off + 2, k_off)
    
    for block_k in range(3):
        k_off = block_k * 2
        add_1_4_2(0, 0, k_off)
        add_2_4_2(1, 0, k_off)
    
    actual_rank = current_rank[0]
    
    # Verify initial decomposition
    U3 = U[:, :actual_rank].reshape(n, m, actual_rank)
    V3 = V[:, :actual_rank].reshape(m, p, actual_rank)
    W3 = W[:, :actual_rank].reshape(p, n, actual_rank)
    result = np.einsum('ijc,jkc,kic->ijk', U3, V3, W3)
    target = np.ones((n, m, p))
    
    if not np.allclose(result, target, atol=1e-9):
        # Fallback to schoolbook
        rank_sb = n * m * p
        U_sb = np.zeros((n * m, rank_sb), dtype=float)
        V_sb = np.zeros((m * p, rank_sb), dtype=float)
        W_sb = np.zeros((p * n, rank_sb), dtype=float)
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_sb[i*m+j, col] = 1
                    V_sb[j*p+k, col] = 1
                    W_sb[k*n+i, col] = 1
                    col += 1
        return {"U": U_sb.tolist(), "V": V_sb.tolist(), "W": W_sb.tolist()}
    
    # Try greedy rank reduction: attempt to remove each rank-1 term
    # by checking if the remaining terms can still span the tensor
    # Strategy: for each term r, check if removing it still gives exact decomposition
    # If not, try adjusting coefficients of nearby terms
    
    best_U, best_V, best_W, best_rank = U[:, :actual_rank].copy(), V[:, :actual_rank].copy(), W[:, :actual_rank].copy(), actual_rank
    
    # Try removing each term one at a time
    for r_try in range(actual_rank):
        if not check_deadline():
            break
        
        # Skip this term
        mask = np.ones(actual_rank, dtype=bool)
        mask[r_try] = False
        
        U_test = U[:, mask]
        V_test = V[:, mask]
        W_test = W[:, mask]
        
        U3_t = U_test.reshape(n, m, -1)
        V3_t = V_test.reshape(m, p, -1)
        W3_t = W_test.reshape(p, n, -1)
        result_t = np.einsum('ijc,jkc,kic->ijk', U3_t, V3_t, W3_t)
        
        if np.allclose(result_t, target, atol=1e-9):
            # This term can be removed!
            best_U, best_V, best_W = U_test.copy(), V_test.copy(), W_test.copy()
            best_rank = len(mask[mask])
            # Update U, V, W for next iteration
            U = U_test
            V = V_test
            W = W_test
            actual_rank = best_rank
            # Restart the removal loop with the new smaller set
            r_try = -1  # restart from beginning
            continue
    
    # Final verification
    U3_f = best_U.reshape(n, m, best_rank)
    V3_f = best_V.reshape(m, p, best_rank)
    W3_f = best_W.reshape(p, n, best_rank)
    result_f = np.einsum('ijc,jkc,kic->ijk', U3_f, V3_f, W3_f)
    
    if not np.allclose(result_f, target, atol=1e-9):
        # Fallback to schoolbook
        rank_sb = n * m * p
        U_sb = np.zeros((n * m, rank_sb), dtype=float)
        V_sb = np.zeros((m * p, rank_sb), dtype=float)
        W_sb = np.zeros((p * n, rank_sb), dtype=float)
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_sb[i*m+j, col] = 1
                    V_sb[j*p+k, col] = 1
                    W_sb[k*n+i, col] = 1
                    col += 1
        return {"U": U_sb.tolist(), "V": V_sb.tolist(), "W": W_sb.tolist()}
    
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
