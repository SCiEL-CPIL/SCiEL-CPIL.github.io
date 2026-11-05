import time
import numpy as np

_START_TIME = time.time()
_DEADLINE = 140

def _check_deadline():
    return (time.time() - _START_TIME) < _DEADLINE

def construct():
    n, m, p = 3, 4, 6
    
    # Build the target tensor T[a,b,c]
    # a = i*m + j (i in [0,n), j in [0,m))
    # b = j*p + k (j in [0,m), k in [0,p))
    # c = k*n + i (k in [0,p), i in [0,n))
    T = np.zeros((n*m, m*p, p*n), dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1.0

    # Rank-66 decomposition using Strassen-style blocks
    rank_init = 66
    U = np.zeros((n * m, rank_init), dtype=float)
    V = np.zeros((m * p, rank_init), dtype=float)
    W = np.zeros((p * n, rank_init), dtype=float)
    current_rank = 0

    def add_rank1(u_indices, u_vals, v_indices, v_vals, w_indices, w_vals):
        nonlocal current_rank
        if current_rank >= rank_init or not _check_deadline():
            return
        for idx, val in zip(u_indices, u_vals):
            U[idx, current_rank] += val
        for idx, val in zip(v_indices, v_vals):
            V[idx, current_rank] += val
        for idx, val in zip(w_indices, w_vals):
            W[idx, current_rank] += val
        current_rank += 1

    # Strassen 2x2x2 block
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

    # 1x4x2 block (schoolbook for 1x4 * 4x2)
    def add_1_4_2(i_off, j_off, k_off):
        for j in range(4):
            for k in range(2):
                u_idx = i_off * m + (j_off + j)
                v_idx = (j_off + j) * p + (k_off + k)
                w_idx = (k_off + k) * n + i_off
                add_rank1([u_idx], [1], [v_idx], [1], [w_idx], [1])

    # 2x4x2 = two 2x2x2 blocks
    def add_2_4_2(i_off, j_off, k_off):
        add_2_2_2(i_off, j_off, k_off)
        add_2_2_2(i_off, j_off + 2, k_off)

    # Build the rank-66 decomposition
    for block_k in range(3):
        k_off = block_k * 2
        add_1_4_2(0, 0, k_off)   # 8 terms
        add_2_4_2(1, 0, k_off)   # 14 terms
    # Total: 3 * (8 + 14) = 3 * 22 = 66

    def verify_decomposition(U_mat, V_mat, W_mat, r):
        """Verify that the decomposition is exact."""
        if r == 0:
            return False
        T_recon = np.einsum('ar,br,cr->abc', U_mat[:, :r], V_mat[:, :r], W_mat[:, :r])
        return np.allclose(T, T_recon, atol=1e-8)

    # Try to reduce rank using gradient-based optimization
    # This is a search within the current design family
    
    best_U = U[:, :current_rank].copy()
    best_V = V[:, :current_rank].copy()
    best_W = W[:, :current_rank].copy()
    best_rank = current_rank
    best_err = np.inf

    # Try reducing from rank 66 downward
    for target_r in range(min(60, current_rank - 5), current_rank + 1):
        if not _check_deadline():
            break
        
        # Initialize with first target_r components
        U_cur = U[:, :target_r].copy()
        V_cur = V[:, :target_r].copy()
        W_cur = W[:, :target_r].copy()
        
        # Run gradient descent
        max_iter = 50
        lr = 0.005
        prev_err = np.inf
        
        for it in range(max_iter):
            if not _check_deadline():
                break
            
            T_recon = np.einsum('ar,br,cr->abc', U_cur, V_cur, W_cur)
            err = np.linalg.norm(T - T_recon, 'fro')
            
            if err < best_err:
                best_err = err
                best_U = U_cur.copy()
                best_V = V_cur.copy()
                best_W = W_cur.copy()
                best_rank = target_r
            
            if abs(prev_err - err) < 1e-7:
                break
            prev_err = err
            
            # Gradient descent step
            grad_U = np.einsum('abc,br,cr->ar', T - T_recon, V_cur, W_cur)
            grad_V = np.einsum('abc,ar,cr->br', T - T_recon, U_cur, W_cur)
            grad_W = np.einsum('abc,ar,br->cr', T - T_recon, U_cur, V_cur)
            
            U_cur -= lr * grad_U
            V_cur -= lr * grad_V
            W_cur -= lr * grad_W
            
            # Hard thresholding: zero out small entries
            U_cur[np.abs(U_cur) < 1e-2] = 0
            V_cur[np.abs(V_cur) < 1e-2] = 0
            W_cur[np.abs(W_cur) < 1e-2] = 0
            
            # Snap to integers or half-integers
            U_cur = np.round(U_cur * 2) / 2
            V_cur = np.round(V_cur * 2) / 2
            W_cur = np.round(W_cur * 2) / 2
            
            # Check if we can verify
            if verify_decomposition(U_cur, V_cur, W_cur, target_r):
                return {"U": U_cur.tolist(), "V": V_cur.tolist(), "W": W_cur.tolist()}

    # Final verification
    if not verify_decomposition(best_U, best_V, best_W, best_rank):
        # Fall back to original rank-66 decomposition
        return {"U": U[:, :current_rank].tolist(), "V": V[:, :current_rank].tolist(), "W": W[:, :current_rank].tolist()}
    
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
