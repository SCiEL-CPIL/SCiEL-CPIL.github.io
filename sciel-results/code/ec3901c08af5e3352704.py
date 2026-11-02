import numpy as np
import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin

    n, m, p = 2, 3, 9
    
    # Fallback solution (Rank 50)
    # Reuse the logic from the current program to ensure we have a valid baseline
    def get_fallback():
        rank = 50
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]

        col = 0
        u_a, u_b, u_c, u_d = 0, 1, 3, 4

        for b in range(4):
            k0 = 2 * b
            k1 = 2 * b + 1
            v_e = k0
            v_f = k1
            v_g = p + k0
            v_h = p + k1

            U[u_a][col] += 1
            U[u_d][col] += 1
            V[v_e][col] += 1
            V[v_h][col] += 1

            U[u_c][col+1] += 1
            U[u_d][col+1] += 1
            V[v_e][col+1] += 1

            U[u_a][col+2] += 1
            V[v_f][col+2] += 1
            V[v_h][col+2] -= 1

            U[u_d][col+3] += 1
            V[v_g][col+3] += 1
            V[v_e][col+3] -= 1

            U[u_a][col+4] += 1
            U[u_b][col+4] += 1
            V[v_h][col+4] += 1

            U[u_c][col+5] += 1
            U[u_a][col+5] -= 1
            V[v_e][col+5] += 1
            V[v_f][col+5] += 1

            U[u_b][col+6] += 1
            U[u_d][col+6] -= 1
            V[v_g][col+6] += 1
            V[v_h][col+6] += 1

            w_00 = k0 * n + 0
            w_01 = k1 * n + 0
            w_10 = k0 * n + 1
            w_11 = k1 * n + 1

            W[w_00][col] += 1
            W[w_00][col+3] += 1
            W[w_00][col+4] -= 1
            W[w_00][col+6] += 1

            W[w_01][col+2] += 1
            W[w_01][col+4] += 1

            W[w_10][col+1] += 1
            W[w_10][col+3] += 1

            W[w_11][col] += 1
            W[w_11][col+1] -= 1
            W[w_11][col+2] += 1
            W[w_11][col+5] += 1

            col += 7

        k8 = 8
        v_e = k8
        v_g = p + k8

        U[u_a][col] += 1
        V[v_e][col] += 1
        U[u_b][col+1] += 1
        V[v_g][col+1] += 1
        U[u_c][col+2] += 1
        V[v_e][col+2] += 1
        U[u_d][col+3] += 1
        V[v_g][col+3] += 1

        w_08 = k8 * n + 0
        w_18 = k8 * n + 1

        W[w_08][col] += 1
        W[w_08][col+1] += 1
        W[w_18][col+2] += 1
        W[w_18][col+3] += 1

        col += 4

        u_ar0, u_ar1 = 2, 5
        for k in range(9):
            v_b = 2 * p + k
            U[u_ar0][col] += 1
            V[v_b][col] += 1
            W[k * n + 0][col] += 1
            U[u_ar1][col+1] += 1
            V[v_b][col+1] += 1
            W[k * n + 1][col+1] += 1
            col += 2

        return {"U": U, "V": V, "W": W}

    # Try to find a better decomposition for M_{2,3,3}
    # If found, use it to build M_{2,3,9}
    
    # Tensor for M_{2,3,3}
    # n=2, m=3, p=3
    n_sub, m_sub, p_sub = 2, 3, 3
    T_sub = np.zeros((n_sub*m_sub, m_sub*p_sub, p_sub*n_sub))
    for i in range(n_sub*m_sub):
        ai = i // m_sub
        am = i % m_sub
        for j in range(m_sub*p_sub):
            bj = j // p_sub
            bm = j % p_sub
            if am == bj:
                k = bm * n_sub + ai
                T_sub[i, j, k] = 1.0

    def solve_sub(rank):
        # Initialize
        U = np.random.randn(n_sub*m_sub, rank)
        V = np.random.randn(m_sub*p_sub, rank)
        W = np.random.randn(p_sub*n_sub, rank)
        
        # Alternating Least Squares
        for it in range(100):
            # Fix U, W, solve V
            # T[i,j,k] = sum_r U[i,r] V[j,r] W[k,r]
            # For fixed U, W, the equation for V[j, :] is:
            # sum_r V[j,r] (U[:,r] W[:,r]) = T[:,j,:]
            # This is a linear system for V[j, :]
            # Let M[j, r, i, k] = U[i,r] W[k,r]
            # We want V[j, :] such that sum_r V[j,r] M[j, r, i, k] = T[i, j, k]
            
            # Reshape T to (i, k, j) -> (i*k, j)
            T_ikj = T_sub.transpose(0, 2, 1).reshape(-1, m_sub*p_sub)
            
            # Build the matrix for the linear system
            # A[j, idx, r] where idx = i*p_sub*n_sub + k
            A = np.zeros((m_sub*p_sub, n_sub*m_sub*p_sub*n_sub, rank))
            for j in range(m_sub*p_sub):
                for r in range(rank):
                    # Vector of size n_sub*m_sub*p_sub*n_sub
                    vec = np.outer(U[:, r], W[:, r]).reshape(-1)
                    A[j, :, r] = vec
            
            # Solve V[j, :] = argmin || A[j] V[j] - T_ikj[:, j] ||
            # A[j] is (n_sub*m_sub*p_sub*n_sub, rank)
            # We need to solve A[j] x = b for x = V[j, :]
            # Using least squares
            V_new = np.zeros_like(V)
            for j in range(m_sub*p_sub):
                try:
                    V_new[j, :] = np.linalg.lstsq(A[j], T_ikj[:, j], rcond=None)[0]
                except:
                    V_new[j, :] = V[j, :]
            V = V_new
            
            # Similarly for U and W
            # Fix V, W, solve U
            T_jki = T_sub.transpose(1, 2, 0).reshape(-1, n_sub*m_sub)
            A_u = np.zeros((n_sub*m_sub, m_sub*p_sub*p_sub*n_sub, rank))
            for i in range(n_sub*m_sub):
                for r in range(rank):
                    vec = np.outer(V[:, r], W[:, r]).reshape(-1)
                    A_u[i, :, r] = vec
            U_new = np.zeros_like(U)
            for i in range(n_sub*m_sub):
                try:
                    U_new[i, :] = np.linalg.lstsq(A_u[i], T_jki[:, i], rcond=None)[0]
                except:
                    U_new[i, :] = U[i, :]
            U = U_new
            
            # Fix U, V, solve W
            T_kij = T_sub.transpose(2, 0, 1).reshape(-1, p_sub*n_sub)
            A_w = np.zeros((p_sub*n_sub, n_sub*m_sub*m_sub*p_sub, rank))
            for k in range(p_sub*n_sub):
                for r in range(rank):
                    vec = np.outer(U[:, r], V[:, r]).reshape(-1)
                    A_w[k, :, r] = vec
            W_new = np.zeros_like(W)
            for k in range(p_sub*n_sub):
                try:
                    W_new[k, :] = np.linalg.lstsq(A_w[k], T_kij[:, k], rcond=None)[0]
                except:
                    W_new[k, :] = W[k, :]
            W = W_new
            
            # Check error
            T_computed = np.einsum('ir,jr,kr->ijk', U, V, W)
            err = np.sum((T_computed - T_sub)**2)
            if err < 1e-8:
                return U, V, W
            if time.time() > deadline - 10:
                return None, None, None
                
        return None, None, None

    # Try ranks 11, 12, 13
    best_sub = None
    for r in [11, 12, 13, 14]:
        if time.time() > deadline - 20:
            break
        for attempt in range(5):
            if time.time() > deadline - 20:
                break
            random.seed(attempt * 100 + r)
            U_s, V_s, W_s = solve_sub(r)
            if U_s is not None:
                best_sub = (r, U_s, V_s, W_s)
                break
        if best_sub:
            break

    if best_sub:
        r, U_s, V_s, W_s = best_sub
        # Build full decomposition
        # M_{2,3,9} is 3 blocks of M_{2,3,3}
        # The columns of B are split into 3 groups of 3
        # We just repeat the U, V, W blocks
        # But we need to be careful with the indices
        
        total_rank = 3 * r
        U_full = np.zeros((n*m, total_rank))
        V_full = np.zeros((m*p, total_rank))
        W_full = np.zeros((p*n, total_rank))
        
        # U is the same for all blocks (it depends on A)
        # V and W need to be shifted for each block
        # Block 0: cols 0-2 of B
        # Block 1: cols 3-5 of B
        # Block 2: cols 6-8 of B
        
        for b in range(3):
            offset = b * r
            # U
            U_full[:, offset:offset+r] = U_s
            # V
            # V_s has shape (9, r) corresponding to B (3x3)
            # We need to map B indices for block b
            # B index in sub-problem: j_sub = k_sub * 3 + col_sub
            # B index in full: j_full = k_full * 9 + col_full
            # For block b, col_full = b*3 + col_sub
            # k_full = k_sub
            # So j_full = k_sub * 9 + (b*3 + col_sub)
            # We need to place V_s[j_sub, r] into V_full[j_full, offset+r]
            
            for k_sub in range(3):
                for col_sub in range(3):
                    j_sub = k_sub * 3 + col_sub
                    j_full = k_sub * 9 + (b*3 + col_sub)
                    V_full[j_full, offset:offset+r] = V_s[j_sub, :]
            
            # W
            # W_s has shape (6, r) corresponding to C (2x3)
            # C index in sub: k_sub = col_sub * 2 + row_sub
            # C index in full: k_full = col_full * 2 + row_sub
            # For block b, col_full = b*3 + col_sub
            # So k_full = (b*3 + col_sub) * 2 + row_sub
            
            for col_sub in range(3):
                for row_sub in range(2):
                    k_sub = col_sub * 2 + row_sub
                    k_full = (b*3 + col_sub) * 2 + row_sub
                    W_full[k_full, offset:offset+r] = W_s[k_sub, :]
        
        # Verify full
        T_full = np.zeros((n*m, m*p, p*n))
        for i in range(n*m):
            ai = i // m
            am = i % m
            for j in range(m*p):
                bj = j // p
                bm = j % p
                if am == bj:
                    k = bm * n + ai
                    T_full[i, j, k] = 1.0
        
        T_comp = np.einsum('ir,jr,kr->ijk', U_full, V_full, W_full)
        if np.sum((T_comp - T_full)**2) < 1e-8:
            return {
                "U": U_full.tolist(),
                "V": V_full.tolist(),
                "W": W_full.tolist()
            }
            
    return get_fallback()
