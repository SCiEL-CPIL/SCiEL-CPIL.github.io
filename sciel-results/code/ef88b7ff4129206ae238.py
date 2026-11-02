"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 7]
import numpy as np
import time

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 110
    
    def schoolbook():
        rank = n * m * p
        U = np.zeros((n*m, rank))
        V = np.zeros((m*p, rank))
        W = np.zeros((p*n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j, col] = 1
                    V[j*p+k, col] = 1
                    W[k*n+i, col] = 1
                    col += 1
        return U.tolist(), V.tolist(), W.tolist()
    
    def find_sub_decomposition(nn, mm, pp, target_rank, max_iters=200, seed=0):
        if time.time() > deadline:
            return None
        rng = np.random.default_rng(seed)
        U = rng.standard_normal((nn*mm, target_rank)) * 0.3
        V = rng.standard_normal((mm*pp, target_rank)) * 0.3
        W = rng.standard_normal((pp*nn, target_rank)) * 0.3
        
        T = np.zeros((nn*mm, mm*pp, pp*nn))
        for i in range(nn):
            for j in range(mm):
                for k in range(pp):
                    T[i*mm+j, j*pp+k, k*nn+i] = 1.0
        
        for it in range(max_iters):
            if time.time() > deadline:
                break
            
            U_new = np.zeros_like(U)
            for a in range(nn*mm):
                M = np.einsum('bd,cd->bcd', V, W).reshape(mm*pp, pp*nn, target_rank).reshape(mm*pp*pp*nn, target_rank)
                t = T[a].reshape(-1)
                sol, _, _, _ = np.linalg.lstsq(M, t, rcond=None)
                U_new[a, :] = sol
            
            V_new = np.zeros_like(V)
            for b in range(mm*pp):
                M = np.einsum('ad,cd->acd', U, W).reshape(nn*mm, pp*nn, target_rank).reshape(nn*mm*pp*nn, target_rank)
                t = T[:, b, :].reshape(-1)
                sol, _, _, _ = np.linalg.lstsq(M, t, rcond=None)
                V_new[b, :] = sol
            
            W_new = np.zeros_like(W)
            for c in range(pp*nn):
                M = np.einsum('ad,bd->abd', U, V).reshape(nn*mm, mm*pp, target_rank).reshape(nn*mm*mm*pp, target_rank)
                t = T[:, :, c].reshape(-1)
                sol, _, _, _ = np.linalg.lstsq(M, t, rcond=None)
                W_new[c, :] = sol
            
            U, V, W = U_new, V_new, W_new
            
            if it % 20 == 0:
                D = np.einsum('ad,bd,cd->abc', U, V, W)
                loss = np.sum((D - T)**2)
                if loss < 1e-14:
                    return U, V, W
        
        D = np.einsum('ad,bd,cd->abc', U, V, W)
        loss = np.sum((D - T)**2)
        if loss < 1e-10:
            return U, V, W
        return None
    
    for sub_rank in [14, 15, 16, 18, 20, 24, 28]:
        if time.time() > deadline:
            break
        for seed in range(5):
            if time.time() > deadline:
                break
            result = find_sub_decomposition(2, 2, 7, sub_rank, max_iters=80, seed=seed)
            if result is not None:
                U_sub, V_sub, W_sub = result
                full_rank = 4 * sub_rank
                U_full = np.zeros((16, full_rank))
                V_full = np.zeros((28, full_rank))
                W_full = np.zeros((28, full_rank))
                
                for s in range(4):
                    off = s * sub_rank
                    i_off = 2 * (s // 2)  # 0 for s=0,1; 2 for s=2,3
                    j_off = 2 * (s % 2)   # 0 for s=0,2; 2 for s=1,3
                    
                    for si in range(2):
                        for sj in range(2):
                            sub_u_row = si * 2 + sj
                            full_u_row = (si + i_off) * 4 + (sj + j_off)
                            U_full[full_u_row, off:off+sub_rank] += U_sub[sub_u_row, :]
                    
                    for sj in range(2):
                        for k in range(7):
                            sub_v_row = sj * 7 + k
                            full_v_row = (sj + j_off) * 7 + k
                            V_full[full_v_row, off:off+sub_rank] += V_sub[sub_v_row, :]
                    
                    for k in range(7):
                        for si in range(2):
                            sub_w_row = k * 2 + si
                            full_w_row = k * 4 + (si + i_off)
                            W_full[full_w_row, off:off+sub_rank] += W_sub[sub_w_row, :]
                
                T = np.zeros((16, 28, 28))
                for i in range(4):
                    for j in range(4):
                        for k in range(7):
                            T[i*4+j, j*7+k, k*4+i] = 1.0
                D = np.einsum('ad,bd,cd->abc', U_full, V_full, W_full)
                if np.allclose(D, T, atol=1e-8):
                    return {"U": U_full.tolist(), "V": V_full.tolist(), "W": W_full.tolist()}
    
    U, V, W = schoolbook()
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END

