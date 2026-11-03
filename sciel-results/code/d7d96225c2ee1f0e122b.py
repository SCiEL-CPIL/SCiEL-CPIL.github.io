"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 5]

# EVOLVE-BLOCK-START
import numpy as np
import time

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140
    
    # Build the multiplication tensor T
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c_idx = k * n + i
                T[a, b, c_idx] = 1.0
    
    # Schoolbook solution as fallback
    rank_sb = n * m * p
    U_sb = np.zeros((n*m, rank_sb))
    V_sb = np.zeros((m*p, rank_sb))
    W_sb = np.zeros((p*n, rank_sb))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c_idx = k * n + i
                U_sb[a, col] = 1.0
                V_sb[b, col] = 1.0
                W_sb[c_idx, col] = 1.0
                col += 1
    
    best_U, best_V, best_W = U_sb, V_sb, W_sb
    best_rank = rank_sb
    
    lower_bound = max(n * p, m * p, n * m)
    
    for r in range(lower_bound, rank_sb):
        if time.time() > deadline - 25:
            break
        
        found = False
        for trial in range(20):
            if time.time() > deadline - 25:
                break
            
            np.random.seed(42 + trial)
            U = np.random.randn(n*m, r)
            V = np.random.randn(m*p, r)
            W = np.random.randn(p*n, r)
            
            for it in range(200):
                if time.time() > deadline - 25:
                    break
                
                # Solve for U
                W_mat = np.zeros((m*p*p*n, r))
                idx = 0
                for b in range(m*p):
                    for c_idx in range(p*n):
                        W_mat[idx, :] = V[b, :] * W[c_idx, :]
                        idx += 1
                T_mat = T.reshape(n*m, m*p*p*n)
                U_new, _, _, _ = np.linalg.lstsq(W_mat.T, T_mat.T, rcond=None)
                U = U_new.T
                
                # Solve for V
                V_mat = np.zeros((n*m*p*n, r))
                idx = 0
                for c_idx in range(p*n):
                    for a in range(n*m):
                        V_mat[idx, :] = U[a, :] * W[c_idx, :]
                        idx += 1
                T_mat_v = T.transpose(1, 0, 2).reshape(m*p, n*m*p*n)
                V_new, _, _, _ = np.linalg.lstsq(V_mat.T, T_mat_v.T, rcond=None)
                V = V_new.T
                
                # Solve for W
                W_mat2 = np.zeros((n*m*m*p, r))
                idx = 0
                for b in range(m*p):
                    for a in range(n*m):
                        W_mat2[idx, :] = U[a, :] * V[b, :]
                        idx += 1
                T_mat_w = T.transpose(2, 0, 1).reshape(p*n, n*m*m*p)
                W_new, _, _, _ = np.linalg.lstsq(W_mat2.T, T_mat_w.T, rcond=None)
                W = W_new.T
                
                if it % 50 == 49:
                    T_recon = np.zeros((n*m, m*p, p*n))
                    for c in range(r):
                        T_recon += U[:, c] * V[:, c] * W[:, c]
                    if np.allclose(T_recon, T, atol=1e-5):
                        break
            
            T_recon = np.zeros((n*m, m*p, p*n))
            for c in range(r):
                T_recon += U[:, c] * V[:, c] * W[:, c]
            
            if np.allclose(T_recon, T, atol=1e-4):
                if r < best_rank:
                    best_rank = r
                    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
                found = True
                break
        
        if found:
            break
    
    # Round to half-integers if possible
    def round_half(mat):
        m2 = mat * 2
        mr = np.round(m2) / 2
        if np.allclose(mat, mr, atol=1e-3):
            return mr
        mi = np.round(mat)
        if np.allclose(mat, mi, atol=1e-3):
            return mi
        return mat
    
    Uo = round_half(best_U)
    Vo = round_half(best_V)
    Wo = round_half(best_W)
    
    Tr = np.zeros((n*m, m*p, p*n))
    for c in range(best_rank):
        Tr += Uo[:, c] * Vo[:, c] * Wo[:, c]
    
    if not np.allclose(Tr, T, atol=1e-3):
        Uo, Vo, Wo = best_U, best_V, best_W
    
    return {"U": Uo.tolist(), "V": Vo.tolist(), "W": Wo.tolist()}
# EVOLVE-BLOCK-END
