"""Matrix multiplication 4,4,5 - rank-minimal exact tensor decomposition."""
import time
import numpy as np

DIMENSIONS = [4, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    deadline = time.time() + 120.0
    n, m, p = DIMENSIONS  # 4, 4, 5
    
    # Build the multiplication tensor T of shape (n*m, m*p, p*n)
    T = np.zeros((n*m, m*p, p*n), dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1.0
    
    def _als_solve_U(T, V, W, n, m, p):
        r = V.shape[1]
        Z = np.zeros((r, m*p * p*n))
        T_mat = np.zeros((n*m, m*p * p*n))
        idx = 0
        for b in range(m*p):
            for c in range(p*n):
                Z[:, idx] = V[b, :] * W[c, :]
                T_mat[:, idx] = T[:, b, c]
                idx += 1
        try:
            U = T_mat @ np.linalg.pinv(Z.T)
        except:
            U = np.zeros((n*m, r))
        return U
    
    def _als_solve_V(T, U, W, n, m, p):
        r = U.shape[1]
        Z = np.zeros((r, n*m * p*n))
        T_mat = np.zeros((m*p, n*m * p*n))
        idx = 0
        for a in range(n*m):
            for c in range(p*n):
                Z[:, idx] = U[a, :] * W[c, :]
                T_mat[:, idx] = T[a, :, c]
                idx += 1
        try:
            V = T_mat @ np.linalg.pinv(Z.T)
        except:
            V = np.zeros((m*p, r))
        return V
    
    def _als_solve_W(T, U, V, n, m, p):
        r = U.shape[1]
        Z = np.zeros((r, n*m * m*p))
        T_mat = np.zeros((p*n, n*m * m*p))
        idx = 0
        for a in range(n*m):
            for b in range(m*p):
                Z[:, idx] = U[a, :] * V[b, :]
                T_mat[:, idx] = T[a, b, :]
                idx += 1
        try:
            W = T_mat @ np.linalg.pinv(Z.T)
        except:
            W = np.zeros((p*n, r))
        return W
    
    def _compute_error(U, V, W, T):
        T_recon = np.einsum('ar,br,cr->abc', U, V, W)
        return np.linalg.norm(T - T_recon)
    
    def _round_to_halfint(mat):
        return np.round(mat * 2) / 2.0
    
    def _verify(U, V, W, T, tol=1e-6):
        T_recon = np.einsum('ar,br,cr->abc', U, V, W)
        return np.max(np.abs(T - T_recon)) < tol
    
    # Schoolbook as fallback
    rank_sb = n * m * p
    U_sb = np.zeros((n*m, rank_sb))
    V_sb = np.zeros((m*p, rank_sb))
    W_sb = np.zeros((p*n, rank_sb))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i*m + j, col] = 1
                V_sb[j*p + k, col] = 1
                W_sb[k*n + i, col] = 1
                col += 1
    
    best_rank = rank_sb
    best_U, best_V, best_W = U_sb.copy(), V_sb.copy(), W_sb.copy()
    
    # Try ranks from low to high
    for target_rank in [7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 18, 20, 24, 28, 32, 40, 48, 56, 64, 72, 80]:
        if time.time() > deadline - 30:
            break
        if target_rank >= best_rank:
            break
        
        found = False
        # Multiple random restarts
        n_restarts = 5 if target_rank <= 16 else 3
        for restart in range(n_restarts):
            if time.time() > deadline - 20:
                break
            np.random.seed(target_rank * 1000 + restart)
            U = np.random.randn(n*m, target_rank)
            V = np.random.randn(m*p, target_rank)
            W = np.random.randn(p*n, target_rank)
            
            max_iter = 300
            prev_err = float('inf')
            for it in range(max_iter):
                if time.time() > deadline - 15:
                    break
                U = _als_solve_U(T, V, W, n, m, p)
                V = _als_solve_V(T, U, W, n, m, p)
                W = _als_solve_W(T, U, V, n, m, p)
                err = _compute_error(U, V, W, T)
                if abs(prev_err - err) < 1e-14:
                    break
                prev_err = err
            
            if _verify(U, V, W, T, tol=1e-8):
                U_r = _round_to_halfint(U)
                V_r = _round_to_halfint(V)
                W_r = _round_to_halfint(W)
                if _verify(U_r, V_r, W_r, T, tol=1e-9):
                    best_rank = target_rank
                    best_U, best_V, best_W = U_r, V_r, W_r
                    found = True
                    break
        
        if found:
            break
    
    if best_rank < rank_sb:
        return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
    else:
        return {"U": U_sb.tolist(), "V": V_sb.tolist(), "W": W_sb.tolist()}
# EVOLVE-BLOCK-END
