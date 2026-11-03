import time
import numpy as np
from scipy.optimize import minimize

def construct():
    n, m, p = 3, 3, 4
    start_time = time.time()
    deadline = start_time + 140

    # Full tensor T[a, b, c]
    T = np.zeros((n*m, m*p, p*n), dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1.0

    # Schoolbook fallback
    rank_fb = n * m * p
    U_fb = np.zeros((n*m, rank_fb), dtype=float)
    V_fb = np.zeros((m*p, rank_fb), dtype=float)
    W_fb = np.zeros((p*n, rank_fb), dtype=float)
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_fb[i*m+j, col] = 1
                V_fb[j*p+k, col] = 1
                W_fb[k*n+i, col] = 1
                col += 1
    
    best_result = (U_fb, V_fb, W_fb)
    best_rank = rank_fb

    def compute_decomp(U, V, W):
        return np.einsum('at,bt,ct->abc', U, V, W, optimize=True)

    def check_decomp(U, V, W, T_target):
        T_decomp = compute_decomp(U, V, W)
        return np.allclose(T_decomp, T_target, atol=1e-9)

    def snap_to_half(U, V, W):
        U_s = np.round(U * 2) / 2
        V_s = np.round(V * 2) / 2
        W_s = np.round(W * 2) / 2
        return U_s, V_s, W_s

    # 1. Construct the 3x3x3 target tensor T333
    # T333[a, b_local, c_local]
    # a = i*3 + j
    # b_local = j*3 + k (k in 0..2)
    # c_local = k*3 + i
    T333 = np.zeros((9, 9, 9), dtype=float)
    for i in range(3):
        for j in range(3):
            for k in range(3):
                a = i * 3 + j
                b_local = j * 3 + k
                c_local = k * 3 + i
                T333[a, b_local, c_local] = 1.0

    def objective_333(params, rank):
        U = params[:9*rank].reshape(9, rank)
        V = params[9*rank:18*rank].reshape(9, rank)
        W = params[18*rank:].reshape(9, rank)
        T_decomp = np.einsum('at,bt,ct->abc', U, V, W, optimize=True)
        return np.sum((T333 - T_decomp) ** 2)

    def try_rank_333(rank, num_trials=50):
        """Find rank-r decomposition for 3x3x3 tensor."""
        for trial in range(num_trials):
            if time.time() > deadline - 20:
                return None
            
            np.random.seed(trial * 100 + rank)
            U = np.random.randn(9, rank) * 0.5
            V = np.random.randn(9, rank) * 0.5
            W = np.random.randn(9, rank) * 0.5
            params = np.concatenate([U.flatten(), V.flatten(), W.flatten()])
            
            res = minimize(objective_333, params, args=(rank,), method='L-BFGS-B', 
                           options={'maxiter': 2000, 'maxfun': 20000, 'gtol': 1e-10})
            
            if res.fun < 1e-8:
                U_opt = res.x[:9*rank].reshape(9, rank)
                V_opt = res.x[9*rank:18*rank].reshape(9, rank)
                W_opt = res.x[18*rank:].reshape(9, rank)
                
                if check_decomp(U_opt, V_opt, W_opt, T333):
                    return U_opt, V_opt, W_opt
        return None

    # 2. Construct the fixed 9 terms for the 4th column (k=3)
    # U_fixed, V_fixed, W_fixed of shape (9,9), (12,9), (12,9)
    U_fixed = np.zeros((9, 9), dtype=float)
    V_fixed = np.zeros((12, 9), dtype=float)
    W_fixed = np.zeros((12, 9), dtype=float)
    
    for i in range(3):
        for j in range(3):
            col_idx = i * 3 + j
            U_fixed[i*3+j, col_idx] = 1.0
            V_fixed[j*4+3, col_idx] = 1.0 # k=3
            W_fixed[3*3+i, col_idx] = 1.0  # k=3, so 3*3+i = 9+i

    # 3. Search for rank r for 3x3x3
    # Try r from 23 to 27
    for r in range(23, 28):
        if time.time() > deadline - 30:
            break
        
        res_333 = try_rank_333(r, num_trials=20)
        if res_333 is not None:
            U3, V3, W3 = res_333
            
            # Map to full tensor
            U_full = np.zeros((9, r + 9), dtype=float)
            V_full = np.zeros((12, r + 9), dtype=float)
            W_full = np.zeros((12, r + 9), dtype=float)
            
            # Copy 3x3x3 part
            U_full[:, :r] = U3
            W_full[:, :r] = W3
            for j in range(3):
                for k in range(3):
                    b_local = j * 3 + k
                    b_full = j * 4 + k
                    V_full[b_full, :r] = V3[b_local, :r]
            
            # Copy fixed 4th column part
            U_full[:, r:] = U_fixed
            V_full[:, r:] = V_fixed
            W_full[:, r:] = W_fixed
            
            # Verify
            if check_decomp(U_full, V_full, W_full, T):
                best_result = (U_full, V_full, W_full)
                best_rank = r + 9
                break

    U_best, V_best, W_best = best_result
    return {"U": U_best.tolist(), "V": V_best.tolist(), "W": W_best.tolist()}
