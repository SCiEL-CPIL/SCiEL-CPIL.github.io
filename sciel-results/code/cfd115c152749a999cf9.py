import time
import numpy as np
from scipy.optimize import least_squares

DIMENSIONS = [3, 3, 8]
DEADLINE_S = 110  # safety margin from 120s limit

def construct():
    start_time = time.time()
    n, m, p = DIMENSIONS
    nm, mp, pn = n*m, m*p, p*n
    
    def time_left():
        return DEADLINE_S - (time.time() - start_time)
    
    def schoolbook():
        rank = n*m*p
        U = np.zeros((nm, rank))
        V = np.zeros((mp, rank))
        W = np.zeros((pn, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m + j, col] = 1.0
                    V[j*p + k, col] = 1.0
                    W[k*n + i, col] = 1.0
                    col += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    def validate(U, V, W, tol=1e-10):
        """Check that sum_c U[:,c] V[:,c] W[:,c] == multiplication tensor."""
        U = np.array(U)
        V = np.array(V)
        W = np.array(W)
        rank = U.shape[1]
        if V.shape[1] != rank or W.shape[1] != rank:
            return False
        
        T_computed = np.zeros((n, m, p))
        for c in range(rank):
            u_mat = U[:, c].reshape(n, m)
            v_mat = V[:, c].reshape(m, p)
            w_mat = W[:, c].reshape(p, n)
            for i in range(n):
                for j in range(m):
                    T_computed[i,j,:] += u_mat[i,j] * (v_mat[j,:] * w_mat[:,i])
        
        return np.max(np.abs(T_computed - 1.0)) < tol

    def get_17_rank_block():
        """Returns U, V, W for a 3x3x2 block with rank 17."""
        rank = 17
        U = np.zeros((9, rank))
        V = np.zeros((6, rank))
        W = np.zeros((6, rank))
        
        col = 0
        # Strassen part (7 terms) for the 2x2x2 sub-block (rows 0,1; cols 0,1)
        U_coeffs = [
            {0: 1, 3: 1},       
            {2: 1, 3: 1},       
            {0: 1},             
            {3: 1},             
            {0: 1, 1: 1},       
            {2: 1, 0: -1},      
            {1: 1, 3: -1}       
        ]
        V_coeffs = [
            {0: 1, 3: 1},       
            {0: 1},             
            {1: 1, 3: -1},      
            {2: 1, 0: -1},      
            {3: 1},             
            {0: 1, 1: 1},       
            {2: 1, 3: 1}        
        ]
        W_coeffs = [
            {0: 1, 3: 1},       
            {2: 1, 3: -1},      
            {1: 1, 3: 1},       
            {0: 1, 2: 1},       
            {0: -1, 1: 1},      
            {3: 1},             
            {0: 1}              
        ]
        
        for l in range(7):
            for li, coeff in U_coeffs[l].items():
                i_local = li // 2
                j_local = li % 2
                U[i_local*3 + j_local, col] += coeff
            for lj, coeff in V_coeffs[l].items():
                j_local = lj // 2
                k_local = lj % 2
                V[j_local*2 + k_local, col] += coeff
            for lc, coeff in W_coeffs[l].items():
                i_local = lc // 2
                k_local = lc % 2
                W[k_local*3 + i_local, col] += coeff
            col += 1
        
        # Schoolbook part (10 terms) for the remaining parts
        # i in {0,1}, j=2, k in {0,1}
        for i in range(2):
            for k in range(2):
                U[i*3 + 2, col] = 1.0
                V[2*2 + k, col] = 1.0
                W[k*3 + i, col] = 1.0
                col += 1
        # i=2, j in {0,1}, k in {0,1}
        for j in range(2):
            for k in range(2):
                U[2*3 + j, col] = 1.0
                V[j*2 + k, col] = 1.0
                W[k*3 + 2, col] = 1.0
                col += 1
        # i=2, j=2, k in {0,1}
        for k in range(2):
            U[2*3 + 2, col] = 1.0
            V[2*2 + k, col] = 1.0
            W[k*3 + 2, col] = 1.0
            col += 1
            
        return U, V, W

    def try_find_rank_16_block():
        """Try to find a rank 16 decomposition for 3x3x2."""
        U17, V17, W17 = get_17_rank_block()
        
        def compute_tensor(U, V, W):
            T = np.zeros((3, 3, 2))
            for c in range(U.shape[1]):
                u = U[:, c].reshape(3, 3)
                v = V[:, c].reshape(3, 2)
                w = W[:, c].reshape(2, 3)
                for i in range(3):
                    for j in range(3):
                        T[i,j,:] += u[i,j] * (v[j,:] * w[:,i])
            return T
        
        def residuals(x):
            U = x[:144].reshape(9, 16)
            V = x[144:240].reshape(6, 16)
            W = x[240:336].reshape(6, 16)
            T = compute_tensor(U, V, W)
            return (T - 1.0).flatten()
        
        def check_rounded_solution(U_opt, V_opt, W_opt):
            for scale in [1, 2, 4, 8, 16]:
                U_rounded = np.round(U_opt * scale) / scale
                V_rounded = np.round(V_opt * scale) / scale
                W_rounded = np.round(W_opt * scale) / scale
                
                T_check = compute_tensor(U_rounded, V_rounded, W_rounded)
                if np.max(np.abs(T_check - 1.0)) < 1e-10:
                    return U_rounded, V_rounded, W_rounded
            return None
        
        def polish(x0, max_iter=500):
            try:
                result = least_squares(residuals, x0, method='lm', max_nfev=max_iter, ftol=1e-15, xtol=1e-15, gtol=1e-15)
                return result.x, np.max(np.abs(result.fun))
            except Exception:
                return x0, np.max(np.abs(residuals(x0)))
        
        best_result = None
        best_residual = np.inf
        
        # Phase 1: Try dropping each of the 17 columns
        for drop_idx in range(17):
            if time_left() < 5:
                break
            keep_cols = [c for c in range(17) if c != drop_idx]
            U_init = U17[:, keep_cols].flatten()
            V_init = V17[:, keep_cols].flatten()
            W_init = W17[:, keep_cols].flatten()
            x0 = np.concatenate([U_init, V_init, W_init])
            
            x_opt, residual = polish(x0, max_iter=300)
            if residual < best_residual:
                best_residual = residual
                best_result = x_opt
                if residual < 1e-8:
                    U_opt = x_opt[:144].reshape(9, 16)
                    V_opt = x_opt[144:240].reshape(6, 16)
                    W_opt = x_opt[240:336].reshape(6, 16)
                    rounded = check_rounded_solution(U_opt, V_opt, W_opt)
                    if rounded is not None:
                        return rounded
        
        # Phase 2: Random perturbations
        rng = np.random.default_rng(42)
        for attempt in range(20): # Increased attempts
            if time_left() < 5:
                break
            drop_idx = rng.integers(0, 17)
            keep_cols = [c for c in range(17) if c != drop_idx]
            U_init = U17[:, keep_cols].copy()
            V_init = V17[:, keep_cols].copy()
            W_init = W17[:, keep_cols].copy()
            
            noise_scale = 0.05 + 0.01 * attempt
            U_init += rng.normal(0, noise_scale, U_init.shape)
            V_init += rng.normal(0, noise_scale, V_init.shape)
            W_init += rng.normal(0, noise_scale, W_init.shape)
            
            x0 = np.concatenate([U_init.flatten(), V_init.flatten(), W_init.flatten()])
            
            x_opt, residual = polish(x0, max_iter=200)
            if residual < best_residual:
                best_residual = residual
                best_result = x_opt
                if residual < 1e-8:
                    U_opt = x_opt[:144].reshape(9, 16)
                    V_opt = x_opt[144:240].reshape(6, 16)
                    W_opt = x_opt[240:336].reshape(6, 16)
                    rounded = check_rounded_solution(U_opt, V_opt, W_opt)
                    if rounded is not None:
                        return rounded
        
        return None

    # Try to find rank 16 block
    block_16 = None
    if time_left() > 10:
        block_16 = try_find_rank_16_block()
    
    if block_16 is not None:
        U_block, V_block, W_block = block_16
        rank_block = 16
    else:
        U_block, V_block, W_block = get_17_rank_block()
        rank_block = 17
        
    # Assemble full 3x3x8 decomposition
    rank_total = 4 * rank_block
    U = np.zeros((9, rank_total))
    V = np.zeros((24, rank_total))
    W = np.zeros((24, rank_total))
    
    for g in range(4):
        k0 = 2 * g
        col_start = g * rank_block
        col_end = (g + 1) * rank_block
        
        U[:, col_start:col_end] = U_block
        
        for c in range(rank_block):
            for j in range(3):
                for k_local in range(2):
                    k_global = k0 + k_local
                    V[j*8 + k_global, col_start + c] = V_block[j*2 + k_local, c]
                    
        for c in range(rank_block):
            for k_local in range(2):
                k_global = k0 + k_local
                for i in range(3):
                    W[k_global*3 + i, col_start + c] = W_block[k_local*3 + i, c]

    result = {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    if validate(U, V, W):
        return result
    else:
        return schoolbook()
