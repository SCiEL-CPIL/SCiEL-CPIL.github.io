import numpy as np
from scipy.optimize import minimize
import time

DIMENSIONS = [3, 3, 8]

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10s margin

    n, m, p_small = 3, 3, 2
    
    # Target tensor for 3x3x2
    target = np.zeros((n*m, m*p_small, p_small*n))
    for i in range(n):
        for j in range(m):
            for k in range(p_small):
                u_idx = i * m + j
                v_idx = j * p_small + k
                w_idx = k * n + i
                target[u_idx, v_idx, w_idx] = 1.0
                
    def check_decomp(U, V, W, target_tensor, tol=1e-9):
        T = np.einsum('ir,jr,kr->ijk', U, V, W)
        return np.allclose(T, target_tensor, atol=tol)
        
    def find_decomp_332(R, max_restarts=20):
        for _ in range(max_restarts):
            if time.time() > deadline:
                return None
            
            # Random initialization
            # Use a mix of random and structured initialization
            U = np.random.randn(n*m, R) * 0.5
            V = np.random.randn(m*p_small, R) * 0.5
            W = np.random.randn(p_small*n, R) * 0.5
            
            x0 = np.concatenate([U.flatten(), V.flatten(), W.flatten()])
            
            def objective(x):
                U_flat, V_flat, W_flat = x[:n*m*R], x[n*m*R:n*m*R+m*p_small*R], x[n*m*R+m*p_small*R:]
                U_curr = U_flat.reshape(n*m, R)
                V_curr = V_flat.reshape(m*p_small, R)
                W_curr = W_flat.reshape(p_small*n, R)
                
                T = np.einsum('ir,jr,kr->ijk', U_curr, V_curr, W_curr)
                error = np.sum((T - target)**2)
                return error
            
            # Use Nelder-Mead with tighter tolerances
            res = minimize(objective, x0, method='Nelder-Mead', options={'maxiter': 5000, 'xatol': 1e-10, 'fatol': 1e-12})
            
            if res.fun < 1e-8:
                U_flat, V_flat, W_flat = res.x[:n*m*R], res.x[n*m*R:n*m*R+m*p_small*R], res.x[n*m*R+m*p_small*R:]
                U_sol = U_flat.reshape(n*m, R)
                V_sol = V_flat.reshape(m*p_small, R)
                W_sol = W_flat.reshape(p_small*n, R)
                
                # Snap to half-integers
                U_snap = np.round(U_sol * 2) / 2
                V_snap = np.round(V_sol * 2) / 2
                W_snap = np.round(W_sol * 2) / 2
                
                if check_decomp(U_snap, V_snap, W_snap, target):
                    return U_snap, V_snap, W_snap
                
                # Try snapping to integers? (subset of half-integers)
                U_snap_int = np.round(U_sol)
                V_snap_int = np.round(V_sol)
                W_snap_int = np.round(W_sol)
                if check_decomp(U_snap_int, V_snap_int, W_snap_int, target):
                    return U_snap_int, V_snap_int, W_snap_int

        return None

    # Try ranks for 3x3x2
    # Prioritize lower ranks
    U_332 = V_332 = W_332 = None
    for R in [16, 17, 18]:
        if time.time() > deadline:
            break
        res = find_decomp_332(R, max_restarts=10 if R > 16 else 30)
        if res:
            U_332, V_332, W_332 = res
            break

    if U_332 is not None:
        n, m, p = DIMENSIONS
        R_block = len(U_332[0])
        R_total = 4 * R_block
        
        U = np.zeros((n*m, R_total))
        V = np.zeros((m*p, R_total))
        W = np.zeros((p*n, R_total))
        
        for b in range(4):
            offset = b * R_block
            
            U[:, offset:offset+R_block] = U_332
            
            for j in range(m):
                for l in range(2):
                    k = 2 * b + l
                    V[j * p + k, offset:offset+R_block] = V_332[j * 2 + l, :]
            
            for l in range(2):
                for i in range(n):
                    k = 2 * b + l
                    W[k * n + i, offset:offset+R_block] = W_332[l * n + i, :]
                    
        # Verify
        target_338 = np.zeros((n*m, m*p, p*n))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u_idx = i * m + j
                    v_idx = j * p + k
                    w_idx = k * n + i
                    target_338[u_idx, v_idx, w_idx] = 1.0
        
        if check_decomp(U, V, W, target_338):
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    # Fallback: Schoolbook
    n, m, p = DIMENSIONS
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}
