import numpy as np
import time

def construct():
    start_time = time.time()
    deadline = start_time + 140  # safety margin for 150s limit
    
    n, m, p = 2, 3, 6
    
    # Build target tensor T for 2x3x6
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m+j, j*p+k, k*n+i] = 1.0
                
    # Strategy: Decompose 2x3x6 into two 2x3x3 blocks
    # Find rank-minimal decomposition for 2x3x3
    n2, m2, p2 = 2, 3, 3
    T233 = np.zeros((n2*m2, m2*p2, p2*n2))
    for i in range(n2):
        for j in range(m2):
            for k in range(p2):
                T233[i*m2+j, j*p2+k, k*n2+i] = 1.0
                
    rows_u = n2 * m2  # 6
    rows_v = m2 * p2  # 9
    rows_w = p2 * n2  # 6
    
    def generate_Z_VW(V, W, R):
        Z = np.zeros((R, rows_v * rows_w))
        for r in range(R):
            outer = np.outer(V[:, r], W[:, r])
            Z[r, :] = outer.flatten()
        return Z
        
    def generate_Z_UW(U, W, R):
        Z = np.zeros((R, rows_u * rows_w))
        for r in range(R):
            outer = np.outer(U[:, r], W[:, r])
            Z[r, :] = outer.flatten()
        return Z
        
    def generate_Z_UV(U, V, R):
        Z = np.zeros((R, rows_u * rows_v))
        for r in range(R):
            outer = np.outer(U[:, r], V[:, r])
            Z[r, :] = outer.flatten()
        return Z
        
    def solve_U(V, W, T_target, R):
        Z = generate_Z_VW(V, W, R)
        T_flat = T_target.flatten().reshape(rows_u, -1)
        try:
            U_sol, _, _, _ = np.linalg.lstsq(Z.T, T_flat.T, rcond=None)
            return U_sol.T
        except:
            return None
            
    def solve_V(U, W, T_target, R):
        Z = generate_Z_UW(U, W, R)
        T_flat = T_target.transpose(1, 0, 2).flatten().reshape(rows_v, -1)
        try:
            V_sol, _, _, _ = np.linalg.lstsq(Z.T, T_flat.T, rcond=None)
            return V_sol.T
        except:
            return None
            
    def solve_W(U, V, T_target, R):
        Z = generate_Z_UV(U, V, R)
        T_flat = T_target.transpose(2, 0, 1).flatten().reshape(rows_w, -1)
        try:
            W_sol, _, _, _ = np.linalg.lstsq(Z.T, T_flat.T, rcond=None)
            return W_sol.T
        except:
            return None
            
    def als_decompose(T_target, R, max_iter=50, tol=1e-8, seed=0):
        rng = np.random.RandomState(seed)
        U = rng.randn(rows_u, R) * 0.5
        V = rng.randn(rows_v, R) * 0.5
        W = rng.randn(rows_w, R) * 0.5
        
        for it in range(max_iter):
            U_new = solve_U(V, W, T_target, R)
            if U_new is None:
                return None
            V_new = solve_V(U_new, W, T_target, R)
            if V_new is None:
                return None
            W_new = solve_W(U_new, V_new, T_target, R)
            if W_new is None:
                return None
            
            diff = np.max(np.abs(U_new - U)) + np.max(np.abs(V_new - V)) + np.max(np.abs(W_new - W))
            U, V, W = U_new, V_new, W_new
            
            if diff < tol:
                # Round to nearest half-integer
                U_r = np.round(U * 2) / 2
                V_r = np.round(V * 2) / 2
                W_r = np.round(W * 2) / 2
                
                T_check = np.einsum('ir,jr,kr->ijk', U_r, V_r, W_r)
                if np.allclose(T_check, T_target, atol=1e-9):
                    return (U_r, V_r, W_r)
                
                # Try rounding to nearest integer
                U_i = np.round(U)
                V_i = np.round(V)
                W_i = np.round(W)
                T_check_i = np.einsum('ir,jr,kr->ijk', U_i, V_i, W_i)
                if np.allclose(T_check_i, T_target, atol=1e-9):
                    return (U_i, V_i, W_i)
                    
        return None

    # Try ranks from 12 to 15 for the 2x3x3 block
    best_233 = None
    best_rank = 18 # Fallback
    
    for R233 in range(12, 16):
        if time.time() > deadline - 10:
            break
            
        found = False
        max_restarts = 20
        for restart in range(max_restarts):
            if time.time() > deadline - 5:
                break
                
            result = als_decompose(T233, R233, max_iter=50, seed=restart)
            if result is not None:
                best_233 = result
                best_rank = R233
                found = True
                break
                
        if found:
            break
            
    if best_233 is not None:
        U233, V233, W233 = best_233
        R236 = 2 * best_rank
        U236 = np.zeros((n*m, R236))
        V236 = np.zeros((m*p, R236))
        W236 = np.zeros((p*n, R236))
        
        # Block 1
        U236[:, :best_rank] = U233
        V236[:m2*p2, :best_rank] = V233
        W236[:p2*n2, :best_rank] = W233
        
        # Block 2
        U236[:, best_rank:] = U233
        V236[m2*p2:, best_rank:] = V233
        W236[p2*n2:, best_rank:] = W233
        
        # Verify full tensor
        T_final = np.einsum('ir,jr,kr->ijk', U236, V236, W236)
        if np.allclose(T_final, T, atol=1e-9):
            return {
                "U": U236.tolist(),
                "V": V236.tolist(),
                "W": W236.tolist()
            }
            
    # Fallback to schoolbook rank 36
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
