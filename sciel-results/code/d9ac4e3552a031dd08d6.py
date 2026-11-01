import numpy as np
import time
from scipy.optimize import least_squares

DIMENSIONS = [2, 3, 9]

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140  # 10s margin
    
    # Strategy: Decompose 2x3x3 and replicate 3 times for 2x3x9.
    # Rank(2x3x9) = 3 * Rank(2x3x3)
    
    # First, try to find a low-rank decomposition for 2x3x3
    # n2, m2, p2 = 2, 3, 3
    n2, m2, p2 = 2, 3, 3
    
    best_233 = None
    best_rank_233 = None
    
    # Try ranks for 2x3x3
    # Schoolbook rank is 2*3*3 = 18.
    # Try 11 to 18.
    for r2 in range(11, 19):
        if time.time() > deadline:
            break
            
        # Try a few restarts
        found = False
        for restart in range(50):
            if time.time() > deadline - 10:
                break
                
            np.random.seed(restart * 100 + r2)
            
            # Initialize U, V, W for 2x3x3
            # U: (n2*m2, r2) = (6, r2)
            # V: (m2*p2, r2) = (9, r2)
            # W: (p2*n2, r2) = (6, r2)
            
            U = np.random.randn(n2 * m2, r2) * 0.5
            V = np.random.randn(m2 * p2, r2) * 0.5
            W = np.random.randn(p2 * n2, r2) * 0.5
            
            # Gauge fixing: U[0, :] = 1
            U[0, :] = 1.0
            
            params = np.concatenate([U.flatten(), V.flatten(), W.flatten()])
            
            def residual_233(params):
                U_p = params[0:n2*m2*r2].reshape(n2*m2, r2)
                V_p = params[n2*m2*r2:n2*m2*r2 + m2*p2*r2].reshape(m2*p2, r2)
                W_p = params[n2*m2*r2 + m2*p2*r2:].reshape(p2*n2, r2)
                
                res = np.zeros(n2*m2*p2)
                idx = 0
                for i in range(n2):
                    for j in range(m2):
                        for k in range(p2):
                            u_idx = i*m2 + j
                            v_idx = j*p2 + k
                            w_idx = k*n2 + i
                            # sum_r U[u_idx, r] V[v_idx, r] W[w_idx, r]
                            res[idx] = np.dot(U_p[u_idx, :] * V_p[v_idx, :], W_p[w_idx, :])
                            idx += 1
                return res - 1.0
            
            try:
                sol = least_squares(residual_233, params, method='lm', max_nfev=2000)
            except:
                continue
                
            if np.linalg.norm(sol.fun) < 1e-8:
                U_sol = sol.x[0:n2*m2*r2].reshape(n2*m2, r2)
                V_sol = sol.x[n2*m2*r2:n2*m2*r2 + m2*p2*r2].reshape(m2*p2, r2)
                W_sol = sol.x[n2*m2*r2 + m2*p2*r2:].reshape(p2*n2, r2)
                
                # Snap to half-integers
                U_snap = np.round(U_sol * 2) / 2
                V_snap = np.round(V_sol * 2) / 2
                W_snap = np.round(W_sol * 2) / 2
                
                # Verify
                valid = True
                for i in range(n2):
                    for j in range(m2):
                        for k in range(p2):
                            u_idx = i*m2 + j
                            v_idx = j*p2 + k
                            w_idx = k*n2 + i
                            s = np.dot(U_snap[u_idx, :] * V_snap[v_idx, :], W_snap[w_idx, :])
                            if abs(s - 1.0) > 1e-9:
                                valid = False
                                break
                        if not valid: break
                    if not valid: break
                
                if valid:
                    best_233 = (U_snap, V_snap, W_snap)
                    best_rank_233 = r2
                    found = True
                    break
        
        if found:
            break

    if best_233 is not None:
        U233, V233, W233 = best_233
        r2 = best_rank_233
        
        # Construct 2x3x9 solution
        # Rank R = 3 * r2
        R = 3 * r2
        
        U = np.zeros((n * m, R))
        V = np.zeros((m * p, R))
        W = np.zeros((p * n, R))
        
        for block in range(3):
            # U is shared, so copy U233 to the corresponding block of columns
            U[:, block*r2:(block+1)*r2] = U233
            
            # V: m x p = 3 x 9. Block 'block' corresponds to columns block*3 to block*3+2
            # V233 is 3x3 (m2 x p2).
            # V[j, block*3 + k] = V233[j, k]
            for j in range(m2):
                for k in range(p2):
                    V[j, block*p2 + k] = V233[j, k]
            
            # W: p x n = 9 x 2. Block 'block' corresponds to rows block*3 to block*3+2
            # W233 is 3x2 (p2 x n2).
            # W[block*3 + k, i] = W233[k, i]
            for k in range(p2):
                for i in range(n2):
                    W[block*p2 + k, i] = W233[k, i]
        
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    else:
        # Fallback to schoolbook 2x3x9
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
