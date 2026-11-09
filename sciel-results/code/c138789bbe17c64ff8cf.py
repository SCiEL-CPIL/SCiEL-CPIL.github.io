import numpy as np
import time
from scipy.optimize import least_squares

def construct():
    n, m, p = 2, 3, 4
    deadline = time.time() + 100 # 100s limit
    
    # Schoolbook solution (Rank 24)
    rank_sb = n * m * p
    U_sb = np.zeros((n * m, rank_sb))
    V_sb = np.zeros((m * p, rank_sb))
    W_sb = np.zeros((p * n, rank_sb))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                c = i * m * p + j * p + k
                U_sb[i * m + j, c] = 1
                V_sb[j * p + k, c] = 1
                W_sb[k * n + i, c] = 1
                
    best_result = {"U": U_sb.tolist(), "V": V_sb.tolist(), "W": W_sb.tolist()}
    
    def verify(U, V, W):
        # U, V, W are lists of lists
        n, m, p = 2, 3, 4
        R = len(U[0])
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for c in range(R):
                        s += U[i * m + j][c] * V[j * p + k][c] * W[k * n + i][c]
                    if abs(s - 1.0) > 1e-9:
                        return False
        return True

    def solve_rank(R, max_restarts=20):
        for restart in range(max_restarts):
            if time.time() > deadline:
                return None
            
            np.random.seed(42 + restart)
            # Initialize with random values
            U_init = np.random.randn(n * m, R)
            V_init = np.random.randn(m * p, R)
            W_init = np.random.randn(p * n, R)
            
            x0 = np.concatenate([U_init.flatten(), V_init.flatten(), W_init.flatten()])
            
            def residual(x):
                U = x[:n * m * R].reshape(n * m, R)
                V = x[n * m * R : n * m * R + m * p * R].reshape(m * p, R)
                W = x[n * m * R + m * p * R :].reshape(p * n, R)
                
                res = np.zeros((n, m, p))
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            # res[i,j,k] = sum_c U[i,j,c] V[j,k,c] W[k,i,c]
                            # Vectorized: dot product of the row vectors
                            res[i, j, k] = np.dot(U[i * m + j], V[j * p + k] * W[k * n + i]) - 1.0
                return res.flatten()
                
            try:
                sol = least_squares(residual, x0, method='lm', max_nfev=1000)
                if sol.success and np.linalg.norm(sol.fun) < 1e-6:
                    U_sol = sol.x[:n * m * R].reshape(n * m, R)
                    V_sol = sol.x[n * m * R : n * m * R + m * p * R].reshape(m * p, R)
                    W_sol = sol.x[n * m * R + m * p * R :].reshape(p * n, R)
                    
                    # Round to nearest half-integer
                    U_round = np.round(U_sol * 2) / 2
                    V_round = np.round(V_sol * 2) / 2
                    W_round = np.round(W_sol * 2) / 2
                    
                    # Verify rounded solution
                    if verify(U_round.tolist(), V_round.tolist(), W_round.tolist()):
                        return {
                            "U": U_round.tolist(),
                            "V": V_round.tolist(),
                            "W": W_round.tolist()
                        }
            except Exception:
                pass
        return None

    # Try rank 11
    result = solve_rank(11)
    if result:
        return result
        
    # Fallback to schoolbook
    return best_result
