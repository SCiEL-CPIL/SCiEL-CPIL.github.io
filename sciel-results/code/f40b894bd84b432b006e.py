"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

DIMENSIONS = [3, 3, 5]

# EVOLVE-BLOCK-START
def _verify(U, V, W, n, m, p):
    """Verify that the decomposition is exact."""
    rank = len(U[0])
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0.0
                for c in range(rank):
                    val += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                if abs(val - 1.0) > 1e-9:
                    return False
    return True

def _schoolbook(n, m, p):
    """Return schoolbook decomposition."""
    rank = n * m * p
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j][col] = 1.0
                V[j*p+k][col] = 1.0
                W[k*n+i][col] = 1.0
                col += 1
    return U, V, W

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 110
    
    # Start with schoolbook
    best_U, best_V, best_W = _schoolbook(n, m, p)
    best_rank = n * m * p
    
    # Try to find lower rank using random initialization + least squares for W
    def try_rank(r, max_trials=100):
        for trial in range(max_trials):
            if time.time() > deadline:
                return None
            
            np.random.seed(trial * 17 + r * 1000)
            
            U_cand = np.random.randn(n*m, r)
            V_cand = np.random.randn(m*p, r)
            
            W_cand = np.zeros((p*n, r))
            valid = True
            for k in range(p):
                for i in range(n):
                    A_mat = np.zeros((m, r))
                    for j in range(m):
                        for c in range(r):
                            A_mat[j, c] = U_cand[i*m+j, c] * V_cand[j*p+k, c]
                    
                    try:
                        w_sol, residuals, rank_mat, sv = np.linalg.lstsq(A_mat, np.ones(m), rcond=1e-12)
                        if np.max(np.abs(A_mat @ w_sol - np.ones(m))) > 1e-6:
                            valid = False
                            break
                        W_cand[k*n+i, :] = w_sol
                    except:
                        valid = False
                        break
                if not valid:
                    break
            
            if valid:
                U_list = [[float(x) for x in row] for row in U_cand]
                V_list = [[float(x) for x in row] for row in V_cand]
                W_list = [[float(x) for x in row] for row in W_cand]
                if _verify(U_list, V_list, W_list, n, m, p):
                    return U_list, V_list, W_list
        
        return None
    
    # Try ranks from 30 to 44
    for r in range(30, best_rank):
        if time.time() > deadline:
            break
        
        result = try_rank(r, max_trials=50)
        if result is not None:
            U_cand, V_cand, W_cand = result
            best_rank = r
            best_U, best_V, best_W = U_cand, V_cand, W_cand
            break
    
    return {"U": best_U, "V": best_V, "W": best_W}
# EVOLVE-BLOCK-END
