"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 3]

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import least_squares

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 150s kill
    
    n, m, p = DIMENSIONS  # 2, 3, 3
    
    # Try rank 16 first via numerical optimization
    if time.time() < deadline - 5.0:
        result = _try_rank_optimization(n, m, p, 16, deadline)
        if result is not None:
            return result
    
    # Fall back to rank 17
    result = _rank17_solution(n, m, p)
    return result

def _try_rank_optimization(n, m, p, target_rank, deadline):
    """Try to find a decomposition of the given rank using least_squares."""
    nparams = (n*m + m*p + p*n) * target_rank
    nresid = n * p * m * m  # 54 for 2,3,3
    
    def residuals(params):
        U = params[:n*m*target_rank].reshape(n*m, target_rank)
        V = params[n*m*target_rank:n*m*target_rank+m*p*target_rank].reshape(m*p, target_rank)
        W = params[n*m*target_rank+m*p*target_rank:].reshape(p*n, target_rank)
        
        resid = np.zeros(nresid)
        idx = 0
        for i in range(n):
            for k in range(p):
                for j1 in range(m):
                    for j2 in range(m):
                        s = np.dot(W[k*n+i, :], U[i*m+j1, :] * V[j2*p+k, :])
                        target = 1.0 if j1 == j2 else 0.0
                        resid[idx] = s - target
                        idx += 1
        return resid
    
    # Get rank 17 solution for initialization
    U17, V17, W17 = _rank17_arrays(n, m, p)
    
    # Initialize: take first target_rank columns from rank 17
    U0 = np.zeros((n*m, target_rank))
    V0 = np.zeros((m*p, target_rank))
    W0 = np.zeros((p*n, target_rank))
    U0[:, :target_rank] = U17[:, :target_rank]
    V0[:, :target_rank] = V17[:, :target_rank]
    W0[:, :target_rank] = W17[:, :target_rank]
    
    params0 = np.concatenate([U0.flatten(), V0.flatten(), W0.flatten()])
    
    try:
        sol = least_squares(residuals, params0, method='lm', max_nfev=50000)
        
        if time.time() > deadline - 2.0:
            return None
        
        if np.max(np.abs(sol.fun)) < 1e-7:
            U_sol = sol.x[:n*m*target_rank].reshape(n*m, target_rank)
            V_sol = sol.x[n*m*target_rank:n*m*target_rank+m*p*target_rank].reshape(m*p, target_rank)
            W_sol = sol.x[n*m*target_rank+m*p*target_rank:].reshape(p*n, target_rank)
            
            if _verify(U_sol, V_sol, W_sol, n, m, p, target_rank):
                return {"U": U_sol.tolist(), "V": V_sol.tolist(), "W": W_sol.tolist()}
    except Exception:
        pass
    
    return None

def _rank17_arrays(n, m, p):
    rank = 17
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    # Part 1: Strassen for A[0:2, 0:2] * B[0:2, 0:2] -> C[0:2, 0:2]
    U[0, 0] = 1; U[4, 0] = 1
    V[0, 0] = 1; V[4, 0] = 1
    W[0, 0] = 1; W[3, 0] = 1
    
    U[3, 1] = 1; U[4, 1] = 1
    V[0, 1] = 1
    W[1, 1] = 1; W[3, 1] = -1
    
    U[0, 2] = 1
    V[1, 2] = 1; V[4, 2] = -1
    W[2, 2] = 1; W[3, 2] = 1
    
    U[4, 3] = 1
    V[3, 3] = 1; V[0, 3] = -1
    W[0, 3] = 1; W[1, 3] = 1
    
    U[0, 4] = 1; U[1, 4] = 1
    V[4, 4] = 1
    W[0, 4] = -1; W[2, 4] = 1
    
    U[3, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[1, 5] = 1
    W[3, 5] = 1
    
    U[1, 6] = 1; U[4, 6] = -1
    V[3, 6] = 1; V[4, 6] = 1
    W[0, 6] = 1
    
    # Part 2: Schoolbook for A[0:2, 2] * B[2, 0:2] -> C[0:2, 0:2]
    U[2, 7] = 1
    V[6, 7] = 1
    W[0, 7] = 1
    
    U[2, 8] = 1
    V[7, 8] = 1
    W[2, 8] = 1
    
    U[5, 9] = 1
    V[6, 9] = 1
    W[1, 9] = 1
    
    U[5, 10] = 1
    V[7, 10] = 1
    W[3, 10] = 1
    
    # Part 3: Schoolbook for A[0:2, 0:3] * B[0:3, 2] -> C[0:2, 2]
    U[0, 11] = 1
    V[2, 11] = 1
    W[4, 11] = 1
    
    U[1, 12] = 1
    V[5, 12] = 1
    W[4, 12] = 1
    
    U[2, 13] = 1
    V[8, 13] = 1
    W[4, 13] = 1
    
    U[3, 14] = 1
    V[2, 14] = 1
    W[5, 14] = 1
    
    U[4, 15] = 1
    V[5, 15] = 1
    W[5, 15] = 1
    
    U[5, 16] = 1
    V[8, 16] = 1
    W[5, 16] = 1
    
    return U, V, W

def _rank17_solution(n, m, p):
    U, V, W = _rank17_arrays(n, m, p)
    
    if not _verify(U, V, W, n, m, p, 17):
        rank_sb = n * m * p
        U_fb = np.zeros((n * m, rank_sb))
        V_fb = np.zeros((m * p, rank_sb))
        W_fb = np.zeros((p * n, rank_sb))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_fb[i*m+j, col] = 1
                    V_fb[j*p+k, col] = 1
                    W_fb[k*n+i, col] = 1
                    col += 1
        return {"U": U_fb.tolist(), "V": V_fb.tolist(), "W": W_fb.tolist()}
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

def _verify(U_mat, V_mat, W_mat, n_, m_, p_, rank_):
    for i in range(n_):
        for k in range(p_):
            for j1 in range(m_):
                for j2 in range(m_):
                    s = np.sum(W_mat[k*n_+i, :] * U_mat[i*m_+j1, :] * V_mat[j2*p_+k, :])
                    target = 1.0 if j1 == j2 else 0.0
                    if abs(s - target) > 1e-8:
                        return False
    return True
# EVOLVE-BLOCK-END
