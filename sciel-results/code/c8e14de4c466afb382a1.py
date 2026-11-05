import numpy as np
from fractions import Fraction
import time
import itertools
from scipy.optimize import minimize

DIMENSIONS = [2, 3, 5]

def _verify_decomposition(U, V, W, n, m, p):
    rank = len(U[0])
    U_arr = np.array([[float(x) for x in row] for row in U], dtype=float)
    V_arr = np.array([[float(x) for x in row] for row in V], dtype=float)
    W_arr = np.array([[float(x) for x in row] for row in W], dtype=float)
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                idx_u = i * m + j
                idx_v = j * p + k
                idx_w = k * n + i
                s = 0.0
                for r in range(rank):
                    s += U_arr[idx_u, r] * V_arr[idx_v, r] * W_arr[idx_w, r]
                if abs(s - 1.0) > 1e-9:
                    return False
    return True

def _build_rank28():
    """Build the known rank-28 decomposition as a fallback."""
    n, m, p = 2, 3, 5
    rank = 28
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    r = 0
    # Part 1: j=2 (10 terms)
    for i in range(n):
        for k in range(p):
            U[i * m + 2][r] = 1
            V[2 * p + k][r] = 1
            W[k * n + i][r] = 1
            r += 1
            
    # Part 2: j=0,1, k=4 (4 terms)
    for i in range(n):
        for j in range(2):
            U[i * m + j][r] = 1
            V[j * p + 4][r] = 1
            W[4 * n + i][r] = 1
            r += 1
            
    # Part 3: Strassen for j=0,1 k=0,1 (7 terms)
    U[0 * m + 0][r] = 1; U[1 * m + 1][r] = 1; V[0 * p + 0][r] = 1; V[1 * p + 1][r] = 1; W[0 * n + 0][r] = 1; W[1 * n + 1][r] = 1; r += 1
    U[1 * m + 0][r] = 1; U[1 * m + 1][r] = 1; V[0 * p + 0][r] = 1; W[0 * n + 1][r] = 1; W[1 * n + 1][r] = -1; r += 1
    U[0 * m + 0][r] = 1; V[0 * p + 1][r] = 1; V[1 * p + 1][r] = -1; W[1 * n + 0][r] = 1; W[1 * n + 1][r] = 1; r += 1
    U[1 * m + 1][r] = 1; V[1 * p + 0][r] = 1; V[0 * p + 0][r] = -1; W[0 * n + 0][r] = 1; W[0 * n + 1][r] = 1; r += 1
    U[0 * m + 0][r] = 1; U[0 * m + 1][r] = 1; V[1 * p + 1][r] = 1; W[0 * n + 0][r] = -1; W[1 * n + 0][r] = 1; r += 1
    U[1 * m + 0][r] = 1; U[0 * m + 0][r] = -1; V[0 * p + 0][r] = 1; V[0 * p + 1][r] = 1; W[1 * n + 1][r] = 1; r += 1
    U[0 * m + 1][r] = 1; U[1 * m + 1][r] = -1; V[1 * p + 0][r] = 1; V[1 * p + 1][r] = 1; W[0 * n + 0][r] = 1; r += 1
    
    # Part 4: Strassen for j=0,1 k=2,3 (7 terms)
    U[0 * m + 0][r] = 1; U[1 * m + 1][r] = 1; V[0 * p + 2][r] = 1; V[1 * p + 3][r] = 1; W[2 * n + 0][r] = 1; W[3 * n + 1][r] = 1; r += 1
    U[1 * m + 0][r] = 1; U[1 * m + 1][r] = 1; V[0 * p + 2][r] = 1; W[2 * n + 1][r] = 1; W[3 * n + 1][r] = -1; r += 1
    U[0 * m + 0][r] = 1; V[0 * p + 3][r] = 1; V[1 * p + 3][r] = -1; W[3 * n + 0][r] = 1; W[3 * n + 1][r] = 1; r += 1
    U[1 * m + 1][r] = 1; V[1 * p + 2][r] = 1; V[0 * p + 2][r] = -1; W[2 * n + 0][r] = 1; W[2 * n + 1][r] = 1; r += 1
    U[0 * m + 0][r] = 1; U[0 * m + 1][r] = 1; V[1 * p + 3][r] = 1; W[2 * n + 0][r] = -1; W[3 * n + 0][r] = 1; r += 1
    U[1 * m + 0][r] = 1; U[0 * m + 0][r] = -1; V[0 * p + 2][r] = 1; V[0 * p + 3][r] = 1; W[3 * n + 1][r] = 1; r += 1
    U[0 * m + 1][r] = 1; U[1 * m + 1][r] = -1; V[1 * p + 2][r] = 1; V[1 * p + 3][r] = 1; W[2 * n + 0][r] = 1; r += 1
    
    return U, V, W

def _search_subtensor_rank(target_rank, n_sub, m_sub, p_sub, num_trials=20, timeout=10):
    """
    Search for a decomposition of the n_sub x m_sub x p_sub multiplication tensor.
    Returns U_sub, V_sub, W_sub as numpy arrays or None.
    """
    start_time = time.time()
    
    def objective(params):
        # Reshape params into U, V, W
        # U: (n_sub * m_sub, rank)
        # V: (m_sub * p_sub, rank)
        # W: (p_sub * n_sub, rank)
        U = params[:target_rank * n_sub * m_sub].reshape(target_rank, n_sub * m_sub).T
        V = params[target_rank * n_sub * m_sub : target_rank * n_sub * m_sub + target_rank * m_sub * p_sub].reshape(target_rank, m_sub * p_sub).T
        W = params[target_rank * n_sub * m_sub + target_rank * m_sub * p_sub :].reshape(target_rank, p_sub * n_sub).T
        
        residual = 0.0
        for i in range(n_sub):
            for j in range(m_sub):
                for k in range(p_sub):
                    # Term is sum_r U[i,j,r] V[j,k,r] W[k,i,r]
                    # Vectorized: dot(U[i,j], V[j,k] * W[k,i])
                    val = np.dot(U[i * m_sub + j], V[j * p_sub + k] * W[k * n_sub + i])
                    residual += (val - 1.0) ** 2
        return residual

    best_obj = np.inf
    best_params = None
    
    for trial in range(num_trials):
        if time.time() - start_time > timeout:
            break
            
        np.random.seed(trial * 101 + 7)
        # Initialize with small random values
        x0 = np.random.randn(target_rank * (n_sub * m_sub + m_sub * p_sub + p_sub * n_sub)) * 0.5
        
        try:
            res = minimize(objective, x0, method='L-BFGS-B', options={'maxiter': 100, 'ftol': 1e-10, 'gtol': 1e-8})
            if res.fun < best_obj:
                best_obj = res.fun
                best_params = res.x
                if res.fun < 1e-6:
                    break
        except:
            continue
            
    if best_obj < 1e-5:
        U = best_params[:target_rank * n_sub * m_sub].reshape(target_rank, n_sub * m_sub).T
        V = best_params[target_rank * n_sub * m_sub : target_rank * n_sub * m_sub + target_rank * m_sub * p_sub].reshape(target_rank, m_sub * p_sub).T
        W = best_params[target_rank * n_sub * m_sub + target_rank * m_sub * p_sub :].reshape(target_rank, p_sub * n_sub).T
        return U, V, W
    return None

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # Safety margin
    
    # Fallback: Rank 28
    U28, V28, W28 = _build_rank28()
    if _verify_decomposition(U28, V28, W28, n, m, p):
        best_U, best_V, best_W = U28, V28, W28
        best_rank = 28
    else:
        raise ValueError("Fallback failed")
        
    # Strategy:
    # 1. The j=2 part (row 2 of A, col 2 of B) can be done in rank 1.
    #    Term: U[i,2]=1, V[2,k]=1, W[k,i]=1 for all i,k.
    # 2. The j=0,1 part is a 2x2x5 multiplication tensor.
    #    We search for its minimal rank.
    
    # We will try to find a decomposition for the 2x2x5 block with ranks 14, 15, 16, 17, 18.
    # If we find rank R_sub for the 2x2x5 block, total rank is 1 + R_sub.
    
    target_ranks = [14, 15, 16, 17, 18, 19, 20]
    
    for r_sub in target_ranks:
        if time.time() - start_time > deadline - 20:
            break
            
        # Search for 2x2x5 decomposition
        result = _search_subtensor_rank(r_sub, 2, 2, 5, num_trials=10, timeout=10)
        
        if result is not None:
            U_sub, V_sub, W_sub = result
            
            # Construct full U, V, W
            total_rank = 1 + r_sub
            U = [[0] * total_rank for _ in range(n * m)]
            V = [[0] * total_rank for _ in range(m * p)]
            W = [[0] * total_rank for _ in range(p * n)]
            
            # Fill the rank-1 term for j=2
            # U[i, 2, 0] = 1
            # V[2, k, 0] = 1
            # W[k, i, 0] = 1
            for i in range(n):
                for k in range(p):
                    U[i * m + 2][0] = 1
                    V[2 * p + k][0] = 1
                    W[k * n + i][0] = 1
            
            # Fill the sub-tensor terms
            for r in range(r_sub):
                # U_sub indices: i (0,1), j (0,1)
                for i in range(2):
                    for j in range(2):
                        U[i * m + j][1 + r] = U_sub[i * 2 + j, r]
                        
                for j in range(2):
                    for k in range(5):
                        V[j * p + k][1 + r] = V_sub[j * 5 + k, r]
                        
                for k in range(5):
                    for i in range(2):
                        W[k * n + i][1 + r] = W_sub[k * 2 + i, r]
            
            # Verify
            if _verify_decomposition(U, V, W, n, m, p):
                best_U, best_V, best_W = U, V, W
                best_rank = total_rank
                break # Found a valid lower rank, stop searching for higher sub-ranks
                
    return {"U": best_U, "V": best_V, "W": best_W}
