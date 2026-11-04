"""Program for rank-minimal tensor decomposition of 2x3x7 matrix multiplication."""
DIMENSIONS = [2, 3, 7]

# EVOLVE-BLOCK-START
import numpy as np
import time
import random
from scipy.optimize import least_squares

def construct():
    """
    Construct a rank-minimal exact tensor decomposition for multiplying a 2x3 by 3x7 matrix.
    """
    start_time = time.time()
    deadline = 145.0  # Safety margin before 150s kill
    
    n, m, p = DIMENSIONS
    N_U = n * m  # 6
    N_V = m * p  # 21
    N_W = p * n  # 14
    schoolbook_rank = n * m * p  # 42
    
    # Build the target tensor
    T = np.zeros((N_U, N_V, N_W), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                alpha = i * m + j
                beta = j * p + k
                gamma = k * n + i
                T[alpha, beta, gamma] = 1.0
    T_flat = T.flatten()
    
    # Schoolbook solution as fallback
    def schoolbook():
        rank = schoolbook_rank
        U = [[0] * rank for _ in range(N_U)]
        V = [[0] * rank for _ in range(N_V)]
        W = [[0] * rank for _ in range(N_W)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}
    
    best_result = schoolbook()
    best_rank = schoolbook_rank
    
    def verify_exact(U, V, W, tol=1e-9):
        """Verify that U, V, W exactly decompose the tensor."""
        U_arr = np.array(U, dtype=np.float64)
        V_arr = np.array(V, dtype=np.float64)
        W_arr = np.array(W, dtype=np.float64)
        result = np.einsum('rc,sc,tc->rst', U_arr, V_arr, W_arr)
        return np.max(np.abs(result - T)) < tol

    def try_rank_numeric(R, time_budget, max_restarts=20):
        """
        Try to find a valid decomposition of rank R using numerical optimization.
        """
        if R >= best_rank:
            return None
        
        end_time = time.time() + time_budget
        best_cost = np.inf
        best_U = None
        best_V = None
        best_W = None
        
        for restart in range(max_restarts):
            if time.time() > end_time:
                break
            
            # Random initialization
            U = np.random.randn(N_U, R)
            V = np.random.randn(N_V, R)
            W = np.random.randn(N_W, R)
            
            x0 = np.concatenate([U.flatten(), V.flatten(), W.flatten()])
            
            def res(params):
                len_U = N_U * R
                len_V = N_V * R
                U_p = params[:len_U].reshape(N_U, R)
                V_p = params[len_U:len_U+len_V].reshape(N_V, R)
                W_p = params[len_U+len_V:].reshape(N_W, R)
                T_calc = np.einsum('ar,br,cr->abc', U_p, V_p, W_p)
                return (T_calc.flatten() - T_flat)
            
            try:
                sol = least_squares(res, x0, method='lm', max_nfev=1000)
                if sol.cost < best_cost:
                    best_cost = sol.cost
                    best_U = sol.x[:N_U*R].reshape(N_U, R).copy()
                    best_V = sol.x[N_U*R:N_U*R+N_V*R].reshape(N_V, R).copy()
                    best_W = sol.x[N_U*R+N_V*R:].reshape(N_W, R).copy()
                    
                    if best_cost < 1e-10:
                        # Found a very good solution, try to round
                        U_r = np.round(best_U * 2) / 2.0
                        V_r = np.round(best_V * 2) / 2.0
                        W_r = np.round(best_W * 2) / 2.0
                        
                        # Check if rounded solution is exact
                        T_calc = np.einsum('ar,br,cr->abc', U_r, V_r, W_r)
                        if np.max(np.abs(T_calc - T)) < 1e-9:
                            return {"U": U_r.tolist(), "V": V_r.tolist(), "W": W_r.tolist()}
                        
            except Exception:
                continue
        
        # If we found a good numerical solution but rounding failed, we might need more precision or different rounding
        # But for rank 17, integer solutions exist.
        return None

    # The known optimal rank for 2x3x7 is 17.
    # We attempt to find it directly.
    target_rank = 17
    
    # Allocate most of the time to finding rank 17
    time_budget = deadline - time.time() - 5.0
    result = try_rank_numeric(target_rank, time_budget, max_restarts=50)
    
    if result is not None:
        best_result = result
        best_rank = target_rank
    else:
        # If rank 17 fails, try 18, 19... up to a reasonable limit
        for R in range(target_rank + 1, min(schoolbook_rank, target_rank + 5)):
            if time.time() > deadline - 5:
                break
            time_budget = deadline - time.time() - 5.0
            result = try_rank_numeric(R, time_budget, max_restarts=20)
            if result is not None:
                best_result = result
                best_rank = R
                break
    
    return best_result
# EVOLVE-BLOCK-END
