import time
import sys
import numpy as np
from scipy.optimize import minimize

DIMENSIONS = [3, 3, 6]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = 140.0  # 150s limit, keep 10s margin
    
    n, m, p = DIMENSIONS
    # U: n*m x rank = 9 x r
    # V: m*p x rank = 18 x r  
    # W: p*n x rank = 18 x r
    
    # Build the target tensor T[i,j,k] where i in [0,n), j in [0,m), k in [0,p)
    # T[i,j,k] = 1 (for matrix multiplication)
    # The decomposition: T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    
    # We want to find minimum rank r such that this holds exactly
    
    # Start with schoolbook rank
    schoolbook_rank = n * m * p  # 54
    
    # Try to find lower rank decompositions
    # Use a numerical approach: for each rank r, try to find U, V, W
    
    def build_target_tensor():
        """Return the target tensor as a 3D array T[i,j,k]"""
        T = np.zeros((n, m, p))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    T[i,j,k] = 1.0
        return T
    
    T_target = build_target_tensor()
    
    def compute_decomposition(U, V, W):
        """Compute the tensor from U, V, W and return as 3D array T[i,j,k]"""
        r = U.shape[1]
        T = np.zeros((n, m, p))
        for c in range(r):
            # U[:,c] is vector of length n*m
            # V[:,c] is vector of length m*p
            # W[:,c] is vector of length p*n
            u_vec = U[:, c]  # length n*m
            v_vec = V[:, c]  # length m*p
            w_vec = W[:, c]  # length p*n
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        T[i,j,k] += u_vec[i*m+j] * v_vec[j*p+k] * w_vec[k*n+i]
        return T
    
    def compute_error(U, V, W):
        """Compute squared error between decomposition and target"""
        T_comp = compute_decomposition(U, V, W)
        return np.sum((T_comp - T_target)**2)
    
    def flatten_params(U, V, W):
        """Flatten U, V, W into a single vector"""
        return np.concatenate([U.flatten(), V.flatten(), W.flatten()])
    
    def unflatten_params(params, r):
        """Unflatten params into U, V, W"""
        size_u = n * m * r
        size_v = m * p * r
        U = params[:size_u].reshape(n*m, r)
        V = params[size_u:size_u+size_v].reshape(m*p, r)
        W = params[size_u+size_v:].reshape(p*n, r)
        return U, V, W
    
    def objective(params, r):
        """Objective function for optimization"""
        U, V, W = unflatten_params(params, r)
        return compute_error(U, V, W)
    
    # Try ranks from low to high
    best_result = None
    
    # Start from a reasonable lower bound
    # The minimum possible rank is at least max(n*m, m*p, p*n) = max(9, 18, 18) = 18
    # But actually the lower bound is more subtle
    
    # Let's try a range of ranks
    for r in range(18, schoolbook_rank + 1):
        if time.time() > deadline - 20:
            break
            
        # Try multiple random initializations
        found = False
        for trial in range(3):
            if time.time() > deadline - 10:
                break
                
            np.random.seed(42 + r * 100 + trial)
            
            # Random initialization
            U_init = np.random.randn(n*m, r) * 0.1
            V_init = np.random.randn(m*p, r) * 0.1
            W_init = np.random.randn(p*n, r) * 0.1
            
            params_init = flatten_params(U_init, V_init, W_init)
            
            # Use Nelder-Mead or L-BFGS
            result = minimize(
                objective,
                params_init,
                args=(r,),
                method='L-BFGS-B',
                options={'maxiter': 500, 'ftol': 1e-10, 'gtol': 1e-8}
            )
            
            if result.success or result.fun < 1e-6:
                U_opt, V_opt, W_opt = unflatten_params(result.x, r)
                error = compute_error(U_opt, V_opt, W_opt)
                if error < 1e-6:
                    # Found a valid decomposition!
                    # Convert to lists
                    U_list = [[float(U_opt[i,c]) for c in range(r)] for i in range(n*m)]
                    V_list = [[float(V_opt[i,c]) for c in range(r)] for i in range(m*p)]
                    W_list = [[float(W_opt[i,c]) for c in range(r)] for i in range(p*n)]
                    best_result = {"U": U_list, "V": V_list, "W": W_list}
                    found = True
                    break
        
        if found:
            break
    
    # If no better solution found, return schoolbook
    if best_result is None:
        rank = schoolbook_rank
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
        best_result = {"U": U, "V": V, "W": W}
    
    return best_result
# EVOLVE-BLOCK-END
