"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 5]

# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import minimize

def construct():
    start_time = time.time()
    deadline = start_time + 140  # safety margin before 150s kill
    
    n, m, p = DIMENSIONS
    # Schoolbook as fallback
    rank_school = n * m * p
    U_school = [[0] * rank_school for _ in range(n * m)]
    V_school = [[0] * rank_school for _ in range(m * p)]
    W_school = [[0] * rank_school for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j][column] = 1
                V_school[j * p + k][column] = 1
                W_school[k * n + i][column] = 1
                column += 1
    
    best_result = {"U": U_school, "V": V_school, "W": W_school}
    best_rank = rank_school

    # Precompute indices for faster error calculation
    # T[i, j, k] = 1 for all i, j, k
    # Error = sum (sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] - 1)^2
    
    def tensor_error(params, rank, U_shape, V_shape, W_shape):
        """Calculate squared error of the tensor decomposition."""
        # Reshape params into U, V, W
        num_u = U_shape[0] * rank
        num_v = V_shape[0] * rank
        num_w = W_shape[0] * rank
        
        U_flat = params[:num_u]
        V_flat = params[num_u:num_u+num_v]
        W_flat = params[num_u+num_v:num_u+num_v+num_w]
        
        U = U_flat.reshape((U_shape[0], rank))
        V = V_flat.reshape((V_shape[0], rank))
        W = W_flat.reshape((W_shape[0], rank))
        
        error = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    # Compute sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                    # Vectorized: dot product of rows
                    val = np.dot(np.dot(U[i*m+j], V[j*p+k]), W[k*n+i])
                    diff = val - 1.0
                    error += diff * diff
        return error

    def round_and_verify(U, V, W, rank):
        """Round to half-integers and verify exactness."""
        U_r = np.round(U * 2) / 2.0
        V_r = np.round(V * 2) / 2.0
        W_r = np.round(W * 2) / 2.0
        
        # Check if rounding changed values significantly
        # If not, check exactness
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    total = 0.0
                    for c in range(rank):
                        total += U_r[i*m+j, c] * V_r[j*p+k, c] * W_r[k*n+i, c]
                    if abs(total - 1.0) > 1e-9:
                        return None
        return {
            "U": U_r.tolist(),
            "V": V_r.tolist(),
            "W": W_r.tolist()
        }

    # Try ranks from lower bound to schoolbook
    # Lower bound is 15. We try 20, 21, 22, 23, 24, 25.
    # We prioritize lower ranks.
    
    U_shape = (n * m, 1)
    V_shape = (m * p, 1)
    W_shape = (p * n, 1)
    
    for target_rank in [20, 21, 22, 23, 24, 25, 26, 27, 28, 29]:
        if time.time() > deadline:
            break
        
        if target_rank >= best_rank:
            continue
            
        np.random.seed(42 + target_rank)
        
        # Try a few random initializations
        for attempt in range(5):
            if time.time() > deadline:
                break
                
            # Initialize with random values
            U_init = np.random.randn(n * m, target_rank) * 0.5
            V_init = np.random.randn(m * p, target_rank) * 0.5
            W_init = np.random.randn(p * n, target_rank) * 0.5
            
            # Flatten
            params0 = np.concatenate([U_init.flatten(), V_init.flatten(), W_init.flatten()])
            
            # Define objective function for optimizer
            def objective(params):
                return tensor_error(params, target_rank, U_shape, V_shape, W_shape)
            
            try:
                # Use L-BFGS-B for efficiency
                res = minimize(objective, params0, method='L-BFGS-B', options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-12})
                
                if res.success or res.fun < 1e-10:
                    # Extract U, V, W from result
                    num_u = n * m * target_rank
                    num_v = m * p * target_rank
                    num_w = p * n * target_rank
                    
                    U_opt = res.x[:num_u].reshape((n * m, target_rank))
                    V_opt = res.x[num_u:num_u+num_v].reshape((m * p, target_rank))
                    W_opt = res.x[num_u+num_v:num_u+num_v+num_w].reshape((p * n, target_rank))
                    
                    # Round and verify
                    verified = round_and_verify(U_opt, V_opt, W_opt, target_rank)
                    if verified:
                        best_rank = target_rank
                        best_result = verified
                        break
            except Exception as e:
                pass
                
        if best_rank == target_rank:
            break
            
    return best_result
# EVOLVE-BLOCK-END
