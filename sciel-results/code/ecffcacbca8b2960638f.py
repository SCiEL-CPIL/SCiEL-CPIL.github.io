import numpy as np
import time
from scipy.optimize import minimize, differential_evolution

DIMENSIONS = [2, 3, 3]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    target_rank = 14
    # Try to find a decomposition of rank 14
    # If not found, fallback to 18
    
    start_time = time.time()
    deadline = 110.0  # 10 second margin
    
    # Target tensor T[i, j, k] = 1
    # Decomposition: T[i, j, k] = sum_c U[i, c] * V[j, c] * W[k, c]
    
    def compute_error(params, rank):
        # params: flat array of length rank * (n + m + p)
        # Order: U (n*rank), V (m*rank), W (p*rank)
        U = params[:n*rank].reshape(n, rank)
        V = params[n*rank:n*rank + m*rank].reshape(m, rank)
        W = params[n*rank + m*rank:].reshape(p, rank)
        
        # Compute tensor product
        # T[i, j, k] = sum_c U[i, c] V[j, c] W[k, c]
        # We can do this efficiently:
        # UV[i, j, c] = U[i, c] * V[j, c]
        # T[i, j, k] = sum_c UV[i, j, c] * W[k, c]
        
        # Reshape for broadcasting
        # U: (n, 1, rank)
        # V: (1, m, rank)
        # UV: (n, m, rank)
        UV = U[:, np.newaxis, :] * V[np.newaxis, :, :]
        
        # W: (1, 1, p, rank) -> (1, 1, p, rank)
        # T: (n, m, p)
        T = np.sum(UV[:, :, np.newaxis, :] * W[np.newaxis, np.newaxis, :, :], axis=3)
        
        # Error: sum of squares of (T - 1)
        return np.sum((T - 1.0)**2)

    def objective(params):
        return compute_error(params, target_rank)

    # Random initialization
    np.random.seed(42)
    
    # Try a few random restarts
    best_score = float('inf')
    best_params = None
    
    # Bounds for differential evolution: [-2, 2] is usually enough for small integer solutions
    bounds = [(-2.0, 2.0)] * (target_rank * (n + m + p))
    
    # Use differential evolution for global search
    # This might be slow, so we limit the time
    
    # Alternative: Simple random search with local refinement
    # Given the small dimension, we can try to solve it exactly if we find a real solution close to integers.
    
    # Let's try a few random initializations and use Nelder-Mead
    for i in range(20):
        if time.time() > deadline:
            break
            
        x0 = np.random.uniform(-1.5, 1.5, target_rank * (n + m + p))
        
        try:
            res = minimize(objective, x0, method='Nelder-Mead', options={'maxiter': 500, 'xatol': 1e-6, 'fatol': 1e-6})
            if res.fun < best_score:
                best_score = res.fun
                best_params = res.x
        except Exception:
            pass
        
        if best_score < 1e-6:
            break

    if best_params is not None and best_score < 1e-4:
        # Round to nearest 0.5
        U = best_params[:n*target_rank].reshape(n, target_rank)
        V = best_params[n*target_rank:n*target_rank + m*target_rank].reshape(m, target_rank)
        W = best_params[n*target_rank + m*target_rank:].reshape(p, target_rank)
        
        U_int = np.round(U * 2) / 2
        V_int = np.round(V * 2) / 2
        W_int = np.round(W * 2) / 2
        
        # Verify exactness
        UV = U_int[:, np.newaxis, :] * V_int[np.newaxis, :, :]
        T = np.sum(UV[:, :, np.newaxis, :] * W_int[np.newaxis, np.newaxis, :, :], axis=3)
        
        if np.allclose(T, 1.0, atol=1e-9):
            # Convert to lists
            U_list = U_int.tolist()
            V_list = V_int.tolist()
            W_list = W_int.tolist()
            return {"U": U_list, "V": V_list, "W": W_list}

    # Fallback: Schoolbook rank 18
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
# EVOLVE-BLOCK-END
