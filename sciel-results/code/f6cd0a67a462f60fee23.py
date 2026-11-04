"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from scipy.optimize import minimize

DIMENSIONS = [3, 4, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # safety margin
    
    # Trivial decomposition as fallback
    rank_trivial = n * m * p
    U_trivial = [[0] * rank_trivial for _ in range(n * m)]
    V_trivial = [[0] * rank_trivial for _ in range(m * p)]
    W_trivial = [[0] * rank_trivial for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_trivial[i * m + j][col] = 1
                V_trivial[j * p + k][col] = 1
                W_trivial[k * n + i][col] = 1
                col += 1
    
    best_result = {"U": U_trivial, "V": V_trivial, "W": W_trivial}
    best_rank = rank_trivial
    
    # Try to find lower rank decomposition
    # The tensor has 84 entries, each must equal 1
    # We need: sum_t U[i*m+j, t] * V[j*p+k, t] * W[k*n+i, t] = 1 for all i,j,k
    
    def check_decomposition(U, V, W, rank):
        """Check if the decomposition is exact."""
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = sum(U[i*m+j][t] * V[j*p+k][t] * W[k*n+i][t] for t in range(rank))
                    if abs(val - 1.0) > 1e-6:
                        return False
        return True
    
    # Try ranks from a lower bound upward
    # Lower bound: max(n*m, m*p, n*p) = max(12, 28, 21) = 28
    # But we can try to find something better
    
    # Use a randomized approach with local optimization
    for target_rank in range(28, min(84, 60)):
        if time.time() > deadline - 10:
            break
            
        # Try multiple random initializations
        for attempt in range(3):
            if time.time() > deadline - 5:
                break
                
            # Random initialization
            np.random.seed(target_rank * 100 + attempt)
            U_init = np.random.randn(n*m, target_rank) * 0.1
            V_init = np.random.randn(m*p, target_rank) * 0.1
            W_init = np.random.randn(p*n, target_rank) * 0.1
            
            # Flatten
            x0 = np.concatenate([U_init.flatten(), V_init.flatten(), W_init.flatten()])
            
            def objective(x):
                U_x = x[:n*m*target_rank].reshape(n*m, target_rank)
                V_x = x[n*m*target_rank : n*m*target_rank + m*p*target_rank].reshape(m*p, target_rank)
                W_x = x[n*m*target_rank + m*p*target_rank :].reshape(p*n, target_rank)
                
                error = 0.0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            val = sum(U_x[i*m+j, t] * V_x[j*p+k, t] * W_x[k*n+i, t] for t in range(target_rank))
                            error += (val - 1.0) ** 2
                return error
            
            # Try to minimize
            try:
                result = minimize(objective, x0, method='L-BFGS-B', 
                                 options={'maxiter': 50, 'ftol': 1e-10})
                
                if result.fun < 1e-4:
                    # Good solution found, verify
                    U_sol = result.x[:n*m*target_rank].reshape(n*m, target_rank).tolist()
                    V_sol = result.x[n*m*target_rank : n*m*target_rank + m*p*target_rank].reshape(m*p, target_rank).tolist()
                    W_sol = result.x[n*m*target_rank + m*p*target_rank :].reshape(p*n, target_rank).tolist()
                    
                    if check_decomposition(U_sol, V_sol, W_sol, target_rank):
                        best_result = {"U": U_sol, "V": V_sol, "W": W_sol}
                        best_rank = target_rank
                        break  # Found good rank, try to go lower
            except Exception:
                pass
    
    return best_result
# EVOLVE-BLOCK-END
