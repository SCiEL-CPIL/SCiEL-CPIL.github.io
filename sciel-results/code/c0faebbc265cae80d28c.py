import numpy as np
import time
from scipy.optimize import minimize

DIMENSIONS = [2, 3, 5]

def construct():
    start_time = time.time()
    deadline = start_time + 145.0
    
    n, m, p = DIMENSIONS
    N_U = n * m  # 6
    N_V = m * p  # 15
    N_W = p * n  # 10
    
    # Precompute valid indices and target values
    # Target is 1 for all valid (i, j, k)
    # Map (i, j, k) to (u_idx, v_idx, w_idx)
    u_indices = []
    v_indices = []
    w_indices = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                u_indices.append(u_idx)
                v_indices.append(v_idx)
                w_indices.append(w_idx)
                
    u_indices = np.array(u_indices)
    v_indices = np.array(v_indices)
    w_indices = np.array(w_indices)
    
    # Schoolbook fallback
    def get_schoolbook():
        rank = n * m * p
        U = np.zeros((N_U, rank))
        V = np.zeros((N_V, rank))
        W = np.zeros((N_W, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    best_solution = get_schoolbook()
    best_rank = n * m * p
    
    # Search for lower rank
    # Lower bound heuristic: max(N_U, N_V, N_W) is not a strict lower bound for rank,
    # but rank must be at least the number of non-zero entries in any row/col/slice?
    # Actually, rank >= max(N_U, N_V, N_W) is false for matrix mult.
    # R(2,2,2)=7 < 4? No, 7 > 4.
    # So start from a reasonable number.
    # Let's try from 10 to best_rank - 1.
    
    # To save time, we will try a few ranks.
    # If we find a solution, we return it.
    
    for R in range(10, best_rank):
        if time.time() > deadline:
            break
            
        # Time budget per R
        time_per_R = (deadline - time.time()) / (best_rank - R)
        
        found_solution = None
        max_restarts = 20 # Keep low to ensure speed
        
        for restart in range(max_restarts):
            if time.time() > deadline:
                break
                
            # Random initialization
            # Use a fixed seed for reproducibility in the search, 
            # but vary it per restart
            np.random.seed(42 + R * 100 + restart)
            
            U_init = np.random.randn(N_U, R)
            V_init = np.random.randn(N_V, R)
            W_init = np.random.randn(N_W, R)
            
            x0 = np.concatenate([U_init.flatten(), V_init.flatten(), W_init.flatten()])
            
            def loss(x):
                U = x[:N_U * R].reshape(N_U, R)
                V = x[N_U * R : N_U * R + N_V * R].reshape(N_V, R)
                W = x[N_U * R + N_V * R :].reshape(N_W, R)
                
                # Compute sum for each (i,j,k)
                # U[u_idx, :] * V[v_idx, :] * W[w_idx, :]
                # Sum over R
                terms = U[u_indices] * V[v_indices] * W[w_indices]
                sums = terms.sum(axis=1)
                
                errors = sums - 1.0
                return np.sum(errors ** 2)
            
            try:
                # Use BFGS for small problems
                res = minimize(loss, x0, method='BFGS', options={'maxiter': 50, 'gtol': 1e-8})
            except Exception:
                continue
                
            if res.fun < 1e-6:
                # Potential solution found, try to round
                x_opt = res.x
                U_opt = x_opt[:N_U * R].reshape(N_U, R)
                V_opt = x_opt[N_U * R : N_U * R + N_V * R].reshape(N_V, R)
                W_opt = x_opt[N_U * R + N_V * R :].reshape(N_W, R)
                
                # Round to nearest half-integer
                U_round = np.round(U_opt * 2) / 2
                V_round = np.round(V_opt * 2) / 2
                W_round = np.round(W_opt * 2) / 2
                
                # Verify exactness
                terms_r = U_round[u_indices] * V_round[v_indices] * W_round[w_indices]
                sums_r = terms_r.sum(axis=1)
                
                if np.allclose(sums_r, 1.0, atol=1e-9):
                    found_solution = {
                        "U": U_round.tolist(),
                        "V": V_round.tolist(),
                        "W": W_round.tolist()
                    }
                    best_rank = R
                    break # Break restart loop
        
        if found_solution:
            best_solution = found_solution
            # Since we iterate R increasing, the first valid R is the minimal one found.
            # We can return immediately.
            return best_solution
            
    return best_solution
