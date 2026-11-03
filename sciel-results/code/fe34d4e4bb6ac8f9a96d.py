"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import itertools
import numpy as np
from scipy.optimize import minimize

DIMENSIONS = [3, 3, 7]

def _build_tensor(n, m, p):
    """Build the matrix multiplication tensor T[a,b,c] where:
    a = i*m + j, b = j'*p + k, c = k'*n + i'
    T[a,b,c] = 1 if j==j' and i==i' and k==k', else 0
    """
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1.0
    return T

def _verify_decomposition(U, V, W, T, tol=1e-6):
    """Verify that sum_r U[a,r]*V[b,r]*W[c,r] == T[a,b,c] for all a,b,c."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    rank = U.shape[1]
    # Compute the tensor decomposition
    # For each r, outer product of U[:,r], V[:,r], W[:,r]
    # Sum over r
    reconstructed = np.zeros_like(T)
    for r in range(rank):
        reconstructed += np.einsum('a,b,c->abc', U[:, r], V[:, r], W[:, r])
    max_err = np.max(np.abs(reconstructed - T))
    return max_err < tol

def _try_random_search(n, m, p, T, target_rank, n_trials=100, deadline=None):
    """Try random initializations and local search to find a rank-r decomposition."""
    best_result = None
    for trial in range(n_trials):
        if deadline and time.time() > deadline:
            break
        # Random initialization with integer/half-integer values
        scale = 2.0
        U = np.random.choice([-1, 0, 1], size=(n*m, target_rank)) * scale
        V = np.random.choice([-1, 0, 1], size=(m*p, target_rank)) * scale
        W = np.random.choice([-1, 0, 1], size=(p*n, target_rank)) * scale
        
        # Define the objective: sum of squared errors
        def objective(params):
            U_flat = params[:n*m*target_rank].reshape(n*m, target_rank)
            V_flat = params[n*m*target_rank:2*n*m*target_rank].reshape(m*p, target_rank)
            W_flat = params[2*n*m*target_rank:].reshape(p*n, target_rank)
            reconstructed = np.zeros_like(T)
            for r in range(target_rank):
                reconstructed += np.einsum('a,b,c->abc', U_flat[:, r], V_flat[:, r], W_flat[:, r])
            return np.sum((reconstructed - T) ** 2)
        
        x0 = np.concatenate([U.flatten(), V.flatten(), W.flatten()])
        
        # Try a few optimization steps
        try:
            result = minimize(objective, x0, method='Nelder-Mead', 
                            options={'maxiter': 50, 'xatol': 1e-8, 'fatol': 1e-8})
            if result.fun < 1e-6:
                U_opt = result.x[:n*m*target_rank].reshape(n*m, target_rank)
                V_opt = result.x[n*m*target_rank:2*n*m*target_rank].reshape(m*p, target_rank)
                W_opt = result.x[2*n*m*target_rank:].reshape(p*n, target_rank)
                
                # Round to nearest half-integer
                U_rounded = np.round(U_opt * 2) / 2
                V_rounded = np.round(V_opt * 2) / 2
                W_rounded = np.round(W_opt * 2) / 2
                
                if _verify_decomposition(U_rounded.tolist(), V_rounded.tolist(), W_rounded.tolist(), T):
                    return (U_rounded.tolist(), V_rounded.tolist(), W_rounded.tolist())
        except Exception:
            pass
    return best_result

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = 145.0  # safety margin before 150s kill

    n, m, p = DIMENSIONS
    T = _build_tensor(n, m, p)
    
    # Schoolbook decomposition as fallback
    rank_school = n * m * p
    U = [[0] * rank_school for _ in range(n * m)]
    V = [[0] * rank_school for _ in range(m * p)]
    W = [[0] * rank_school for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    best = {"U": U, "V": V, "W": W, "rank": rank_school}
    
    # Try to find better decompositions with decreasing rank
    # Based on known results: rank(3,3,3)=7, so try ranks from 20 to 62
    # We'll try a few candidate ranks
    
    candidate_ranks = [21, 24, 27, 30, 33, 36, 42, 48, 54, 60]
    
    for target_rank in candidate_ranks:
        if time.time() > deadline - 30:  # Stop if we're running out of time
            break
        if target_rank >= best["rank"]:
            continue
            
        result = _try_random_search(n, m, p, T, target_rank, n_trials=50, deadline=deadline - 30)
        if result is not None:
            U_found, V_found, W_found = result
            best = {"U": U_found, "V": V_found, "W": W_found, "rank": target_rank}
            break  # Found a valid decomposition, return it
    
    return {"U": best["U"], "V": best["V"], "W": best["W"]}
# EVOLVE-BLOCK-END
