"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 7]

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import least_squares

def construct():
    n, m, p = DIMENSIONS
    # Try to find a rank decomposition lower than n*m*p
    # Start with schoolbook rank and try to reduce
    schoolbook_rank = n * m * p
    
    # Build the target tensor
    T = np.zeros((n * m, m * p, p * n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m + j, j * p + k, k * n + i] = 1
    
    # Try ranks from schoolbook_rank-1 down to see if we can find a decomposition
    # Use a heuristic: try to find a rank-R decomposition via optimization
    
    best_rank = schoolbook_rank
    best_U = None
    best_V = None
    best_W = None
    
    # Initialize with schoolbook
    U_sb = np.zeros((n * m, schoolbook_rank))
    V_sb = np.zeros((m * p, schoolbook_rank))
    W_sb = np.zeros((p * n, schoolbook_rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j, col] = 1
                V_sb[j * p + k, col] = 1
                W_sb[k * n + i, col] = 1
                col += 1
    
    # Try to reduce rank by 1, 2, ... using optimization
    for target_rank in range(schoolbook_rank - 1, max(schoolbook_rank - 5, 0), -1):
        # Initialize with random perturbations of the first target_rank schoolbook terms
        np.random.seed(42)
        U_init = U_sb[:, :target_rank].copy()
        V_init = V_sb[:, :target_rank].copy()
        W_init = W_sb[:, :target_rank].copy()
        
        # Add small random perturbations
        U_init += np.random.randn(n * m, target_rank) * 0.1
        V_init += np.random.randn(m * p, target_rank) * 0.1
        W_init += np.random.randn(p * n, target_rank) * 0.1
        
        def residual(params):
            U_r = params[:n * m * target_rank].reshape(n * m, target_rank)
            V_r = params[n * m * target_rank:n * m * target_rank + m * p * target_rank].reshape(m * p, target_rank)
            W_r = params[n * m * target_rank + m * p * target_rank:].reshape(p * n, target_rank)
            
            # Compute sum_r U_r[:,r] ⊗ V_r[:,r] ⊗ W_r[:,r]
            T_recon = np.zeros_like(T)
            for r in range(target_rank):
                T_recon += np.einsum('a,b,c->abc', U_r[:, r], V_r[:, r], W_r[:, r])
            
            return (T_recon - T).flatten()
        
        x0 = np.concatenate([U_init.flatten(), V_init.flatten(), W_init.flatten()])
        result = least_squares(residual, x0, method='lm', max_nfev=10000)
        
        if result.cost < 1e-6:
            U_r = result.x[:n * m * target_rank].reshape(n * m, target_rank)
            V_r = result.x[n * m * target_rank:n * m * target_rank + m * p * target_rank].reshape(m * p, target_rank)
            W_r = result.x[n * m * target_rank + m * p * target_rank:].reshape(p * n, target_rank)
            
            # Check if we can round to integers/half-integers
            U_round = np.round(U_r * 2) / 2
            V_round = np.round(V_r * 2) / 2
            W_round = np.round(W_r * 2) / 2
            
            # Verify the rounded version
            T_check = np.zeros_like(T)
            for r in range(target_rank):
                T_check += np.einsum('a,b,c->abc', U_round[:, r], V_round[:, r], W_round[:, r])
            
            if np.allclose(T_check, T, atol=1e-6):
                best_rank = target_rank
                best_U = U_round
                best_V = V_round
                best_W = W_round
                break
    
    if best_U is not None:
        # Convert to lists
        U_list = best_U.tolist()
        V_list = best_V.tolist()
        W_list = best_W.tolist()
        return {"U": U_list, "V": V_list, "W": W_list}
    else:
        # Fall back to schoolbook
        U_list = U_sb.tolist()
        V_list = V_sb.tolist()
        W_list = W_sb.tolist()
        return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
