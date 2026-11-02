import numpy as np
import time

DIMENSIONS = [3, 3, 5]

def compute_residual(params, n, m, p, rank):
    """Compute the residual tensor for the given parameters."""
    size_u = n * m * rank
    size_v = m * p * rank
    size_w = p * n * rank
    
    U = params[:size_u].reshape((n * m, rank))
    V = params[size_u:size_u + size_v].reshape((m * p, rank))
    W = params[size_u + size_v:].reshape((p * n, rank))
    
    U3 = U.reshape(n, m, rank)
    V3 = V.reshape(m, p, rank)
    W3 = W.reshape(p, n, rank)
    
    # C[i,j,k] = sum_r U[i,j,r] * V[j,k,r] * W[k,i,r]
    # Using einsum for efficiency
    # Temp[i,j,k,r] = U[i,j,r] * V[j,k,r]
    Temp = np.einsum('ijr,jkr->ijk r', U3, V3)
    C = np.einsum('ijk r,kir->ijk', Temp, W3)
    
    # Target: T[i,j,k] = 1 if i==k else 0
    Target = np.zeros((n, p))
    for i in range(n):
        if i < p:
            Target[i, i] = 1.0
    T_target = np.broadcast_to(Target[:, np.newaxis, :], (n, m, p))
    
    return C - T_target

def verify_decomposition(U, V, W, n, m, p, rank, tol=1e-10):
    """Verify if the decomposition is exact."""
    U3 = U.reshape(n, m, rank)
    V3 = V.reshape(m, p, rank)
    W3 = W.reshape(p, n, rank)
    
    Temp = np.einsum('ijr,jkr->ijk r', U3, V3)
    C = np.einsum('ijk r,kir->ijk', Temp, W3)
    
    Target = np.zeros((n, p))
    for i in range(n):
        if i < p:
            Target[i, i] = 1.0
    T_target = np.broadcast_to(Target[:, np.newaxis, :], (n, m, p))
    
    Residual = C - T_target
    return np.max(np.abs(Residual)) < tol

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 145  # Safety margin
    
    # Seed decomposition (Rank 45)
    rank_seed = n * m * p
    U_seed = np.zeros((n * m, rank_seed))
    V_seed = np.zeros((m * p, rank_seed))
    W_seed = np.zeros((p * n, rank_seed))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_seed[i * m + j, col] = 1
                V_seed[j * p + k, col] = 1
                W_seed[k * n + i, col] = 1
                col += 1
                
    best_rank = rank_seed
    best_params = None
    best_snapped = False
    
    # We will try to find lower rank decompositions.
    # The known lower bound for 3x3x5 is likely higher than 23 (3x3x3).
    # Let's try ranks from 44 down to 30.
    
    # To improve success rate, we can use a smarter initialization or optimizer.
    # The current seed uses least_squares which is good but can get stuck.
    # Let's try using a global optimization approach if available, or more restarts.
    
    # Since we have 150s, we can afford many restarts.
    
    for target_rank in range(44, 29, -1):
        if time.time() > deadline:
            break
            
        # Number of restarts depends on rank. Lower rank is harder.
        # Let's do 10 restarts for each rank.
        num_restarts = 10
        
        for restart in range(num_restarts):
            if time.time() > deadline:
                break
                
            size_u = n * m * target_rank
            size_v = m * p * target_rank
            size_w = p * n * target_rank
            total_size = size_u + size_v + size_w
            
            # Random initialization
            # Using small random values helps avoid singularities
            x0 = np.random.uniform(-0.1, 0.1, total_size)
            
            # Define objective for least_squares
            def objective(params):
                return compute_residual(params, n, m, p, target_rank).flatten()
            
            try:
                # Use Levenberg-Marquardt
                # max_nfev can be increased if time permits
                res = least_squares(objective, x0, method='lm', max_nfev=2000)
                
                # Check cost
                # cost = 0.5 * sum(residuals^2)
                if res.cost < 1e-12:
                    U = res.x[:size_u].reshape((n * m, target_rank))
                    V = res.x[size_u:size_u + size_v].reshape((m * p, target_rank))
                    W = res.x[size_u + size_v:].reshape((p * n, target_rank))
                    
                    # Try to snap to integers/half-integers
                    # The problem states "integer/half-integer entries"
                    U_snap = np.round(U * 2) / 2
                    V_snap = np.round(V * 2) / 2
                    W_snap = np.round(W * 2) / 2
                    
                    # Verify snapped version
                    if verify_decomposition(U_snap, V_snap, W_snap, n, m, p, target_rank, tol=1e-10):
                        # Found a valid snapped solution!
                        best_rank = target_rank
                        best_params = np.concatenate([U_snap.flatten(), V_snap.flatten(), W_snap.flatten()])
                        best_snapped = True
                        break # Break restart loop, move to lower rank
                    else:
                        # If snapped fails, check if the real solution is valid
                        # and if we can't find snapped, we might still want to keep it
                        # but the prompt requires integer/half-integer.
                        # If we can't snap, this solution is invalid for the contract.
                        # However, sometimes the optimizer finds a solution that is
                        # very close to integers. Let's try rounding to nearest integer.
                        U_int = np.round(U)
                        V_int = np.round(V)
                        W_int = np.round(W)
                        if verify_decomposition(U_int, V_int, W_int, n, m, p, target_rank, tol=1e-10):
                            best_rank = target_rank
                            best_params = np.concatenate([U_int.flatten(), V_int.flatten(), W_int.flatten()])
                            best_snapped = True
                            break
                        else:
                            # If no snapped solution, we discard this rank for now
                            # unless we find no other option.
                            # We continue searching for snapped solutions.
                            pass
                            
            except Exception as e:
                pass # Ignore optimization failures
        
        # If we found a snapped solution for this rank, we can break
        # and try to find an even lower rank.
        if best_snapped:
            # We found a valid solution for target_rank.
            # We should continue to the next lower rank to see if we can do better.
            # But we must ensure we don't exceed time.
            pass

    # If we found a better rank, use it.
    if best_params is not None and best_rank < rank_seed:
        U = best_params[:n*m*best_rank].reshape((n*m, best_rank))
        V = best_params[n*m*best_rank:n*m*best_rank+m*p*best_rank].reshape((m*p, best_rank))
        W = best_params[n*m*best_rank+m*p*best_rank:].reshape((p*n, best_rank))
        
        return {
            "U": U.tolist(),
            "V": V.tolist(),
            "W": W.tolist()
        }
        
    # Fallback to seed
    return {
        "U": U_seed.tolist(),
        "V": V_seed.tolist(),
        "W": W_seed.tolist()
    }

# Import needed at top level for least_squares
from scipy.optimize import least_squares
