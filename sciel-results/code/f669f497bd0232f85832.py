import numpy as np
import time
from scipy.optimize import least_squares

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
    C = np.einsum('ijr,jkr,kir->ijk', U3, V3, W3, optimize=True)
    
    # Target: T[i,j,k] = 1 if i==k else 0
    # Note: The problem statement says "output factor follows the official trace convention W[k*n+i,r]"
    # This implies W is indexed by (k, i).
    # The target tensor for matrix multiplication C = A * B is T[i, j, k] = delta_{ik}.
    # Here i is row of A (0..n-1), j is col of A / row of B (0..m-1), k is col of B (0..p-1).
    # So T[i, j, k] = 1 if i == k, else 0.
    
    # Construct target efficiently
    # T[i, j, k] = 1 if i == k
    # i in 0..n-1, k in 0..p-1.
    # If i == k, then i must be < p and k must be < n.
    # Since n=3, p=5, i in 0..2, k in 0..4.
    # Matches: (0,0), (1,1), (2,2).
    
    # Create a 3D array of zeros
    T_target = np.zeros((n, m, p))
    for i in range(n):
        if i < p:
            T_target[i, :, i] = 1.0
            
    return C - T_target

def verify_decomposition(U, V, W, n, m, p, rank, tol=1e-10):
    """Verify if the decomposition is exact."""
    U3 = U.reshape(n, m, rank)
    V3 = V.reshape(m, p, rank)
    W3 = W.reshape(p, n, rank)
    
    C = np.einsum('ijr,jkr,kir->ijk', U3, V3, W3, optimize=True)
    
    T_target = np.zeros((n, m, p))
    for i in range(n):
        if i < p:
            T_target[i, :, i] = 1.0
            
    Residual = C - T_target
    return np.max(np.abs(Residual)) < tol

def polish_solution(U, V, W, n, m, p, rank, start_time, deadline):
    """High-precision polish pass: refine a near-exact solution to exact."""
    if time.time() > deadline:
        return None
    
    # Flatten to a single parameter vector for least_squares
    size_u = n * m * rank
    size_v = m * p * rank
    size_w = p * n * rank
    x0 = np.concatenate([U.flatten(), V.flatten(), W.flatten()])
    
    def objective(params):
        return compute_residual(params, n, m, p, rank).flatten()
    
    try:
        # Run with tight tolerance and more iterations
        # Use 'lm' for speed, but if it fails, try 'trf'
        res = least_squares(objective, x0, method='lm', max_nfev=20000, ftol=1e-15, xtol=1e-15, gtol=1e-15)
        
        if res.cost < 1e-14:
            U_p = res.x[:size_u].reshape((n * m, rank))
            V_p = res.x[size_u:size_u + size_v].reshape((m * p, rank))
            W_p = res.x[size_u + size_v:].reshape((p * n, rank))
            
            # Try snapping with multiple strategies
            for strategy in ['half', 'int', 'hybrid']:
                if time.time() > deadline:
                    break
                
                if strategy == 'half':
                    U_snap = np.round(U_p * 2) / 2
                    V_snap = np.round(V_p * 2) / 2
                    W_snap = np.round(W_p * 2) / 2
                elif strategy == 'int':
                    U_snap = np.round(U_p)
                    V_snap = np.round(V_p)
                    W_snap = np.round(W_p)
                else:  # hybrid: round to nearest 0.5, but if very close to integer, use integer
                    U_snap = np.where(np.abs(U_p - np.round(U_p)) < 0.01, np.round(U_p), np.round(U_p * 2) / 2)
                    V_snap = np.where(np.abs(V_p - np.round(V_p)) < 0.01, np.round(V_p), np.round(V_p * 2) / 2)
                    W_snap = np.where(np.abs(W_p - np.round(W_p)) < 0.01, np.round(W_p), np.round(W_p * 2) / 2)
                
                if verify_decomposition(U_snap, V_snap, W_snap, n, m, p, rank, tol=1e-10):
                    return np.concatenate([U_snap.flatten(), V_snap.flatten(), W_snap.flatten()])
            
            # If no snapped version works, check if the raw solution is close enough
            # to be considered valid (shouldn't happen if we require exact, but as fallback)
            if verify_decomposition(U_p, V_p, W_p, n, m, p, rank, tol=1e-10):
                # This shouldn't be returned as it's not integer/half-integer
                # But if the evaluator accepts it, we could return it
                # For now, return None to force us to keep searching
                pass
        
    except Exception:
        pass
    
    return None

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
    # Theoretical bound: Strassen 3x3x3 is 23.
    # 3x3x5 can be seen as 3x3x3 + 3x3x2.
    # Rank(3x3x3) = 23.
    # Rank(3x3x2) is likely 12 or 13.
    # Total rank <= 23 + 13 = 36.
    # Let's search from 44 down to 35.
    
    # To improve success rate, use better initialization.
    # Instead of pure random, we can try to initialize with a scaled seed or structured noise.
    # However, random restarts with LM are standard.
    
    # Increase restarts and time per restart if possible.
    # Total time ~ 145s.
    # Ranks: 44, 43, 42, 41, 40, 39, 38, 37, 36, 35. (10 ranks)
    # If we spend 10s per rank, that's 100s.
    
    for target_rank in range(44, 34, -1):
        if time.time() > deadline:
            break
            
        # Number of restarts: more for higher ranks? Or more for lower?
        # Lower ranks are harder to find.
        # Let's do 5 restarts per rank, but with more iterations each.
        num_restarts = 5
        
        for restart in range(num_restarts):
            if time.time() > deadline:
                break
                
            size_u = n * m * target_rank
            size_v = m * p * target_rank
            size_w = p * n * target_rank
            total_size = size_u + size_v + size_w
            
            # Random initialization
            # Using small random values helps avoid singularities
            # But maybe we need larger values to escape 0?
            # Let's try uniform(-1, 1)
            x0 = np.random.uniform(-1.0, 1.0, total_size)
            
            # Define objective for least_squares
            def objective(params):
                return compute_residual(params, n, m, p, target_rank).flatten()
            
            try:
                # Use Levenberg-Marquardt
                # Increase max_nfev to allow convergence
                res = least_squares(objective, x0, method='lm', max_nfev=5000)
                
                # Check cost
                if res.cost < 1e-10:
                    U = res.x[:size_u].reshape((n * m, target_rank))
                    V = res.x[size_u:size_u + size_v].reshape((m * p, target_rank))
                    W = res.x[size_u + size_v:].reshape((p * n, target_rank))
                    
                    # High-precision polish pass
                    polished = polish_solution(U, V, W, n, m, p, target_rank, time.time(), deadline)
                    if polished is not None:
                        best_rank = target_rank
                        best_params = polished
                        best_snapped = True
                        break # Break restart loop, move to lower rank
                    else:
                        # Fallback: try direct snapping without polish
                        U_snap = np.round(U * 2) / 2
                        V_snap = np.round(V * 2) / 2
                        W_snap = np.round(W * 2) / 2
                        if verify_decomposition(U_snap, V_snap, W_snap, n, m, p, target_rank, tol=1e-10):
                            best_rank = target_rank
                            best_params = np.concatenate([U_snap.flatten(), V_snap.flatten(), W_snap.flatten()])
                            best_snapped = True
                            break
                        else:
                            U_int = np.round(U)
                            V_int = np.round(V)
                            W_int = np.round(W)
                            if verify_decomposition(U_int, V_int, W_int, n, m, p, target_rank, tol=1e-10):
                                best_rank = target_rank
                                best_params = np.concatenate([U_int.flatten(), V_int.flatten(), W_int.flatten()])
                                best_snapped = True
                                break
                            
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
