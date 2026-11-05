import numpy as np
import time
import random
from itertools import product

N, M, P = 3, 5, 7
TENSOR_DIM = N * M * P  # 105

def _verify(U, V, W, n=N, m=M, p=P):
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    if U.ndim != 2 or V.ndim != 2 or W.ndim != 2:
        return False
    rank = U.shape[1]
    if U.shape[0] != n * m or V.shape[0] != m * p or W.shape[0] != p * n:
        return False
    if V.shape[1] != rank or W.shape[1] != rank:
        return False
    # Check exact decomposition
    for i in range(n):
        for j in range(m):
            for k in range(p):
                recon = U[i * m + j] @ (V[j * p + k] * W[k * n + i])
                if abs(recon - 1.0) > 1e-7:
                    return False
    return True

def _schoolbook(n=N, m=M, p=P):
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

def _optimize_decomposition(n, m, p, initial_rank, time_limit=60, seed=42):
    """
    Use Levenberg-Marquardt to optimize U, V, W for a given rank.
    We fix the rank and try to find a solution.
    """
    rng = np.random.RandomState(seed)
    
    # Initialize
    # Scale initialization to help convergence
    U = rng.randn(n * m, initial_rank) * 0.1
    V = rng.randn(m * p, initial_rank) * 0.1
    W = rng.randn(p * n, initial_rank) * 0.1
    
    # Precompute indices for efficiency
    idx_i = np.arange(n)
    idx_j = np.arange(m)
    idx_k = np.arange(p)
    
    # We want to minimize sum_{i,j,k} (sum_r U[i,j,r] V[j,k,r] W[k,i,r] - 1)^2
    # This is a non-linear least squares problem.
    # We can use scipy.optimize.least_squares or implement a simple gradient step.
    # Given the 120s limit, we need to be careful.
    
    # Let's use a simple custom gradient descent with adaptive step size,
    # as scipy might be slow or unavailable in some minimal environments (though prompt says SciPy is available).
    # Let's try scipy first if available.
    
    try:
        from scipy.optimize import least_squares
    except ImportError:
        least_squares = None
    
    def residuals(x):
        # x is flattened U, V, W
        U_flat = x[:n*m*initial_rank].reshape(n*m, initial_rank)
        V_flat = x[n*m*initial_rank:n*m*initial_rank + m*p*initial_rank].reshape(m*p, initial_rank)
        W_flat = x[n*m*initial_rank + m*p*initial_rank:].reshape(p*n, initial_rank)
        
        res = np.zeros(n*m*p)
        idx = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    # sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                    val = np.dot(U_flat[i*m+j], V_flat[j*p+k] * W_flat[k*n+i])
                    res[idx] = val - 1.0
                    idx += 1
        return res
    
    x0 = np.concatenate([U.flatten(), V.flatten(), W.flatten()])
    
    if least_squares is not None:
        # Use scipy least squares
        # We need to be careful with the number of iterations and time
        start_time = time.time()
        try:
            # max_nfev is number of function evaluations
            # We have ~120s, let's give it 60s
            res = least_squares(residuals, x0, method='lm', max_nfev=1000, xtol=1e-10, ftol=1e-10, gtol=1e-10)
        except Exception:
            return None
            
        if time.time() - start_time > time_limit:
            return None
            
        U_opt = res.x[:n*m*initial_rank].reshape(n*m, initial_rank)
        V_opt = res.x[n*m*initial_rank:n*m*initial_rank + m*p*initial_rank].reshape(m*p, initial_rank)
        W_opt = res.x[n*m*initial_rank + m*p*initial_rank:].reshape(p*n, initial_rank)
        
        # Check if solution is valid
        if _verify(U_opt.tolist(), V_opt.tolist(), W_opt.tolist(), n, m, p):
            return {"U": U_opt, "V": V_opt, "W": W_opt}
        else:
            return None
    else:
        # Fallback: Simple Gradient Descent (likely too slow for this complexity, but try)
        # This is a backup in case scipy is not available.
        lr = 0.01
        for step in range(1000):
            if time.time() - time.time() > time_limit: # Placeholder, need start time
                break
            # This part is complex to implement correctly without backprop. 
            # Since scipy is stated as available, we assume the above path is taken.
            pass
        return None

def _prune_columns(U, V, W, threshold=1e-6):
    """
    Remove columns that are effectively zero in all three matrices.
    """
    U = np.array(U)
    V = np.array(V)
    W = np.array(W)
    
    rank = U.shape[1]
    keep = []
    for r in range(rank):
        # Check if column r is zero in all matrices
        if (np.all(np.abs(U[:, r]) < threshold) and 
            np.all(np.abs(V[:, r]) < threshold) and 
            np.all(np.abs(W[:, r]) < threshold)):
            continue
        else:
            keep.append(r)
            
    if not keep:
        return U, V, W
        
    U_new = U[:, keep]
    V_new = V[:, keep]
    W_new = W[:, keep]
    
    return U_new, V_new, W_new

def construct():
    n, m, p = N, M, P
    deadline = time.time() + 115  # Safety margin
    
    # Start with schoolbook as the guaranteed fallback
    best = _schoolbook(n, m, p)
    best_rank = n * m * p
    
    # Strategy: Try to find a lower rank decomposition using optimization.
    # The theoretical lower bound is max(n*m, m*p, p*n) = max(15, 35, 21) = 35.
    # Known upper bounds for 3x5x7 might be lower than 105.
    # Let's try ranks from a reasonable estimate down.
    
    # Heuristic initial rank guess:
    # Schoolbook is 105.
    # Block decomposition with 2x2x2 Strassen blocks:
    # 3x5x7 can be split into:
    # (2x2x2) + (1x2x2) + (2x2x1) + ... 
    # Actually, let's just try a few ranks.
    
    # Try ranks: 50, 45, 40, 35
    # Optimization is expensive, so we limit the number of trials.
    
    target_ranks = [60, 50, 45, 40, 35]
    
    for target_rank in target_ranks:
        if time.time() > deadline - 20:
            break
            
        # Try a few seeds for this rank
        for seed in [1, 2, 3]:
            if time.time() > deadline - 10:
                break
                
            print(f"Trying rank {target_rank} with seed {seed}...")
            result = _optimize_decomposition(n, m, p, target_rank, time_limit=30, seed=seed)
            
            if result is not None:
                U_opt, V_opt, W_opt = result["U"], result["V"], result["W"]
                
                # Prune zero columns to potentially reduce rank further
                U_pruned, V_pruned, W_pruned = _prune_columns(U_opt, V_opt, W_opt)
                
                # Verify the pruned version
                if _verify(U_pruned.tolist(), V_pruned.tolist(), W_pruned.tolist(), n, m, p):
                    current_rank = U_pruned.shape[1]
                    if current_rank < best_rank:
                        best_rank = current_rank
                        best = {
                            "U": U_pruned.tolist(),
                            "V": V_pruned.tolist(),
                            "W": W_pruned.tolist()
                        }
                        print(f"Found valid decomposition with rank {current_rank}")
                        # If we found one, we can try to go lower, but let's stop if we hit a good one
                        # or continue to try for even lower
                        break # Move to next lower target rank
                else:
                    # If pruning broke it, use the unpruned if valid
                    if _verify(U_opt.tolist(), V_opt.tolist(), W_opt.tolist(), n, m, p):
                        current_rank = U_opt.shape[1]
                        if current_rank < best_rank:
                            best_rank = current_rank
                            best = {
                                "U": U_opt.tolist(),
                                "V": V_opt.tolist(),
                                "W": W_opt.tolist()
                            }
                            print(f"Found valid decomposition with rank {current_rank} (unpruned)")
                            break

    # Final verification
    if not _verify(best["U"], best["V"], best["W"], n, m, p):
        # Fallback to schoolbook
        best = _schoolbook(n, m, p)
        best_rank = n * m * p

    return best

if __name__ == "__main__":
    result = construct()
    print(f"Best Rank: {len(result['U'][0])}")
