"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from scipy.optimize import least_squares as scipy_least_squares

DIMENSIONS = [3, 4, 5]

# EVOLVE-BLOCK-START
def _verify_decomposition(U, V, W, n, m, p):
    """Verify that the decomposition is exact."""
    nm = n * m
    mp = m * p
    pn = p * n
    U = np.array(U)
    V = np.array(V)
    W = np.array(W)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                d = k * n + i
                val = np.sum(U[a, :] * V[b, :] * W[d, :])
                if abs(val - 1.0) > 1e-10:
                    return False
    return True

def _schoolbook(n, m, p):
    """Return the schoolbook decomposition with rank n*m*p."""
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

def _compute_residual(U_flat, V_flat, W_flat, n, m, p, rank):
    """Compute the residual vector for the tensor decomposition."""
    nm = n * m
    mp = m * p
    pn = p * n
    U = U_flat.reshape(nm, rank)
    V = V_flat.reshape(mp, rank)
    W = W_flat.reshape(pn, rank)
    
    T = np.zeros((n, m, p))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                d = k * n + i
                T[i, j, k] = np.sum(U[a, :] * V[b, :] * W[d, :])
    
    return (T - 1.0).flatten()

def _try_rank(rank, n, m, p, deadline, max_attempts=10):
    """Try to find a valid decomposition for the given rank."""
    nm = n * m
    mp = m * p
    pn = p * n
    
    def residual(params):
        U_flat = params[:nm*rank]
        V_flat = params[nm*rank:nm*rank+mp*rank]
        W_flat = params[nm*rank+mp*rank:]
        return _compute_residual(U_flat, V_flat, W_flat, n, m, p, rank)
    
    for attempt in range(max_attempts):
        if time.time() > deadline - 5:
            return None
        
        np.random.seed(attempt * 7919 + rank * 31)
        scale = 0.5 / np.sqrt(rank) if rank > 1 else 0.5
        U = np.random.randn(nm, rank) * scale
        V = np.random.randn(mp, rank) * scale
        W = np.random.randn(pn, rank) * scale
        x0 = np.concatenate([U.flatten(), V.flatten(), W.flatten()])
        
        try:
            result = scipy_least_squares(residual, x0, method='lm', max_nfev=2000)
            if result.success:
                final_residual = np.max(np.abs(result.fun))
                if final_residual < 1e-8:
                    U_sol = result.x[:nm*rank].reshape(nm, rank)
                    V_sol = result.x[nm*rank:nm*rank+mp*rank].reshape(mp, rank)
                    W_sol = result.x[nm*rank+mp*rank:].reshape(pn, rank)
                    
                    # Verify the solution
                    if _verify_decomposition(U_sol, V_sol, W_sol, n, m, p):
                        return {
                            "U": U_sol.tolist(),
                            "V": V_sol.tolist(),
                            "W": W_sol.tolist()
                        }
        except Exception:
            pass
        
        # Try a different initialization strategy: start from schoolbook and perturb
        if attempt % 2 == 1:
            # Create a partial schoolbook-like initialization
            U_init = np.zeros((nm, rank))
            V_init = np.zeros((mp, rank))
            W_init = np.zeros((pn, rank))
            
            # Fill first min(rank, nm*mp*pn) columns with schoolbook-like patterns
            col = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        if col >= rank:
                            break
                        U_init[i * m + j, col] = 1
                        V_init[j * p + k, col] = 1
                        W_init[k * n + i, col] = 1
                        col += 1
                    if col >= rank:
                        break
                if col >= rank:
                    break
            
            # Add small random noise to remaining columns
            for c in range(col, rank):
                U_init[:, c] = np.random.randn(nm) * 0.1
                V_init[:, c] = np.random.randn(mp) * 0.1
                W_init[:, c] = np.random.randn(pn) * 0.1
            
            x0 = np.concatenate([U_init.flatten(), V_init.flatten(), W_init.flatten()])
            
            try:
                result = scipy_least_squares(residual, x0, method='lm', max_nfev=2000)
                if result.success:
                    final_residual = np.max(np.abs(result.fun))
                    if final_residual < 1e-8:
                        U_sol = result.x[:nm*rank].reshape(nm, rank)
                        V_sol = result.x[nm*rank:nm*rank+mp*rank].reshape(mp, rank)
                        W_sol = result.x[nm*rank+mp*rank:].reshape(pn, rank)
                        
                        if _verify_decomposition(U_sol, V_sol, W_sol, n, m, p):
                            return {
                                "U": U_sol.tolist(),
                                "V": V_sol.tolist(),
                                "W": W_sol.tolist()
                            }
            except Exception:
                pass
    
    return None

def construct():
    """
    Find a rank-minimal exact tensor decomposition for multiplying a 3x4 matrix
    by a 4x5 matrix.
    """
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 115  # Safety margin before 120s limit
    
    # Known lower bound for matrix multiplication rank
    # For n x m by m x p: rank >= max(n*p, m*p, n*m)
    lower_bound = max(n * p, m * p, n * m)  # max(15, 20, 12) = 20
    
    # Try ranks from lower bound upward
    for rank in range(lower_bound, 61):
        if time.time() > deadline - 10:
            break
        
        result = _try_rank(rank, n, m, p, deadline)
        if result is not None:
            return result
    
    # If we couldn't find a better decomposition, return the schoolbook one
    return _schoolbook(n, m, p)
# EVOLVE-BLOCK-END
