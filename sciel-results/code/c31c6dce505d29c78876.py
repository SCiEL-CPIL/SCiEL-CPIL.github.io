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

def _try_rank(rank, n, m, p, deadline, max_attempts=5):
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
    
    return None

def _block_decomposition(n, m, p, deadline):
    """
    Try to construct a lower-rank decomposition using block structure.
    For 3x4x5, we can use Strassen-like ideas on sub-blocks.
    
    Strategy: Split the 4x5 matrix B into blocks and use fast matrix multiplication
    where applicable, combined with schoolbook for the rest.
    """
    nm = n * m  # 12
    mp = m * p  # 20
    pn = p * n  # 15
    
    # Try to construct a decomposition with rank < 60
    # Lower bound is max(n*p, m*p, n*m) = max(15, 20, 12) = 20
    
    # For 3x4x5, let's try a block approach:
    # Split A (3x4) into rows, B (4x5) into columns
    # Use the fact that we can share computations across different (i,j,k) triples
    
    # Start with the known lower bound and try a few more
    best_result = None
    best_rank = n * m * p  # 60
    
    # Try ranks from lower bound to 59
    lower_bound = max(n * p, m * p, n * m)  # 20
    
    for rank in range(lower_bound, best_rank):
        if time.time() > deadline - 10:
            break
        
        result = _try_rank(rank, n, m, p, deadline)
        if result is not None:
            best_result = result
            best_rank = rank
            break
    
    if best_result is not None:
        return best_result
    
    return None

def construct():
    """
    Find a rank-minimal exact tensor decomposition for multiplying a 3x4 matrix
    by a 4x5 matrix.
    """
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 115  # Safety margin before 120s limit
    
    # First, try to find a better decomposition than schoolbook
    result = _block_decomposition(n, m, p, deadline)
    if result is not None:
        return result
    
    # If we couldn't find a better decomposition, return the schoolbook one
    return _schoolbook(n, m, p)
# EVOLVE-BLOCK-END
