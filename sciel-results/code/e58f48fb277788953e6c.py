import numpy as np
import time
import random
from scipy.optimize import minimize

DIMENSIONS = [3, 3, 4]
START_TIME = time.time()
DEADLINE = 140  # 140 seconds to leave safety margin

def _check_exact(U, V, W):
    """Check if the decomposition is exact and has integer/half-integer entries."""
    n, m, p = DIMENSIONS
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    
    rank = U.shape[1]
    
    # Check for integer or half-integer entries
    for mat in [U, V, W]:
        doubled = mat * 2
        if not np.allclose(doubled, np.round(doubled), atol=1e-9):
            return False
    
    # Verify the tensor decomposition
    # T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # Should equal delta(j) for the multiplication tensor, i.e., 
    # C[i,k] = sum_j A[i,j]*B[j,k], so T[i,j,k] = 1 if this is the (i,j,k) term
    # Actually T[i,j,k] should be 1 for all valid (i,j,k) in the multiplication
    # The multiplication tensor has T[i,j,k] = 1 for all i,j,k (since each product term contributes)
    # Wait no: the tensor for matrix mult is T[i,j,k] where the structure is:
    # C[i,k] = sum_j A[i,j] B[j,k]
    # So the tensor T has indices (i, j, k) and T[i,j,k] = 1 for all valid i,j,k
    # because each (i,j,k) triple corresponds to one multiplication-addition term
    
    # Let's verify: for the decomposition to be correct,
    # sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] should equal 1 for all i,j,k
    # if the tensor is the standard multiplication tensor
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for r in range(rank):
                    s += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                if abs(s - 1.0) > 1e-9:
                    return False
    return True

def _try_rank(rank, n=3, m=3, p=4, max_iters=5000, seed=None):
    """Try to find an exact decomposition of given rank."""
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
    
    nm = n * m
    mp = m * p
    pn = p * n
    
    best_error = float('inf')
    best_U = best_V = best_W = None
    
    for attempt in range(5):
        if time.time() > DEADLINE:
            break
        
        # Initialize with random values
        U = np.random.randn(nm, rank) * 0.5
        V = np.random.randn(mp, rank) * 0.5
        W = np.random.randn(pn, rank) * 0.5
        
        def objective(params):
            U_flat = params[:nm*rank].reshape(nm, rank)
            V_flat = params[nm*rank:nm*rank+mp*rank].reshape(mp, rank)
            W_flat = params[nm*rank+mp*rank:].reshape(pn, rank)
            
            error = 0.0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        s = 0.0
                        for r in range(rank):
                            s += U_flat[i*m+j, r] * V_flat[j*p+k, r] * W_flat[k*n+i, r]
                        error += (s - 1.0) ** 2
            return error
        
        params0 = np.concatenate([U.flatten(), V.flatten(), W.flatten()])
        
        try:
            result = minimize(objective, params0, method='L-BFGS-B', 
                            options={'maxiter': max_iters, 'maxfun': 10000})
            if result.success or result.fun < best_error:
                best_error = result.fun
                if best_error < 1e-6:
                    U_opt = result.x[:nm*rank].reshape(nm, rank)
                    V_opt = result.x[nm*rank:nm*rank+mp*rank].reshape(mp, rank)
                    W_opt = result.x[nm*rank+mp*rank:].reshape(pn, rank)
                    
                    # Try to round to half-integers
                    U_round = np.round(U_opt * 2) / 2
                    V_round = np.round(V_opt * 2) / 2
                    W_round = np.round(W_opt * 2) / 2
                    
                    if _check_exact(U_round.tolist(), V_round.tolist(), W_round.tolist()):
                        return U_round.tolist(), V_round.tolist(), W_round.tolist()
                    
                    best_U, best_V, best_W = U_opt, V_opt, W_opt
        except Exception:
            continue
    
    return None

def construct():
    n, m, p = DIMENSIONS
    nm, mp, pn = n*m, m*p, p*n
    
    # Schoolbook as fallback
    schoolbook_rank = nm * mp
    U_sb = [[0] * schoolbook_rank for _ in range(nm)]
    V_sb = [[0] * schoolbook_rank for _ in range(mp)]
    W_sb = [[0] * schoolbook_rank for _ in range(pn)]
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                r = i * (m * p) + j * p + k
                U_sb[i * m + j][r] = 1
                V_sb[j * p + k][r] = 1
                W_sb[k * n + i][r] = 1
    
    # Try to find lower rank decompositions
    # Known lower bounds suggest rank around 20-25 for 3x3x4
    for rank in range(20, 37):
        if time.time() > DEADLINE - 10:
            break
        result = _try_rank(rank, n, m, p, max_iters=2000, seed=rank)
        if result is not None:
            U, V, W = result
            return {"U": U, "V": V, "W": W}
    
    # Return schoolbook
    return {"U": U_sb, "V": V_sb, "W": W_sb}
