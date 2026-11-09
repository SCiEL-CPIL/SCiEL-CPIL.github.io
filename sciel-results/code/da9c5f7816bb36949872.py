"""Matrix multiplication 4,4,5 - rank-minimal exact tensor decomposition"""
import numpy as np
import time
import itertools
from scipy.optimize import minimize

DIMENSIONS = [4, 4, 5]

# EVOLVE-BLOCK-START
def _build_schoolbook(n, m, p):
    """Build the standard (schoolbook) decomposition."""
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
    return U, V, W, rank

def _decomposition_error(U, V, W, n, m, p):
    """Compute the error of a decomposition against the true tensor."""
    # The tensor T[i,j,k,r] = delta_{i,k} * delta_{j,r} for (i,j) in input, (k,r) in output
    # Actually for matrix mult: C = A * B, C[i,r] = sum_j A[i,j] * B[j,r]
    # Tensor: T[i,j,r,k] where A has indices (i,j), B has (j,r), C has (i,r)
    # Decomposition: T[i,j,r,k] = sum_c U[i*m+j, c] * V[j*p+r, c] * W[r*n+i, c]
    # Wait, let me re-read the convention: W[k*n+i, r]
    
    # Let me think carefully. The tensor for matrix multiplication C = A*B where
    # A is n x m, B is m x p, C is n x p
    # C[i, r] = sum_j A[i,j] * B[j,r]
    # 
    # The bilinear map can be written as:
    # C[i,r] = sum_c U[i,j,c] * V[j,r,c] * ... 
    # 
    # Actually the standard tensor decomposition for matrix mult:
    # T_{i,j,r} (with j being the contracted index) 
    # T[i, j, r] = delta_{i,?} ... no.
    
    # Let me use the convention from the problem:
    # U has shape (n*m, rank), indexed by (i*m + j)
    # V has shape (m*p, rank), indexed by (j*p + r)  
    # W has shape (p*n, rank), indexed by (r*n + i)  [since W[k*n+i] with k=r]
    
    # The tensor T[i,j,r] for matrix mult satisfies:
    # C[i,r] = sum_j A[i,j] B[j,r]
    # So the tensor is T[i,j,r] = 1 if we're computing the (i,r) entry using (i,j) and (j,r)
    # More precisely, the structure tensor has T[i,j,r] = 1 when the indices are consistent
    # Actually T[i,j,r] = delta_{i,?}... 
    
    # The matrix multiplication tensor is:
    # T[i, j, r] = 1 for all valid (i,j,r) where j is shared
    # But that's not right either. Let me think again.
    
    # For C = A*B: C[i,r] = sum_j A[i,j] * B[j,r]
    # The tensor T has indices (i, j, r) and T[i,j,r] = 1 (it's just the identity in the contracted dimension)
    # No wait, T[i,j,r] should encode that A[i,j] and B[j,r] contribute to C[i,r]
    # So T[i,j,r] = 1 for all i,j,r (it's the complete tensor)
    # And the decomposition is: T[i,j,r] = sum_c U[i*m+j, c] * V[j*p+r, c] * W[r*n+i, c]
    
    # So we need: sum_c U[i*m+j, c] * V[j*p+r, c] * W[r*n+i, c] = 1 for all i, j, r
    
    error = 0.0
    for i in range(n):
        for j in range(m):
            for r in range(p):
                val = np.sum(U[i*m+j, :] * V[j*p+r, :] * W[r*n+i, :])
                error += (val - 1.0) ** 2
    return error

def _try_exact_decomposition(rank, n, m, p, max_iter=5000, tol=1e-10):
    """Try to find an exact decomposition of given rank using optimization."""
    np.random.seed(42)
    
    # Initialize with random values
    U = np.random.randn(n * m, rank)
    V = np.random.randn(m * p, rank)
    W = np.random.randn(p * n, rank)
    
    # Scale to reasonable magnitude
    U /= np.sqrt(rank)
    V /= np.sqrt(rank)
    W /= np.sqrt(rank)
    
    def objective(params):
        U_flat = params[:n*m*rank].reshape(n*m, rank)
        V_flat = params[n*m*rank:n*m*rank + m*p*rank].reshape(m*p, rank)
        W_flat = params[n*m*rank + m*p*rank:].reshape(p*n, rank)
        return _decomposition_error(U_flat, V_flat, W_flat, n, m, p)
    
    x0 = np.concatenate([U.flatten(), V.flatten(), W.flatten()])
    
    try:
        result = minimize(objective, x0, method='L-BFGS-B', 
                         options={'maxiter': max_iter, 'gtol': 1e-12, 'ftol': 1e-15})
        U_opt = result.x[:n*m*rank].reshape(n*m, rank)
        V_opt = result.x[n*m*rank:n*m*rank + m*p*rank].reshape(m*p, rank)
        W_opt = result.x[n*m*rank + m*p*rank:].reshape(p*n, rank)
        
        err = _decomposition_error(U_opt, V_opt, W_opt, n, m, p)
        if err < tol:
            return U_opt, V_opt, W_opt, True
    except Exception:
        pass
    
    return None, None, None, False

def _round_to_halfint(U, V, W):
    """Try to round entries to nearest half-integers and check if still exact."""
    U_r = np.round(U * 2) / 2
    V_r = np.round(V * 2) / 2
    W_r = np.round(W * 2) / 2
    return U_r, V_r, W_r

def _check_exact(U, V, W, n, m, p, tol=1e-9):
    """Check if decomposition is exact (within tolerance)."""
    for i in range(n):
        for j in range(m):
            for r in range(p):
                val = np.sum(U[i*m+j, :] * V[j*p+r, :] * W[r*n+i, :])
                if abs(val - 1.0) > tol:
                    return False
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 115  # Safety margin before 120s limit
    
    n, m, p = DIMENSIONS
    
    # Start with schoolbook
    U, V, W, rank = _build_schoolbook(n, m, p)
    best_U, best_V, best_W, best_rank = U, V, W, rank
    
    # Try to find better decompositions
    # Known results suggest rank can be significantly lower than 80
    # Let's try various ranks from low to high
    
    # First, try some known structured decompositions
    # For 4x4x4, there are decompositions of rank 49 or lower
    # For 4x4x5, we can adapt these
    
    # Try optimization for various ranks
    for target_rank in range(25, 81, 1):
        if time.time() > deadline:
            break
        
        if target_rank >= best_rank:
            break
            
        # Try multiple random initializations
        found = False
        for seed in range(10):
            if time.time() > deadline:
                break
            
            np.random.seed(seed)
            U_init = np.random.randn(n * m, target_rank)
            V_init = np.random.randn(m * p, target_rank)
            W_init = np.random.randn(p * n, target_rank)
            
            # Scale
            U_init /= np.sqrt(target_rank)
            V_init /= np.sqrt(target_rank)
            W_init /= np.sqrt(target_rank)
            
            def objective(params):
                U_f = params[:n*m*target_rank].reshape(n*m, target_rank)
                V_f = params[n*m*target_rank:n*m*target_rank + m*p*target_rank].reshape(m*p, target_rank)
                W_f = params[n*m*target_rank + m*p*target_rank:].reshape(p*n, target_rank)
                return _decomposition_error(U_f, V_f, W_f, n, m, p)
            
            x0 = np.concatenate([U_init.flatten(), V_init.flatten(), W_init.flatten()])
            
            try:
                result = minimize(objective, x0, method='L-BFGS-B',
                                 options={'maxiter': 3000, 'gtol': 1e-10, 'ftol': 1e-14})
                U_opt = result.x[:n*m*target_rank].reshape(n*m, target_rank)
                V_opt = result.x[n*m*target_rank:n*m*target_rank + m*p*target_rank].reshape(m*p, target_rank)
                W_opt = result.x[n*m*target_rank + m*p*target_rank:].reshape(p*n, target_rank)
                
                err = _decomposition_error(U_opt, V_opt, W_opt, n, m, p)
                if err < 1e-10:
                    # Check if we can get half-integer entries
                    U_r, V_r, W_r = _round_to_halfint(U_opt, V_opt, W_opt)
                    if _check_exact(U_r, V_r, W_r, n, m, p):
                        best_U, best_V, best_W, best_rank = U_r, V_r, W_r, target_rank
                        found = True
                        break
                    else:
                        # Use the floating point result but convert to list
                        # The problem says integer/half-integer entries required
                        # If rounding doesn't work, skip this solution
                        pass
            except Exception:
                continue
    
    # Convert to lists
    U_list = [[float(U_list_row[j]) for j in range(best_rank)] for i, U_list_row in enumerate(best_U)]
    V_list = [[float(V_list_row[j]) for j in range(best_rank)] for i, V_list_row in enumerate(best_V)]
    W_list = [[float(W_list_row[j]) for j in range(best_rank)] for i, W_list_row in enumerate(best_W)]
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
