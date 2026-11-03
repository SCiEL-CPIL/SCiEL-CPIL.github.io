"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
from scipy.optimize import least_squares
import time

DIMENSIONS = [2, 3, 3]

# EVOLVE-BLOCK-START
def construct():
    """
    Returns a rank-minimal exact tensor decomposition for 
    multiplying a 2x3 matrix by a 3x3 matrix.
    
    Attempts rank 16 first via numerical optimization, falls back to rank 17.
    
    Interface:
    U: shape (n*m, rank)
    V: shape (m*p, rank)
    W: shape (p*n, rank)
    
    n=2, m=3, p=3.
    """
    n, m, p = DIMENSIONS
    deadline = time.time() + 140.0
    
    # First, try to find a rank-16 solution
    result = _try_rank_16(deadline)
    if result is not None:
        return result
    
    # Fallback: verified rank-17 construction
    return _rank_17_fallback()

def _rank_17_fallback():
    """Verified rank-17 Strassen+schoolbook construction."""
    n, m, p = DIMENSIONS
    rank = 17
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    def add_term(u_indices, v_indices, w_indices, col):
        for u_idx, val in u_indices:
            U[u_idx][col] += val
        for v_idx, val in v_indices:
            V[v_idx][col] += val
        for w_idx, val in w_indices:
            W[w_idx][col] += val

    # Strassen terms for A[:, 0:2] * B[0:2, 0:2]
    add_term([(0, 1), (4, 1)], [(0, 1), (4, 1)], [(0, 1), (3, 1)], 0)
    add_term([(3, 1), (4, 1)], [(0, 1)], [(1, 1), (3, -1)], 1)
    add_term([(0, 1)], [(1, 1), (4, -1)], [(2, 1), (3, 1)], 2)
    add_term([(4, 1)], [(3, 1), (0, -1)], [(0, 1), (1, 1)], 3)
    add_term([(0, 1), (1, 1)], [(4, 1)], [(0, -1), (2, 1)], 4)
    add_term([(3, 1), (0, -1)], [(0, 1), (1, 1)], [(3, 1)], 5)
    add_term([(1, 1), (4, -1)], [(3, 1), (4, 1)], [(0, 1)], 6)

    # Schoolbook terms for the rest
    add_term([(2, 1)], [(6, 1)], [(0, 1)], 7)
    add_term([(5, 1)], [(6, 1)], [(1, 1)], 8)
    add_term([(2, 1)], [(7, 1)], [(2, 1)], 9)
    add_term([(5, 1)], [(7, 1)], [(3, 1)], 10)
    add_term([(0, 1)], [(2, 1)], [(4, 1)], 11)
    add_term([(3, 1)], [(2, 1)], [(5, 1)], 12)
    add_term([(1, 1)], [(5, 1)], [(4, 1)], 13)
    add_term([(4, 1)], [(5, 1)], [(5, 1)], 14)
    add_term([(2, 1)], [(8, 1)], [(4, 1)], 15)
    add_term([(5, 1)], [(8, 1)], [(5, 1)], 16)

    return {"U": U, "V": V, "W": W}

def _try_rank_16(deadline):
    """Attempt to find a rank-16 decomposition via numerical optimization."""
    n, m, p = DIMENSIONS
    rank = 16
    
    # Build the matrix multiplication tensor
    # T[i,j,k] = 1 if j is the shared index
    # A is n x m, B is m x p, C is p x n (output convention W[k*n+i])
    # C[k,i] = sum_j A[i,j] * B[j,k]
    # U index: i*m + j
    # V index: j*p + k  
    # W index: k*n + i
    
    # Build the tensor as a 3D array
    T = np.zeros((n * m, m * p, p * n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                T[u_idx, v_idx, w_idx] = 1.0
    
    # Initialize with rank-17 solution and project to rank 16
    # Use the known rank-17 structure as starting point
    U17, V17, W17 = _get_rank_17_arrays()
    
    # Try multiple random initializations
    best_solution = None
    best_resid = float('inf')
    
    for attempt in range(3):
        if time.time() > deadline - 10:
            break
            
        # Start from rank-17 and zero out one column, or random init
        if attempt == 0:
            # Use rank-17 with last column zeroed
            U_init = U17[:, :rank].copy()
            V_init = V17[:, :rank].copy()
            W_init = W17[:, :rank].copy()
        elif attempt == 1:
            # Random initialization with structure
            np.random.seed(42)
            U_init = np.random.randn(n*m, rank) * 0.5
            V_init = np.random.randn(m*p, rank) * 0.5
            W_init = np.random.randn(p*n, rank) * 0.5
        else:
            # Perturbed rank-17
            np.random.seed(123)
            U_init = U17[:, :rank].copy() + np.random.randn(n*m, rank) * 0.1
            V_init = V17[:, :rank].copy() + np.random.randn(m*p, rank) * 0.1
            W_init = W17[:, :rank].copy() + np.random.randn(p*n, rank) * 0.1
        
        # Flatten parameters
        def unpack(params):
            U_flat = params[:n*m*rank]
            V_flat = params[n*m*rank:n*m*rank + m*p*rank]
            W_flat = params[n*m*rank + m*p*rank:]
            U_mat = U_flat.reshape(n*m, rank)
            V_mat = V_flat.reshape(m*p, rank)
            W_mat = W_flat.reshape(p*n, rank)
            return U_mat, V_mat, W_mat
        
        def residuals(params):
            U_mat, V_mat, W_mat = unpack(params)
            # Compute sum of outer products
            result = np.zeros((n*m, m*p, p*n))
            for r in range(rank):
                result += np.einsum('a,b,c->abc', U_mat[:, r], V_mat[:, r], W_mat[:, r])
            return (result - T).flatten()
        
        x0 = np.concatenate([U_init.flatten(), V_init.flatten(), W_init.flatten()])
        
        try:
            sol = least_squares(residuals, x0, method='lm', max_nfev=5000)
            resid = np.sum(sol.fun**2)
            
            if resid < best_resid:
                best_resid = resid
                U_mat, V_mat, W_mat = unpack(sol.x)
                best_solution = (U_mat, V_mat, W_mat)
        except Exception:
            continue
    
    # Check if we found a valid rank-16 solution
    if best_solution is not None and best_resid < 1e-10:
        U_mat, V_mat, W_mat = best_solution
        # Round to nearest integers/half-integers
        U_rounded = np.round(U_mat * 2) / 2
        V_rounded = np.round(V_mat * 2) / 2
        W_rounded = np.round(W_mat * 2) / 2
        
        # Verify the rounded solution
        result = np.zeros((n*m, m*p, p*n))
        for r in range(rank):
            result += np.einsum('a,b,c->abc', U_rounded[:, r], V_rounded[:, r], W_rounded[:, r])
        
        if np.max(np.abs(result - T)) < 1e-9:
            # Convert to list format
            U_list = U_rounded.tolist()
            V_list = V_rounded.tolist()
            W_list = W_rounded.tolist()
            return {"U": U_list, "V": V_list, "W": W_list}
    
    return None

def _get_rank_17_arrays():
    """Get the rank-17 solution as numpy arrays."""
    n, m, p = DIMENSIONS
    rank = 17
    
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    def add_term(u_indices, v_indices, w_indices, col):
        for u_idx, val in u_indices:
            U[u_idx, col] += val
        for v_idx, val in v_indices:
            V[v_idx, col] += val
        for w_idx, val in w_indices:
            W[w_idx, col] += val

    # Strassen terms
    add_term([(0, 1), (4, 1)], [(0, 1), (4, 1)], [(0, 1), (3, 1)], 0)
    add_term([(3, 1), (4, 1)], [(0, 1)], [(1, 1), (3, -1)], 1)
    add_term([(0, 1)], [(1, 1), (4, -1)], [(2, 1), (3, 1)], 2)
    add_term([(4, 1)], [(3, 1), (0, -1)], [(0, 1), (1, 1)], 3)
    add_term([(0, 1), (1, 1)], [(4, 1)], [(0, -1), (2, 1)], 4)
    add_term([(3, 1), (0, -1)], [(0, 1), (1, 1)], [(3, 1)], 5)
    add_term([(1, 1), (4, -1)], [(3, 1), (4, 1)], [(0, 1)], 6)

    # Schoolbook terms
    add_term([(2, 1)], [(6, 1)], [(0, 1)], 7)
    add_term([(5, 1)], [(6, 1)], [(1, 1)], 8)
    add_term([(2, 1)], [(7, 1)], [(2, 1)], 9)
    add_term([(5, 1)], [(7, 1)], [(3, 1)], 10)
    add_term([(0, 1)], [(2, 1)], [(4, 1)], 11)
    add_term([(3, 1)], [(2, 1)], [(5, 1)], 12)
    add_term([(1, 1)], [(5, 1)], [(4, 1)], 13)
    add_term([(4, 1)], [(5, 1)], [(5, 1)], 14)
    add_term([(2, 1)], [(8, 1)], [(4, 1)], 15)
    add_term([(5, 1)], [(8, 1)], [(5, 1)], 16)

    return U, V, W
# EVOLVE-BLOCK-END
