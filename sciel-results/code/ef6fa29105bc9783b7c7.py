"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [4, 5, 5]

# EVOLVE-BLOCK-START
_START_TIME = time.time()
_DEADLINE = 145.0  # Safety margin for 150s limit

def _time_exceeded():
    return (time.time() - _START_TIME) > _DEADLINE

def _check_decomposition(U, V, W, n, m, p):
    """
    Verifies if U, V, W form a valid decomposition.
    Returns True if exact (within tolerance).
    """
    # T[i, j1, j2, k] should be delta[j1, j2] for all i, k
    # Sum_c W[k*n+i, c] * U[i*m+j1, c] * V[j2*p+k, c] == delta[j1, j2]
    
    U = np.array(U)
    V = np.array(V)
    W = np.array(W)
    
    # Reshape U, V, W to 3D/4D for easier checking?
    # U: (n*m, r) -> (n, m, r)
    # V: (m*p, r) -> (m, p, r)
    # W: (p*n, r) -> (p, n, r)
    
    U_3d = U.reshape(n, m, -1)
    V_3d = V.reshape(m, p, -1)
    W_3d = W.reshape(p, n, -1)
    
    # Check for a few random instances or all?
    # Since n, m, p are small, we can check all.
    # Condition: for all i, k, j1, j2:
    # sum_c W_3d[k, i, c] * U_3d[i, j1, c] * V_3d[j2, k, c] == 1 if j1==j2 else 0
    
    # We can compute the tensor T_calc[i, j1, j2, k]
    # T_calc = sum_c W_3d[k, i, c] U_3d[i, j1, c] V_3d[j2, k, c]
    
    # Using einsum:
    # W_3d[k, i, c] -> W_k_ic
    # U_3d[i, j1, c] -> U_i_j1_c
    # V_3d[j2, k, c] -> V_j2_k_c
    # Result T[i, j1, j2, k] = sum_c W[k, i, c] U[i, j1, c] V[j2, k, c]
    
    # Note: U depends on i, j1, c. V depends on j2, k, c. W depends on k, i, c.
    # The sum is over c.
    
    # Let's construct the target tensor T_target[i, j1, j2, k] = delta[j1, j2]
    
    # Compute T_calc
    # T_calc[i, j1, j2, k] = np.einsum('kic,ij1c,j2kc->ij1j2k', W_3d, U_3d, V_3d)
    
    try:
        T_calc = np.einsum('kic,ij1c,j2kc->ij1j2k', W_3d, U_3d, V_3d)
    except Exception:
        return False
        
    # Target: delta[j1, j2]
    # T_target[i, j1, j2, k] = 1 if j1 == j2 else 0
    # This is independent of i and k.
    
    # Check max absolute difference
    diff = np.zeros((n, m, m, p))
    for i in range(n):
        for j1 in range(m):
            for j2 in range(m):
                for k in range(p):
                    target = 1.0 if j1 == j2 else 0.0
                    diff[i, j1, j2, k] = T_calc[i, j1, j2, k] - target
                    
    if np.max(np.abs(diff)) < 1e-9:
        return True
    return False

def _try_random_search(rank, n, m, p, iterations=50):
    """
    Attempts to find a decomposition of given rank using random U, V and solving for W.
    """
    for _ in range(iterations):
        if _time_exceeded():
            break
            
        # Initialize U and V randomly
        # Using small integers or random floats?
        # Random floats are better for breaking symmetries.
        U = np.random.randn(n * m, rank)
        V = np.random.randn(m * p, rank)
        
        # Solve for W
        # Equations: for all i, j1, j2, k:
        # sum_c W[k*n+i, c] U[i*m+j1, c] V[j2*p+k, c] = delta[j1, j2]
        
        # This is a linear system A W_flat = b
        # Number of equations: n * m * m * p
        # Number of unknowns: p * n * rank
        
        n_eq = n * m * m * p
        n_unk = p * n * rank
        
        A = np.zeros((n_eq, n_unk))
        b = np.zeros(n_eq)
        
        eq_idx = 0
        for i in range(n):
            for j1 in range(m):
                for j2 in range(m):
                    for k in range(p):
                        target = 1.0 if j1 == j2 else 0.0
                        b[eq_idx] = target
                        
                        # Coefficients for W[k*n+i, c]
                        # Term is W[k*n+i, c] * U[i*m+j1, c] * V[j2*p+k, c]
                        # So A[eq_idx, (k*n+i)*rank + c] = U[i*m+j1, c] * V[j2*p+k, c]
                        
                        # We can vectorize this row construction
                        # For this eq, the non-zero coeffs are at indices (k*n+i)*rank + c
                        # Value is U[i*m+j1, c] * V[j2*p+k, c]
                        
                        # Let's just fill it directly in the loop for simplicity, 
                        # as n_eq is small (500) and n_unk is ~2000.
                        base_w_idx = (k * n + i) * rank
                        for c in range(rank):
                            A[eq_idx, base_w_idx + c] = U[i * m + j1, c] * V[j2 * p + k, c]
                        
                        eq_idx += 1
        
        # Solve A W_flat = b
        # Use least squares if overdetermined or underdetermined.
        # Here n_unk (1980) > n_eq (500), so underdetermined.
        # We want ANY solution. np.linalg.lstsq gives minimum norm solution.
        try:
            W_flat, residuals, rank_A, s = np.linalg.lstsq(A, b, rcond=None)
        except np.linalg.LinAlgError:
            continue
            
        # Check if the solution is exact
        # Residuals should be close to 0
        if len(residuals) == 0 or residuals[0] < 1e-7:
            W = W_flat.reshape(p * n, rank)
            # Verify with the checker to be safe
            if _check_decomposition(U, V, W, n, m, p):
                return U, V, W
                
    return None

def construct():
    """
    Constructs the schoolbook tensor decomposition for multiplying
    a 4x5 matrix by a 5x5 matrix.
    
    Rank = n * m * p = 100.
    
    U: shape (n*m, rank) = (20, 100)
    V: shape (m*p, rank) = (25, 100)
    W: shape (p*n, rank) = (20, 100)
    """
    if _time_exceeded():
        return {"U": [], "V": [], "W": []}
    
    n, m, p = DIMENSIONS
    rank_school = n * m * p
    
    # Try to find a lower rank decomposition
    # Start with rank 99 and go down if time permits?
    # Or just try 99.
    best_U, best_V, best_W = None, None, None
    
    # Try rank 99
    result = _try_random_search(99, n, m, p, iterations=100)
    if result is not None:
        best_U, best_V, best_W = result
    else:
        # Fallback to schoolbook
        rank = rank_school
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
        best_U, best_V, best_W = U, V, W

    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
# EVOLVE-BLOCK-END
