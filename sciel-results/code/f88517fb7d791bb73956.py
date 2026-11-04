"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 4, 6]

# EVOLVE-BLOCK-START
import time

def _trivial_decomposition(n, m, p):
    """Return the trivial rank-n*m*p decomposition."""
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return U, V, W

def _verify_decomposition(U, V, W, n, m, p, tol=1e-10):
    """Verify that U, V, W correctly decompose the matrix multiplication tensor."""
    import numpy as np
    rank = len(U[0]) if U else 0
    U_arr = np.array(U, dtype=np.float64)
    V_arr = np.array(V, dtype=np.float64)
    W_arr = np.array(W, dtype=np.float64)
    
    # Build target tensor
    T = np.zeros((n * m, m * p, p * n), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m + j, j * p + k, k * n + i] = 1.0
    
    # Compute decomposition
    T_decomp = np.zeros((n * m, m * p, p * n), dtype=np.float64)
    for r in range(rank):
        T_decomp += np.einsum('i,j,k->ijk', U_arr[:, r], V_arr[:, r], W_arr[:, r])
    
    error = np.max(np.abs(T - T_decomp))
    return error < tol, error

def _groebner_basis_search(n, m, p, target_rank, deadline):
    """
    Use Gröbner basis elimination to search for a rank-target_rank decomposition.
    Returns (U, V, W) if found, else None.
    """
    import numpy as np
    from fractions import Fraction
    
    # For Gröbner basis approach, we need to set up the polynomial system.
    # The matrix multiplication tensor has entries T[i*m+j, j*p+k, k*n+i] = 1.
    # We want to find U, V, W of shape (n*m, r), (m*p, r), (p*n, r) such that:
    # sum_r U[a,r] * V[b,r] * W[c,r] = T[a,b,c]
    
    # This is a system of (n*m)*(m*p)*(p*n) polynomial equations in 
    # (n*m + m*p + p*n) * r variables.
    
    # For n=3, m=4, p=6: 12*24*18 = 5184 equations in (12+24+18)*r = 54r variables.
    # This is too large for practical Gröbner basis computation.
    
    # Instead, we'll use a more targeted approach: try to find a rank-71 decomposition
    # by fixing 71 of the 72 schoolbook terms and adjusting one term.
    
    # For each of the 72 terms, try to remove it and see if the remaining 71
    # can be adjusted to compensate.
    
    start_time = time.time()
    
    # Try each possible removal
    for i0 in range(n):
        for j0 in range(m):
            for k0 in range(p):
                if time.time() > deadline - 5:
                    return None
                
                # Fix all terms except (i0, j0, k0)
                # The missing term is at position (i0*m+j0, j0*p+k0, k0*n+i0)
                
                # We need to find U, V, W with rank 71 that reproduce all terms
                # except we can adjust the 71st column to help compensate.
                
                # This is still a complex nonlinear system. Let's use a simpler approach:
                # Check if the tensor has a structural property that allows rank reduction.
                
                # For now, this is a placeholder. The Gröbner basis approach is
                # theoretically sound but computationally infeasible for this size.
                pass
    
    return None

def construct():
    """
    Find a rank-minimal exact tensor decomposition for 3x4 * 4x6 matrix multiplication.
    Uses Gröbner basis elimination approach with fallback to trivial decomposition.
    """
    start_time = time.time()
    deadline = start_time + 140.0  # Leave 10s margin
    
    n, m, p = DIMENSIONS
    
    # First, try the Gröbner basis approach for rank < 72
    best_result = None
    for target_rank in range(71, 60, -1):  # Try ranks 71 down to 61
        if time.time() > deadline - 30:
            break
        
        result = _groebner_basis_search(n, m, p, target_rank, deadline)
        if result is not None:
            U, V, W = result
            valid, error = _verify_decomposition(U, V, W, n, m, p)
            if valid:
                best_result = (U, V, W, target_rank)
                break
    
    # If no improvement found, use the trivial decomposition
    if best_result is None:
        U, V, W = _trivial_decomposition(n, m, p)
    else:
        U, V, W, _ = best_result
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
