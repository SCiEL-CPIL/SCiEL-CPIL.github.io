"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 4, 5]

import numpy as np
from scipy.optimize import minimize

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Schoolbook rank
    schoolbook_rank = n * m * p
    # Lower bound
    lower_bound = max(n*m, m*p, n*p)
    
    # Target tensor
    T = np.zeros((n*m, m*p, n*p))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m+j, j*p+k, k*n+i] = 1.0
                
    # Try ranks from lower bound up to schoolbook
    # We prioritize finding a low rank. 
    # ALS is stochastic in init, so we do multiple trials.
    
    for rank in range(lower_bound, schoolbook_rank + 1):
        for trial in range(50):
            # Random initialization
            # Using small random values to help convergence to simple solutions
            U = np.random.randn(n*m, rank)
            V = np.random.randn(m*p, rank)
            W = np.random.randn(n*p, rank)
            
            # Run ALS
            success = _run_als_and_check(T, U, V, W, n, m, p, rank, max_iters=100)
            if success:
                return _format_solution(U, V, W)
                
    # Fallback to schoolbook
    return _schoolbook(n, m, p)

def _run_als_and_check(T, U, V, W, n, m, p, rank, max_iters=100):
    """
    Run Alternating Least Squares to minimize tensor reconstruction error.
    Then check if rounding to half-integers yields an exact match.
    """
    dim_u, dim_v, dim_w = T.shape
    
    for _ in range(max_iters):
        # Update U
        # For each row a in U, solve (V_r ⊗ W_r) u_a = T[a, :, :]
        # Matrix B is (dim_v * dim_w) x rank
        # Column r is V[:, r] ⊗ W[:, r]
        B = np.zeros((dim_v * dim_w, rank))
        for r in range(rank):
            B[:, r] = np.kron(V[:, r], W[:, r])
        
        for a in range(dim_u):
            t_a = T[a, :, :].flatten()
            # Solve B u_a = t_a
            u_a, residuals, rank_solve, sv = np.linalg.lstsq(B, t_a, rcond=None)
            U[a, :] = u_a
            
        # Update V
        # For each row b in V, solve (U_r ⊗ W_r) v_b = T[:, b, :]
        B = np.zeros((dim_u * dim_w, rank))
        for r in range(rank):
            B[:, r] = np.kron(U[:, r], W[:, r])
            
        for b in range(dim_v):
            t_b = T[:, b, :].flatten()
            v_b, residuals, rank_solve, sv = np.linalg.lstsq(B, t_b, rcond=None)
            V[b, :] = v_b
            
        # Update W
        # For each row c in W, solve (U_r ⊗ V_r) w_c = T[:, :, c]
        B = np.zeros((dim_u * dim_v, rank))
        for r in range(rank):
            B[:, r] = np.kron(U[:, r], V[:, r])
            
        for c in range(dim_w):
            t_c = T[:, :, c].flatten()
            w_c, residuals, rank_solve, sv = np.linalg.lstsq(B, t_c, rcond=None)
            W[c, :] = w_c
            
    # Check for exactness with rounding to half-integers
    # Try rounding to integers first, then half-integers
    U_rounded = np.round(U * 2) / 2
    V_rounded = np.round(V * 2) / 2
    W_rounded = np.round(W * 2) / 2
    
    # Reconstruct
    T_recon = np.zeros_like(T)
    for r in range(rank):
        T_recon += np.einsum('a,b,c->abc', U_rounded[:, r], V_rounded[:, r], W_rounded[:, r])
        
    if np.allclose(T_recon, T, atol=1e-9):
        return True
    return False

def _format_solution(U, V, W):
    # Convert to lists of floats (or ints if they are integers)
    # The problem asks for integer/half-integer entries.
    # We return them as floats in the list, e.g., 0.5, 1.0
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }

def _schoolbook(n, m, p):
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
# EVOLVE-BLOCK-END
