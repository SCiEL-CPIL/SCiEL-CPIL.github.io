"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [2, 5, 6]

# EVOLVE-BLOCK-START
def _schoolbook(n, m, p):
    """Return U, V, W as lists for the schoolbook decomposition of rank n*m*p."""
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

def _try_rank(n, m, p, rank, deadline):
    """Try to find a decomposition of given rank using linear algebra."""
    import numpy as np
    from scipy.optimize import nnls
    
    # The matrix multiplication tensor T has entries T[i,j,k] for i in [0,n), j in [0,m), k in [0,p)
    # We need: for all i,j,k: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = delta_{i,j,k}
    # where delta is 1 if (i,j,k) corresponds to a valid multiplication term, 0 otherwise.
    
    # Actually, the matrix multiplication tensor is:
    # T[i,j,k] = 1 if the term A[i,j]*B[j,k] contributes to C[i,k]
    # So T[i,j,k] = 1 for all valid (i,j,k) in range.
    
    # We want to factorize this tensor.
    # Fix W and solve for U,V, or use iterative approach.
    
    # Strategy: parameterize with random initial W, then solve linear system for U and V.
    # For each k,i: C[i,k] = sum_j A[i,j]*B[j,k]
    # = sum_j sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    
    # Let's set up the problem differently.
    # For each r, we have three vectors u_r (length n*m), v_r (length m*p), w_r (length p*n).
    # The constraint is: for all i,j,k: sum_r u_r[i*m+j] * v_r[j*p+k] * w_r[k*n+i] = 1
    
    # This is a bilinear system. Let's fix w_r and solve for u_r, v_r.
    
    # Alternative: use the fact that for fixed k, we have a matrix equation.
    # C[:,k] = A @ B[:,k]
    
    # Let's try a different approach: assume a structure.
    # For [2,5,6], try rank 55, 50, 45, 40, 35, 30...
    
    # Actually, let's use a known technique: 
    # The rank of [n,m,p] is at least max(n*m, m*p, p*n) = max(10, 30, 12) = 30.
    # And at most n*m*p = 60.
    
    # Let's try to find a decomposition by solving the system numerically.
    
    # For a given rank r, we have r*10 + r*30 + r*12 = 52r unknowns.
    # We have n*m*p = 60 equations.
    
    # For r >= 2, we have more unknowns than equations, so a solution likely exists.
    # But we need it to be exact (integer or half-integer).
    
    # Let's try a structured approach:
    # Use the fact that the tensor is "matrix multiplication" which has special structure.
    
    # For [2, m, p], we can write:
    # C[0,k] = sum_j A[0,j]*B[j,k]
    # C[1,k] = sum_j A[1,j]*B[j,k]
    
    # Let's try to find a low-rank decomposition by setting up the equations.
    
    # For each (i,j,k), we need:
    # sum_{r=1}^{rank} U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
    
    # Let's fix W and solve for U and V.
    # For fixed W, the equations become:
    # For each (i,j,k): sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
    
    # This is still bilinear in U and V.
    
    # Let's try: fix U and V, solve for W, then iterate.
    
    # Actually, let's try a simpler approach: use the schoolbook as starting point
    # and try to reduce rank by combining terms.
    
    # For now, let's just return schoolbook if we can't find better quickly.
    return None

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 110  # safety margin before 120s limit
    
    # Start with schoolbook
    U, V, W = _schoolbook(n, m, p)
    best_rank = n * m * p
    
    # Try to find better decompositions
    for r in range(best_rank - 1, 29, -1):
        if time.time() > deadline - 10:
            break
        result = _try_rank(n, m, p, r, deadline)
        if result is not None:
            U, V, W = result
            best_rank = r
            break
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
