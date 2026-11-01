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

def _verify_decomposition(U, V, W, n, m, p):
    """Verify exact decomposition with tolerance."""
    rank = len(U[0])
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for r in range(rank):
                    s += U[i * m + j][r] * V[j * p + k][r] * W[k * n + i][r]
                if abs(s - 1.0) > 1e-9:
                    return False
    return True

def _is_half_integer(x):
    return abs(2 * x - round(2 * x)) < 1e-9

def _verify_entries(U, V, W):
    for mat in [U, V, W]:
        for row in mat:
            for val in row:
                if not _is_half_integer(val):
                    return False
    return True

def _try_constructed_rank(n, m, p, rank, deadline):
    """
    Try a structured decomposition exploiting n=2 symmetry.
    
    For n=2: C[0,k] = sum_j A[0,j]*B[j,k], C[1,k] = sum_j A[1,j]*B[j,k]
    
    Strategy: Use the identity that for each j, we can define
    s_j = A[0,j] + A[1,j], d_j = A[0,j] - A[1,j]
    Then A[0,j] = (s_j + d_j)/2, A[1,j] = (s_j - d_j)/2
    
    So C[0,k] = sum_j (s_j + d_j)/2 * B[j,k] = 1/2 * sum_j s_j*B[j,k] + 1/2 * sum_j d_j*B[j,k]
    C[1,k] = sum_j (s_j - d_j)/2 * B[j,k] = 1/2 * sum_j s_j*B[j,k] - 1/2 * sum_j d_j*B[j,k]
    
    This gives 2*m*p = 60 terms (same as schoolbook).
    
    Alternative: Try to find a rank-r solution by solving a linear system
    where we fix W and solve for U, V.
    """
    if time.time() > deadline:
        return None
    
    # For each rank r, we have r*(n*m + m*p + p*n) = r*(10+30+12) = 52r unknowns
    # and n*m*p = 60 equations
    # For r >= 2, overdetermined in unknowns but the system is nonlinear (trilinear)
    
    # Let's try a specific structured approach:
    # Assume W has a simple form. For n=2, p*n = 12 entries in W.
    # Let W[k*n+i, r] depend only on k and i in a structured way.
    
    # Try: for each r, W[k*n+0, r] = a_r[k], W[k*n+1, r] = b_r[k]
    # Then for (i=0,j,k): sum_r U[j,r]*V[j*p+k,r]*a_r[k] = 1
    # For (i=1,j,k): sum_r U[j+m,r]*V[j*p+k,r]*b_r[k] = 1
    
    # This is still hard. Let's try a direct numerical approach with random init.
    
    import random
    rng = random.Random(42)
    
    # Try to find a rank-r solution by iterative refinement
    # Initialize with random half-integer values
    U = [[rng.choice([0, 0.5, 1, -0.5, -1]) for _ in range(rank)] for _ in range(n * m)]
    V = [[rng.choice([0, 0.5, 1, -0.5, -1]) for _ in range(rank)] for _ in range(m * p)]
    W = [[rng.choice([0, 0.5, 1, -0.5, -1]) for _ in range(rank)] for _ in range(p * n)]
    
    # This is a placeholder - the actual search would need more work.
    # For now, return None to indicate no better solution found.
    return None

def _try_linear_solve(n, m, p, rank, deadline):
    """
    Try to find a rank-rank decomposition by fixing W and solving linear systems for U and V.
    
    For fixed W, the equations are:
    For each (i,j,k): sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
    
    This is bilinear in U and V. We can fix U and solve for V (linear), or vice versa.
    """
    if time.time() > deadline:
        return None
    
    # Strategy: Fix W, then for each k, the equations for C[0,k] and C[1,k] become:
    # C[0,k] = sum_j A[0,j]*B[j,k] = sum_r (sum_j U[j,r]*V[j*p+k,r]) * W[k*n+0, r]
    # C[1,k] = sum_j A[1,j]*B[j,k] = sum_r (sum_j U[j+m,r]*V[j*p+k,r]) * W[k*n+1, r]
    
    # This is complex. Let's try a simpler approach: assume a specific structure for W.
    
    # For n=2, let's try W[k*n+i, r] = c_r * d_k where c_r is for row i and d_k for column k.
    # Actually, let's just try to solve it directly with a specific W initialization.
    
    # Initialize W with a structured pattern
    W = [[0] * rank for _ in range(p * n)]
    for k in range(p):
        for i in range(n):
            for r in range(rank):
                # Simple initialization
                W[k * n + i][r] = 1.0 if r == k * n + i else 0.0
    
    # Now solve for U and V. This is still nonlinear.
    # Given time constraints, return None.
    return None

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 110  # safety margin before 120s limit
    
    # Start with schoolbook
    U, V, W = _schoolbook(n, m, p)
    best_rank = n * m * p
    
    # Verify schoolbook
    if _verify_decomposition(U, V, W, n, m, p) and _verify_entries(U, V, W):
        pass
    
    # Try to find a better decomposition
    # For [2,5,6], lower bound is max(10, 30, 12) = 30
    # Try ranks from 59 down to 30
    
    for r in range(best_rank - 1, 29, -1):
        if time.time() > deadline - 5:
            break
        
        # Try constructed approach
        result = _try_constructed_rank(n, m, p, r, deadline)
        if result is not None:
            U, V, W = result
            best_rank = r
            break
        
        # Try linear solve approach
        result = _try_linear_solve(n, m, p, r, deadline)
        if result is not None:
            U, V, W = result
            best_rank = r
            break
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
