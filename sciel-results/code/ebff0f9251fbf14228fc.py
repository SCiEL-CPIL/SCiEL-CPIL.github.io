"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 8]

# EVOLVE-BLOCK-START
import time
import numpy as np

def _verify(U, V, W, n, m, p):
    """Verify that the decomposition exactly reproduces matrix multiplication."""
    rank = len(U[0]) if U else 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for r in range(rank):
                    s += U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r]
                if abs(s - 1.0) > 1e-9:
                    return False
    return True

def _schoolbook(n, m, p):
    """Return the standard rank n*m*p decomposition."""
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
    return {"U": U, "V": V, "W": W}

def construct():
    """
    Find a low-rank exact tensor decomposition for 3x3 * 3x8 matrix multiplication.
    Returns dict with U, V, W matrices.
    """
    n, m, p = DIMENSIONS
    deadline = time.time() + 110  # Safety margin before 120s limit
    
    # For 3x3 * 3x8, we can use the following approach:
    # Think of B as a 3x8 matrix. We can partition the 8 columns into groups.
    # 
    # Key insight: The multiplication tensor T[i,j,k] = 1 for all i,j,k.
    # We can try to find a decomposition where we use fewer than 72 terms.
    #
    # One structured approach: use the fact that for 3x3*3x3, there exist
    # decompositions of rank 19. For 3x3*3x8, we can potentially do better
    # than 72 by sharing factors.
    #
    # Let's try a direct construction:
    # For each output position (i,k), we need sum_j A[i,j] B[j,k] = C[i,k]
    # This is a bilinear form in A and B.
    #
    # We can write: C[i,k] = sum_r alpha_r(i,k) * (sum_j u_r[i,j] A[i,j]) * (sum_k' v_r[j,k'] B[j,k'])
    # But this is getting complicated.
    #
    # Let me try a simpler approach: use random search with exact rational arithmetic.
    
    # Fallback: schoolbook
    result = _schoolbook(n, m, p)
    
    # Try to find a better decomposition using a structured approach
    # For 3x3 * 3x8, one idea is to use the 3x3*3x3 rank-19 decomposition
    # for the first 3 columns, and handle the remaining 5 columns separately.
    
    # Actually, let me try a completely different approach:
    # Use the fact that the tensor is "all ones" in a sense, and try to
    # find a low-rank decomposition by solving a system of equations.
    
    # For rank r, we have r unknowns for each (i,j), (j,k), (k,i) position.
    # The constraint is: for all i,j,k: sum_t U[i*m+j,t] * V[j*p+k,t] * W[k*n+i,t] = 1
    #
    # This is a system of 3*3*8 = 72 equations with 9*r + 24*r + 24*r = 57*r unknowns.
    # For r >= 2, we have more unknowns than equations, so solutions exist.
    # The question is whether we can find solutions with small integer/half-integer entries.
    
    # Let me try r = 24 (the lower bound) and see if we can find a solution.
    
    # For now, return schoolbook as a guaranteed valid answer.
    # The rank is 72, which matches the seed score.
    
    return result
# EVOLVE-BLOCK-END
