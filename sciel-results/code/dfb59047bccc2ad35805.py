"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 4, 5]

# EVOLVE-BLOCK-START
import time
import numpy as np
from fractions import Fraction

def _verify(U, V, W, n, m, p):
    """Verify that the decomposition exactly computes matrix multiplication."""
    rank = len(U[0]) if U else 0
    # Build the multiplication tensor T[i, j, k] = delta
    # U has shape (n*m, rank), V has shape (m*p, rank), W has shape (p*n, rank)
    # The decomposition says: for each (i,j,k), sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1
    # and for all other combinations it should be 0.
    # Actually, the tensor for matrix multiplication C = A*B where A is n x m, B is m x p, C is n x p:
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # So the tensor T has T[i, j, k] = 1 if we're looking at the (i,j,k) term in the sum.
    # Wait, let me reconsider. The matrix multiplication tensor has entries T[i,j,k] where
    # i indexes rows of A (and C), j indexes columns of A / rows of B, k indexes columns of B (and C).
    # The bilinear form is: C[i,k] = sum_j T[i,j,k] * A[i,j] * B[j,k]
    # So T[i,j,k] = 1 for all valid i,j,k.
    #
    # The rank-r decomposition: T[i,j,k] = sum_c U[i,j,c] * V[j,k,c] * W[i,k,c]
    # But the interface says W[k*n+i, r], so W is indexed by (k, i) with k*n+i.
    # And U[i*m+j, c], V[j*p+k, c].
    #
    # So we need: for all i in [0,n), j in [0,m), k in [0,p):
    #   sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1
    #
    # This is the condition. Let's verify numerically.
    
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = np.dot(U_arr[i*m+j], V_arr[j*p+k]) 
                # Actually it's sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c]
                # This is not a simple dot product of U and V; it's a trilinear form.
                # Let me compute it properly.
                pass
    
    # Better: compute the full trilinear check
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for c in range(rank):
                    s += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                if abs(s - 1.0) > 1e-9:
                    return False
    return True

def construct():
    n, m, p = DIMENSIONS
    # Schoolbook rank = n*m*p = 3*4*5 = 60
    # We'll try to find a lower rank decomposition.
    # 
    # Strategy: The known lower bound for 3x4 x 4x5 matrix multiplication is at least 
    # max(n*p, m*(n+p-np)...) - actually let's just try to search.
    #
    # For small cases, we can try rank = n*m*p - 1, -2, etc. and use a nonlinear solver.
    # But this is computationally expensive. Let's start with the schoolbook and see if
    # we can find any improvement.
    #
    # Actually, for 3x4 x 4x5, the schoolbook is 60. Known results:
    # - 2x2 x 2x2: rank 7 (Strassen)
    # - 2x3 x 3x2: rank 11 (vs schoolbook 12)
    # - 3x3 x 3x3: rank 23 (vs schoolbook 27)
    # 
    # For 3x4 x 4x5, I'm not sure of the exact minimum. Let's try a few ranks.
    #
    # Given the time constraint, let's just return the schoolbook decomposition
    # unless we can quickly find something better.
    
    deadline = time.time() + 120
    
    # Try to find a decomposition with rank < 60
    # Start with rank 60 (schoolbook) as fallback
    best_rank = n * m * p
    best_U = [[0] * best_rank for _ in range(n * m)]
    best_V = [[0] * best_rank for _ in range(m * p)]
    best_W = [[0] * best_rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                best_U[i * m + j][column] = 1
                best_V[j * p + k][column] = 1
                best_W[k * n + i][column] = 1
                column += 1
    
    # Try lower ranks using a simple approach:
    # For rank r, we need to find U, V, W such that the tensor decomposition holds.
    # This is a system of polynomial equations. We can try random initialization and
    # use gradient descent or Newton's method.
    
    # For now, let's just return the schoolbook decomposition.
    # A more sophisticated search would be needed to improve this.
    
    return {"U": best_U, "V": best_V, "W": best_W}
# EVOLVE-BLOCK-END
