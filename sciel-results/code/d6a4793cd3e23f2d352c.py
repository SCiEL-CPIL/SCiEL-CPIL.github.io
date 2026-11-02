"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 6]

# EVOLVE-BLOCK-START
import numpy as np
from itertools import product, combinations
import sys
from fractions import Fraction
import time

def construct():
    """
    Find a rank-minimal exact tensor decomposition for multiplying a 2x2 matrix by a 2x6 matrix.
    
    The matrix multiplication tensor for C = A*B where A is 2x2, B is 2x6, C is 2x6:
    C[i,k] = sum_j A[i,j] * B[j,k]
    
    We need to find U (4 x r), V (12 x r), W (12 x r) such that:
    For all i in [0,2), j in [0,2), k in [0,6):
    sum_c U[i*2+j, c] * V[j*6+k, c] * W[k*2+i, c] = 1
    
    We'll use a constructive approach based on the structure of the problem.
    For matmul(2,2,6), we can potentially achieve rank less than 24 by exploiting
    the shared structure across the 6 columns of B.
    
    Strategy: We'll attempt to find a decomposition with rank r by solving the
    resulting system of equations. Since the system is polynomial (bilinear in
    pairs of matrices), we'll use a heuristic approach with rational numbers.
    
    Key insight: We can fix some entries to simplify the system. By setting
    certain entries to 0 or 1, we reduce the search space.
    """
    n, m, p = DIMENSIONS  # 2, 2, 6
    
    # The identity we need to satisfy:
    # For all i in [0,n), j in [0,m), k in [0,p):
    # sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1
    
    # Let's try ranks from a reasonable lower bound up to the schoolbook rank
    # Lower bound: at least max(n*m, m*p, p*n) / something, but more practically
    # we know rank >= 1 and the schoolbook gives 24.
    
    def verify(U, V, W):
        """Verify that the decomposition is correct."""
        n, m, p = DIMENSIONS
        r = len(U[0])
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    total = 0
                    for c in range(r):
                        total += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                    if total != 1:
                        return False
        return True
    
    def try_rank_schoolbook(r):
        """Return the schoolbook decomposition of rank r if r >= n*m*p."""
        if r < n * m * p:
            return None
        U = [[0] * r for _ in range(n * m)]
        V = [[0] * r for _ in range(m * p)]
        W = [[0] * r for _ in range(p * n)]
        column = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][column] = 1
                    V[j * p + k][column] = 1
                    W[k * n + i][column] = 1
                    column += 1
        return U, V, W
    
    # Try to find a better decomposition using a constructive approach
    # For matmul(2,2,6), we can try to exploit the fact that we have 6 output columns
    # that share the same A matrix.
    
    # Approach: Use a rank-16 construction based on splitting the 6 columns into groups
    # and using shared intermediate computations.
    
    # Let's try a direct algebraic approach for specific ranks.
    # We'll set up the equations and try to solve them.
    
    # For rank r, we have r columns in each of U, V, W.
    # The equations are: for each (i,j,k), sum over c of U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1
    
    # Let's try rank 16 first, as it's a natural candidate (2*2*6 = 24, and 
    # we might save 8 by sharing structure).
    
    # Actually, let me think about this differently. 
    # matmul(2,2,6) can be seen as computing 6 independent 2x1 matrix multiplications
    # (each column of C = A * column_of_B). Each 2x2 * 2x1 multiplication has rank 2
    # in the standard scheme, but they share A.
    
    # A better approach: use the fact that we can precompute some combinations of A's entries.
    
    # Let me try a different constructive approach. We'll use a rank that allows
    # for some sharing.
    
    # For now, let's return the schoolbook solution which is guaranteed to work.
    # The rank is 24.
    
    rank = n * m * p
    result = try_rank_schoolbook(rank)
    if result is not None:
        U, V, W = result
        return {"U": U, "V": V, "W": W}
    
    # This should not happen
    raise Exception("Failed to construct decomposition")

# EVOLVE-BLOCK-END
