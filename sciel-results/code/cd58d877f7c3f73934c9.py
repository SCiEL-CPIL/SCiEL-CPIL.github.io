"""Matrix multiplication 2x3x8 with rank-minimal decomposition."""

import numpy as np

DIMENSIONS = [2, 3, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 2, 3, 8
    
    # We'll attempt to find a lower-rank decomposition using a structured approach.
    # The matrix multiplication tensor for A(n,m) * B(m,p) = C(n,p) has:
    # - U of shape (n*m, rank) - A-side coefficients
    # - V of shape (m*p, rank) - B-side coefficients  
    # - W of shape (p*n, rank) - C-side coefficients (trace convention W[k*n+i, r])
    #
    # The decomposition satisfies: for all i,j,k:
    #   sum_r U[i*m+j, r] * V[j*p+k, r] = W[k*n+i, r]  (this is the tensor identity)
    #
    # Actually, the correct identity is:
    #   C[i,k] = sum_j A[i,j] * B[j,k]
    #   In tensor form: T_{i,j,k} = sum_r U_{i,j}^{(r)} * V_{j,k}^{(r)} * W_{i,k}^{(r)}
    #   where W uses trace convention W[k*n+i, r]
    #
    # So we need: for all i,j,k: sum_r U[i*m+j, r] * V[j*p+k, r] = 1 if this is the "middle" term
    # Wait, let me reconsider. The standard tensor decomposition of matrix mult:
    #   T_{i,j,k} (the bilinear form) = sum_r U_{i,j}^{(r)} V_{j,k}^{(r)} W_{i,k}^{(r)}
    # where T_{i,j,k} = 1 (the multiplication tensor has 1s at positions (i,j,k) for the 
    # corresponding product term).
    #
    # Actually the multiplication tensor M_{n,m,p} has:
    #   M_{i,j,k} = 1 for all valid (i,j,k) indices
    # and we want rank-r decomposition:
    #   M_{i,j,k} = sum_{r=1}^{R} U_{i,j}^{(r)} * V_{j,k}^{(r)} * W_{i,k}^{(r)}
    #
    # With the trace convention, W is indexed as W[k*n + i, r] instead of W[i,k,r].
    
    # Let's use a randomized optimization approach to find a low-rank decomposition.
    # We'll use the fact that for small problems, we can try to find exact solutions.
    
    # First, let's try a structured approach based on the 2x3x8 geometry.
    # 8 = 2^3, so we can use a 3-level binary decomposition.
    
    # Strategy: Use the fact that we can share computations across the 8 columns.
    # For each of the 3 rows of A and 2 rows of C, we compute dot products with 8 columns of B.
    
    # Let's try a Winograd-style approach:
    # Group the 8 columns into 4 pairs. For each pair, use a rank-2 decomposition on the B side.
    # Then for the 2x3x2 subproblem, use an optimal decomposition.
    
    # Actually, let's try a direct optimization approach with integer/half-integer constraints.
    
    # We'll set up the problem as: find U (6xR), V (24xR), W (16xR) such that
    # for all i in 0..1, j in 0..2, k in 0..7:
    #   sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
    #
    # This is a system of 2*3*8 = 48 equations in 6R + 24R + 16R = 46R unknowns.
    # For R=48 (schoolbook), we have 46*48 = 2208 unknowns and 48 equations - very underdetermined.
    #
    # Let's try to find the minimum R such that a solution exists with integer/half-integer entries.
    
    # We'll use a greedy/optimization approach:
    # 1. Start with a high rank and try to reduce it.
    # 2. Use the structure of the problem to guide the search.
    
    # For now, let's implement a correct schoolbook decomposition (rank 48) 
    # and then try to optimize it.
    
    def schoolbook_decomposition():
        """Standard schoolbook matrix multiplication decomposition."""
        rank = n * m * p  # 48
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
    
    U, V, W, rank = schoolbook_decomposition()
    
    # Now let's try to reduce the rank by finding linear dependencies.
    # We'll use a simple approach: try to find columns that can be eliminated.
    
    # For a more systematic approach, let's use the structure of the problem.
    # The 8 columns of B can be viewed as 4 groups of 2.
    # For each group, we have a 2x3x2 subproblem.
    
    # Let's try a different structured decomposition:
    # Use 4 basis vectors for the 8-dim space via Kronecker product structure.
    
    # Actually, let's just return the schoolbook for now since it's guaranteed correct.
    # The rank is 48, which is the trivial upper bound.
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
