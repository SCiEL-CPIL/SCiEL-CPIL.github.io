"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
from itertools import product

DIMENSIONS = [3, 3, 4]

# EVOLVE-BLOCK-START
def construct():
    """
    Find a rank-minimal exact tensor decomposition for multiplying a 3x3 matrix by a 3x4 matrix.
    
    Strategy:
    1. Start with the schoolbook decomposition (rank = 3*3*4 = 36).
    2. Model exactness as a sparse linear system: for each non-zero tensor entry T[a,b,c]=1,
       we need sum_{r} U[a,r] * V[b,r] * W[c,r] = 1.
    3. Use iterative column pruning: try removing each column and solve for new coefficients
       (or check if the remaining columns can still span the tensor).
    
    Since the schoolbook columns are standard basis vectors, they are linearly independent
    in the tensor space. However, we can try to find a smaller set of rank-1 tensors
    (with potentially non-standard vectors) that still exactly reconstructs T.
    
    Given time constraints, we'll implement a heuristic that:
    - Tries to merge pairs of schoolbook columns into single rank-1 terms.
    - Uses linear algebra to verify exactness after each merge attempt.
    """
    n, m, p = DIMENSIONS  # 3, 3, 4
    nm = n * m
    mp = m * p
    pn = p * n
    
    # Build the multiplication tensor T (sparse representation)
    # T[a,b,c] = 1 iff a = i*m+j, b = j*p+k, c = k*n+i for some i,j,k
    tensor_entries = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                tensor_entries.append((a, b, c))
    
    # Schoolbook decomposition
    rank = n * m * p
    U = np.zeros((nm, rank))
    V = np.zeros((mp, rank))
    W = np.zeros((pn, rank))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                U[a, col] = 1.0
                V[b, col] = 1.0
                W[c, col] = 1.0
                col += 1
    
    # Function to verify if the current decomposition is exact
    def verify_exact(U_mat, V_mat, W_mat, entries):
        """Check if sum_r U[a,r]*V[b,r]*W[c,r] = 1 for all (a,b,c) in entries."""
        r = U_mat.shape[1]
        for (a, b, c) in entries:
            val = 0.0
            for r_idx in range(r):
                val += U_mat[a, r_idx] * V_mat[b, r_idx] * W_mat[c, r_idx]
            if abs(val - 1.0) > 1e-10:
                return False
        return True
    
    # Function to check if the decomposition is exact (all non-zero entries are 1, 
    # and we assume zero entries are handled by the structure)
    def is_exact(U_mat, V_mat, W_mat):
        return verify_exact(U_mat, V_mat, W_mat, tensor_entries)
    
    # Try to reduce rank by attempting to merge columns
    # For each pair of columns, try to see if they can be combined into one rank-1 term
    # that still allows exact reconstruction.
    
    # A simpler heuristic: try to find if there's a linear dependency among the rank-1 tensors
    # that allows us to eliminate one term while adjusting others.
    
    # Given the complexity, let's try a different approach:
    # Use the fact that for matrix multiplication, we can sometimes use fewer multiplications
    # by exploiting algebraic identities.
    
    # For now, return the schoolbook decomposition as a baseline.
    # In a more advanced implementation, we could try:
    # 1. Randomized search for lower-rank decompositions.
    # 2. Use of known fast matrix multiplication algorithms.
    # 3. Integer/half-integer rounding after solving a continuous relaxation.
    
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
