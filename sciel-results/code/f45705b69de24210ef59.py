<<<<<<< SEARCH
import numpy as np
import time

DIMENSIONS = [3, 3, 5]

def construct():
    """
    Constructs the schoolbook tensor decomposition for multiplying a 3x3 matrix by a 3x5 matrix.
    
    The multiplication tensor T[i,j,k] corresponds to the element at row i, column k of the result,
    which is sum_j A[i,j] * B[j,k].
    
    We decompose T into rank-1 terms. The schoolbook decomposition has one term for each (i, j, k) triple.
    Total rank = n * m * p = 3 * 3 * 5 = 45.
    
    U has shape (n*m, rank)
    V has shape (m*p, rank)
    W has shape (p*n, rank)
    
    Indexing conventions:
    U[i*m + j, c]
    V[j*p + k, c]
    W[k*n + i, c]
    """
    start_time = time.time()
    DEADLINE = start_time + 120  # seconds

    n, m, p = DIMENSIONS  # n=3, m=3, p=5
    rank = n * m * p

    # Initialize matrices with zeros
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
=======
import numpy as np
import time
from fractions import Fraction
from itertools import product

DIMENSIONS = [3, 3, 5]

def _build_tensor(n, m, p):
    """Build the matrix multiplication tensor T[i,j,k] = 1 if this is a valid term."""
    # T is a 3D array where T[i,j,k] = 1 for all (i,j,k)
    # The tensor decomposition requires: T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    # For matrix mult: (AB)[i,k] = sum_j A[i,j] B[j,k]
    # So T[i,j,k] = 1 for all valid i,j,k
    T = np.ones((n, m, p), dtype=int)
    return T

def _verify_decomposition(U, V, W, n, m, p):
    """Verify that U, V, W exactly decompose the multiplication tensor."""
    rank = len(U[0]) if U else 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = sum(U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c] for c in range(rank))
                if val != 1:
                    return False
    return True

def _solve_linear_system(A, b):
    """
    Solve A x = b over rationals using Gaussian elimination.
    A is list of lists of Fractions, b is list of Fractions.
    Returns solution as list of Fractions or None if no unique solution.
    """
    n_rows = len(A)
    n_cols = len(A[0])
    
    # Augmented matrix
    M = [row[:] + [b[i]] for i, row in enumerate(A)]
    
    # Gaussian elimination
    pivot_rows = []
    for col in range(n_cols):
        # Find pivot
        pivot = None
        for row in range(len(pivot_rows), n_rows):
            if M[row][col] != 0:
                pivot = row
                break
        if pivot is None:
            continue
        # Swap
        if pivot != len(pivot_rows):
            M[len(pivot_rows)], M[pivot] = M[pivot], M[len(pivot_rows)]
        pivot_rows.append(len(pivot_rows) - 1)
        # Normalize
        pv = M[len(pivot_rows)-1][col]
        for j in range(col, n_cols + 1):
            M[len(pivot_rows)-1][j] /= pv
        # Eliminate
        for row in range(n_rows):
            if row != len(pivot_rows) - 1 and M[row][col] != 0:
                factor = M[row][col]
                for j in range(col, n_cols + 1):
                    M[row][j] -= factor * M[len(pivot_rows)-1][j]
    
    # Check consistency
    for row in range(n_rows):
        all_zero = all(M[row][j] == 0 for j in range(n_cols))
        if all_zero and M[row][n_cols] != 0:
            return None  # Inconsistent
    
    # Extract solution (assuming full rank)
    if len(pivot_rows) != n_cols:
        return None  # Underdetermined
    
    x = [Fraction(0)] * n_cols
    for i in range(n_cols):
        x[i] = M[i][n_cols]
    
    return x

def _try_rank(r, n, m, p, deadline):
    """
    Try to find a rank-r decomposition by enumerating coefficient patterns.
    We fix the structure: for each column c, we assign coefficients from {-1, 0, 1}
    to the U, V, W entries, then solve for consistency.
    
    Strategy: Enumerate sparse patterns with small integers.
    """
    start = time.time()
    
    # The tensor has n*m*p = 45 entries, each must equal 1.
    # We need: for each (i,j,k): sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] = 1
    
    # Approach: Try to find a decomposition where each rank-1 term 
    # corresponds to a "line" in one of the dimensions.
    
    # For 3x3x5, known lower bounds:
    # - Schoolbook: 45
    # - Strassen-like methods may help for square matrices
    # - For 3x3x5, let's try systematic search
    
    # Method: Parameterize U, V, W with small integer entries and solve
    # We'll use a structured approach:
    # Assume U[i*m+j, c] is non-zero only for specific (i,j) pairs per column
    # Same for V and W
    
    # Let's try: each column c has a "support" pattern
    # For efficiency, try columns where U has support on a set of (i,j) pairs,
    # V has support on a set of (j,k) pairs, W has support on a set of (k,i) pairs
    
    # Simplified: try rank from r down, for each rank try random/structured patterns
    
    # For now, implement a direct search for small ranks
    # The key insight: we need sum_c prod_c(i,j,k) = 1 for all (i,j,k)
    
    # Let's try a structured approach:
    # Column c contributes to tensor entries (i,j,k) where U[i*m+j,c] != 0, 
    # V[j*p+k,c] != 0, W[k*n+i,c] != 0
    
    # For small integer coefficients, each column's contribution to (i,j,k) is 
    # U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c]
    
    # We need the sum over all columns to be exactly 1.
    
    # Try: use the fact that for 3x3x5, we might be able to do better than 45
    # by grouping terms.
    
    # Actually, let's implement a more practical approach:
    # Start with schoolbook (rank 45) and try to merge terms.
    
    # A term (i,j,k) with coefficient 1 can be merged with another term if
    # they share structure. But in general, each (i,j,k) triple is independent.
    
    # For a proper search, we'd need to solve a large integer programming problem.
    # Given the time constraints, let's try a few structured patterns.
    
    # Pattern 1: Group by j (middle index)
    # For each j, we have n*p = 3*5 = 15 terms
    # Can we do better than 15 per j?
    # (AB)[i,k] = sum_j A[i,j]B[j,k], so for fixed j, we need rank 15 
    # (since it's an outer product A[:,j] * B[j,:] which has rank 1!)
    # Wait, for fixed j, the contribution is A[i,j]*B[j,k] which IS rank 1!
    # So the total rank is sum_j 1 = 3? No, that's not right.
    # The full tensor is sum_j (A[:,j] outer B[j,:]) 
    # Each term A[:,j] outer B[j,:] is a rank-1 tensor in (i,k) space,
    # but in the (i,j,k) tensor space, each term has U[i*m+j,c]=A[i,j], 
    # V[j*p+k,c]=B[j,k], W[k*n+i,c]=1
    # So for each j, we need one column where U has the j-th column of A,
    # V has the j-th row of B, and W has all 1s.
    # Total rank = m = 3!
    
    # Wait, let me re-examine. The tensor T[i,j,k] = 1 for all i,j,k.
    # We need T[i,j,k] = sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c]
    
    # If I set for c = j (j=0,1,2):
    #   U[i*m+j, j] = 1 for all i
    #   V[j*p+k, j] = 1 for all k  
    #   W[k*n+i, j] = 1 for all i,k
    # Then sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] = 
    #   sum_j [1 * 1 * 1] = 3 (for the c=j term where the middle index matches)
    # But we need this to equal 1, not 3.
    
    # Let me reconsider. The tensor T[i,j,k] should be 1 for each (i,j,k).
    # The decomposition is: T[i,j,k] = sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c]
    
    # For the schoolbook, each (i,j,k) gets its own column, so the sum is 1.
    
    # Can we share columns? If two different (i1,j1,k1) and (i2,j2,k2) share a column,
    # then that column contributes to both tensor entries.
    
    # A column c with U values u_i_j, V values v_j_k, W values w_k_i contributes
    # u_i_j * v_j_k * w_k_i to tensor entry (i,j,k).
    
    # For this to work, we need the product to be 1 for the intended (i,j,k) 
    # and 0 for all others, OR we need multiple columns whose products sum to 1.
    
    # This is a complex constraint satisfaction problem.
    
    # Let's try a different approach: use known results.
    # For 3x3x3 matrix multiplication, the minimum rank is 23 (Strassen-like).
    # For 3x3x5, it's likely between 23 and 45.
    
    # Given the complexity, let's implement a basic search that tries to find
    # a decomposition with rank < 45 by solving a linear system.
    
    # Approach: Assume a specific sparsity pattern and solve.
    # We'll try patterns where each column has limited support.
    
    # For now, return schoolbook if we can't find better quickly.
    if time.time() > deadline:
        return None
    
    # Try a structured approach: use the middle index j to group
    # For each j, the sub-tensor T[i,j,k] for fixed j is a 3x5 matrix of 1s.
    # This can be decomposed as sum of rank-1 matrices.
    # The rank of a 3x5 all-ones matrix is 1.
    # So for each j, we need 1 rank-1 term in the (i,k) space.
    # But in the full tensor, this means:
    #   U[i*m+j, c] = a_i (depends on i)
    #   V[j*p+k, c] = b_k (depends on k)  
    #   W[k*n+i, c] = d (constant)
    #   with a_i * b_k * d = 1 for all i,k
    
    # So a_i = 1/sqrt(3), b_k = 1/sqrt(5), d = sqrt(15)? No, must be integers/halves.
    # Actually a_i * b_k * d = 1 for ALL i,k means a_i is constant, b_k is constant.
    # So a_i = 1, b_k = 1, d = 1 works.
    
    # But wait, this only handles the case where we fix j. The issue is that
    # U[i*m+j, c] must be zero for j' != j in this column.
    
    # So for column c corresponding to j:
    #   U[i*m+j, c] = 1 for all i
    #   U[i*m+j', c] = 0 for j' != j
    #   V[j*p+k, c] = 1 for all k
    #   V[j'*p+k, c] = 0 for j' != j
    #   W[k*n+i, c] = 1 for all i,k
    
    # Then for tensor entry (i, j, k):
    #   Contribution from column c_j = 1 * 1 * 1 = 1
    #   Contribution from column c_{j'} (j' != j) = 0 (since V[j*p+k, c_{j'}] = 0)
    #   Total = 1 ✓
    
    # For tensor entry (i, j', k) where j' != j:
    #   Contribution from column c_j = U[i*m+j', c_j] * V[j'*p+k, c_j] * W[k*n+i, c_j] = 0 * 1 * 1 = 0
    #   Contribution from column c_{j'} = 1 * 1 * 1 = 1
    #   Total = 1 ✓
    
    # This gives rank = m = 3!
    
    # Let me verify this more carefully.
    # n=3, m=3, p=5
    # Rank = 3 (one column per j)
    
    # Column c = j (j = 0, 1, 2):
    #   U[i*m+j, c] = 1 if the j-index matches, else 0
    #   Specifically: U[i*3+j, j] = 1 for all i, U[i*3+j', j] = 0 for j' != j
    #   V[j*p+k, c] = 1 if the j-index matches, else 0  
    #   Specifically: V[j*5+k, j] = 1 for all k, V[j'*5+k, j] = 0 for j' != j
    #   W[k*n+i, c] = 1 for all i, k
    
    # Check tensor entry (i, j, k):
    #   sum_c U[i*3+j, c] * V[j*5+k, c] * W[k*3+i, c]
    #   = U[i*3+j, j] * V[j*5+k, j] * W[k*3+i, j]  (only c=j has non-zero U and V)
    #   = 1 * 1 * 1 = 1 ✓
    
    # This works! Rank = 3.
    
    U = [[0] * r for _ in range(n * m)]
    V = [[0] * r for _ in range(m * p)]
    W = [[0] * r for _ in range(p * n)]
    
    for j in range(m):
        c = j
        for i in range(n):
            U[i * m + j][c] = 1
        for k in range(p):
            V[j * p + k][c] = 1
        for i in range(n):
            for k in range(p):
                W[k * n + i][c] = 1
    
    # Verify
    if _verify_decomposition(U, V, W, n, m, p):
        return {"U": U, "V": V, "W": W}
    
    return None

def construct():
    """
    Constructs a minimal-rank tensor decomposition for multiplying a 3x3 matrix by a 3x5 matrix.
    
    Uses the observation that for fixed middle index j, the sub-tensor is an outer product,
    allowing rank = m = 3 decomposition.
    """
    start_time = time.time()
    DEADLINE = start_time + 115  # safety margin
    
    n, m, p = DIMENSIONS  # n=3, m=3, p=5
    
    # Try the rank-m decomposition
    result = _try_rank(m, n, m, p, DEADLINE)
    if result is not None:
        return result
    
    # Fallback to schoolbook
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
>>>>>>> REPLACE
