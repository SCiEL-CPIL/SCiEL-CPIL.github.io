"""Matrix multiplication 2,5,6 rank-minimal tensor decomposition."""
import numpy as np
import time
from fractions import Fraction
from itertools import product

DIMENSIONS = [2, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140.0

    n, m, p = DIMENSIONS  # 2, 5, 6

    # Build the rank-54 decomposition using Strassen + naive blocks
    rank = 54
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    col = 0

    def add_strassen_block(i_base, j_base, k_base):
        nonlocal col
        U[i_base * m + j_base][col] += 1
        U[(i_base + 1) * m + (j_base + 1)][col] += 1
        V[j_base * p + k_base][col] += 1
        V[(j_base + 1) * p + (k_base + 1)][col] += 1
        W[k_base * n + i_base][col] += 1
        W[(k_base + 1) * n + (i_base + 1)][col] += 1
        col += 1
        U[(i_base + 1) * m + j_base][col] += 1
        U[(i_base + 1) * m + (j_base + 1)][col] += 1
        V[j_base * p + k_base][col] += 1
        W[k_base * n + (i_base + 1)][col] += 1
        W[(k_base + 1) * n + (i_base + 1)][col] -= 1
        col += 1
        U[i_base * m + j_base][col] += 1
        V[j_base * p + (k_base + 1)][col] += 1
        V[(j_base + 1) * p + (k_base + 1)][col] -= 1
        W[(k_base + 1) * n + i_base][col] += 1
        W[(k_base + 1) * n + (i_base + 1)][col] += 1
        col += 1
        U[(i_base + 1) * m + (j_base + 1)][col] += 1
        V[(j_base + 1) * p + k_base][col] += 1
        V[j_base * p + k_base][col] -= 1
        W[k_base * n + i_base][col] += 1
        W[k_base * n + (i_base + 1)][col] += 1
        col += 1
        U[i_base * m + j_base][col] += 1
        U[i_base * m + (j_base + 1)][col] += 1
        V[(j_base + 1) * p + (k_base + 1)][col] += 1
        W[k_base * n + i_base][col] -= 1
        W[(k_base + 1) * n + i_base][col] += 1
        col += 1
        U[(i_base + 1) * m + j_base][col] += 1
        U[i_base * m + j_base][col] -= 1
        V[j_base * p + k_base][col] += 1
        V[j_base * p + (k_base + 1)][col] += 1
        W[(k_base + 1) * n + (i_base + 1)][col] += 1
        col += 1
        U[i_base * m + (j_base + 1)][col] += 1
        U[(i_base + 1) * m + (j_base + 1)][col] -= 1
        V[(j_base + 1) * p + k_base][col] += 1
        V[(j_base + 1) * p + (k_base + 1)][col] += 1
        W[k_base * n + i_base][col] += 1
        col += 1

    def add_naive_block(i_base, j_base, k_base):
        nonlocal col
        U[i_base * m + j_base][col] += 1
        V[j_base * p + k_base][col] += 1
        W[k_base * n + i_base][col] += 1
        col += 1
        U[i_base * m + j_base][col] += 1
        V[j_base * p + (k_base + 1)][col] += 1
        W[(k_base + 1) * n + i_base][col] += 1
        col += 1
        U[(i_base + 1) * m + j_base][col] += 1
        V[j_base * p + k_base][col] += 1
        W[k_base * n + (i_base + 1)][col] += 1
        col += 1
        U[(i_base + 1) * m + j_base][col] += 1
        V[j_base * p + (k_base + 1)][col] += 1
        W[(k_base + 1) * n + (i_base + 1)][col] += 1
        col += 1

    add_strassen_block(0, 0, 0)
    add_strassen_block(0, 0, 2)
    add_strassen_block(0, 0, 4)
    add_strassen_block(0, 2, 0)
    add_strassen_block(0, 2, 2)
    add_strassen_block(0, 2, 4)
    add_naive_block(0, 4, 0)
    add_naive_block(0, 4, 2)
    add_naive_block(0, 4, 4)

    # Verify col count
    if col != 54:
        # Fallback to naive
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

    # Now attempt rank reduction via dual-variable elimination
    # The tensor T[i,j,k] = sum_r U[i,j,r] * V[j,k,r] * W[k,i,r]
    # For each (i,j,k), T[i,j,k] = 1 (the standard matrix multiplication tensor)
    # We have 54 rank-1 terms. We want to find if any term can be expressed
    # as a linear combination of the others.
    
    # Approach: For each candidate term r to eliminate, check if the remaining
    # 53 terms can still span the tensor. This is equivalent to checking if
    # term r is in the span of the other 53 terms in the tensor space.
    
    # The tensor space has dimension n*m*p = 60.
    # Each rank-1 term defines a vector in this 60-dim space.
    # If the 54 vectors are linearly dependent, we can potentially reduce rank.
    # But we need a specific structure: one term must be expressible as a
    # combination of others such that the product structure is preserved.
    
    # Simpler approach: Try to find if any of the 54 terms is redundant
    # by checking linear dependence of the corresponding vectors.
    
    # Build the 54 vectors in R^60
    vectors = []
    for r in range(54):
        vec = [0] * (n * m * p)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = U[i * m + j][r] * V[j * p + k][r] * W[k * n + i][r]
                    vec[i * m * p + j * p + k] = val
        vectors.append(vec)
    
    # Check for linear dependence using rank computation
    # Convert to numpy and compute rank
    vec_matrix = np.array(vectors, dtype=float).T  # 60 x 54
    
    # Compute rank
    try:
        rank_val = np.linalg.matrix_rank(vec_matrix, tol=1e-10)
    except:
        rank_val = 54
    
    # If rank < 54, there's linear dependence
    if rank_val < 54:
        # Find which terms are dependent
        # Use QR decomposition with column pivoting
        Q, R = np.linalg.qr(vec_matrix, mode='complete')
        # The diagonal of R tells us which columns are independent
        diag_R = np.abs(np.diag(R[:54, :54]))
        # Columns where diag_R is small are dependent
        tol = 1e-8
        independent = diag_R > tol
        
        # Try to eliminate dependent terms
        # But we need to be careful: the elimination must preserve the
        # rank-1 product structure, not just linear span.
        
        # For now, if we find dependence, we still need to verify
        # that the reduced set actually computes the correct tensor.
        
        # Let's try a different approach: for each term, try to express
        # it as a linear combination of others and verify.
        
        # Actually, for matrix multiplication tensors, the rank-1 terms
        # are typically linearly independent (they correspond to different
        # "patterns" of indices). So this approach may not yield reduction.
        
        # Let's try a more targeted approach: look for pairs of terms
        # that can be merged.
        
        pass  # No reduction found via linear dependence
    
    # Alternative strategy: Try to find a better block decomposition
    # that uses fewer total multiplications.
    
    # Current: 6 Strassen (2x2x2) blocks + 3 naive (2x1x2) blocks
    # = 6*7 + 3*4 = 42 + 12 = 54
    
    # Can we do better? Let's try:
    # Partition the 5 columns as 1+1+1+1+1 (all naive): 5*4 = 20 per... no
    # Wait, naive for 2x1 * 1x2 is 4 mults per 2x2 output block.
    # For 5 columns: 5 * 3 * 4 = 60 (schoolbook). That's worse.
    
    # What about 2+3 partition?
    # First 2 columns: Strassen (7 mults per 2x2 output block) -> 3*7 = 21
    # Last 3 columns: Need a 2x3 * 3x2 decomposition for each of 3 output blocks.
    # If rank(2x3 * 3x2) = 12, then 3*12 = 36. Total = 21+36 = 57. Worse.
    # If rank(2x3 * 3x2) = 9, then 3*9 = 27. Total = 21+27 = 48. Better!
    
    # Let's try to construct a rank-9 decomposition for 2x3 * 3x2.
    # A is 2x3, B is 3x2, C is 2x2.
    # C[i,k] = sum_j A[i,j] * B[j,k]
    
    # This is equivalent to computing a 2x2 matrix product where
    # A has an extra column and B has an extra row.
    
    # One approach: use Strassen for the 2x2 sub-block and add corrections.
    # A = [A1 | a3] where A1 is 2x2, a3 is 2x1
    # B = [B1; b3] where B1 is 2x2, b3 is 1x2
    # C = A1*B1 + a3*b3
    
    # A1*B1: rank 7 (Strassen)
    # a3*b3: outer product, rank 4 in naive
    # Total: 11. Not 9.
    
    # Can we do a3*b3 in rank 2? No, it's 4 independent products.
    
    # What if we interleave the Strassen and the outer product?
    # This is the key question. Let's try to search for a rank-9 decomposition.
    
    # Set up the system: we want 9 rank-1 terms (u_r, v_r, w_r) such that
    # for all i in {0,1}, k in {0,1}:
    #   C[i,k] = sum_{r=1}^{9} u_r[i] * v_r[k] * w_r[i,k]
    # where u_r is 2x3, v_r is 3x2, w_r is 2x2.
    
    # This is a system of 4 equations in many variables.
    # Let's use a randomized approach to find a solution.
    
    if time.time() < deadline - 30:
        # Try to find rank-9 decomposition for 2x3 * 3x2
        # Using the fact that this is a bilinear problem
        
        # The tensor for 2x3 * 3x2 has 2*3*2 = 12 "components"
        # T[i,j,k] = 1 for all i,j,k (in the appropriate sense)
        
        # Actually, let's think of it as:
        # C[i,k] = sum_j A[i,j] B[j,k]
        # The tensor T has T[i,j,k] = 1 for all i in {0,1}, j in {0,1,2}, k in {0,1}
        
        # We want to decompose T as sum of rank-1 tensors.
        # A rank-1 tensor has the form u[i] * v[j] * w[k] where u is length 2,
        # v is length 3, w is length 2.
        
        # Let's try to find 9 such terms.
        
        # Use a simple approach: start with the 11-term decomposition (7+4)
        # and try to find 2-term reduction.
        
        # Or try a completely different approach: use the known result that
        # rank(2,3,2) might be 12, not 9.
        
        # Let's verify: the lower bound is max(2*3, 3*2, 2*2) = 6.
        # Upper bound from Strassen+naive is 11.
        # Is 9 achievable?
        
        # For now, let's try a numerical search.
        
        # Actually, let's try a different partition of the 5 columns.
        # What about 2+2+1 but with a better handling of the single column?
        
        # The single column (j=4) contributes 4 mults per 2x2 output block.
        # Can we share some multiplications between the Strassen blocks
        # and the naive block?
        
        # This is the key idea: instead of treating the blocks independently,
        # we can look for shared structure.
        
        # Let's try to construct a decomposition where the last column
        # shares some multiplications with the Strassen blocks.
        
        # This is complex. For now, let's return the rank-54 solution.
        pass
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
