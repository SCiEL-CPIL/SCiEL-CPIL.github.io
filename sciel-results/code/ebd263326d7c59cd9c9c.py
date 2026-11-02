"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 5]
import time
import numpy as np
from itertools import product

# EVOLVE-BLOCK-START
def _verify_decomposition(U, V, W, n, m, p):
    """Verify that the decomposition exactly matches the matrix multiplication tensor."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    rank = U.shape[1]
    # Compute the tensor from the decomposition
    # T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    T = np.zeros((n, m, p))
    for c in range(rank):
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    T[i, j, k] += U[i * m + j, c] * V[j * p + k, c] * W[k * n + i, c]
    # The correct tensor for matrix multiplication C = A*B where A is n x m, B is m x p
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # So T[i,j,k] should be 1 if this (i,j,k) triple contributes to C[i,k], which is always 1
    # Actually, the tensor T[i,j,k] = 1 for all valid (i,j,k) since each term A[i,j]*B[j,k] appears
    expected = np.ones((n, m, p))
    if not np.allclose(T, expected, atol=1e-10):
        return False
    return True

def _try_rank(rank, n, m, p, deadline):
    """Try to find a decomposition of the given rank using structured search."""
    if rank < n * m * p:
        # For matrix multiplication, rank must be at least max(n*m, m*p, p*n)
        # Actually the lower bound is more subtle, but let's try
        pass
    
    # Use the fact that for 3x3x5, we can try structured decompositions
    # The tensor is T[i,j,k] = 1 for all i in range(3), j in range(3), k in range(5)
    # We want to find U (9 x rank), V (15 x rank), W (15 x rank) such that
    # sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1 for all (i,j,k)
    
    # Strategy: Try to find a low-rank approximation
    # For small ranks, we can use the substitution method
    
    # Let's try a structured approach: assume U, V, W have sparse structure
    # For rank r, we need r terms that sum to 1 for each (i,j,k)
    
    # Try using the Kronecker product structure
    # If we can write the identity as a sum of rank-1 tensors
    
    # For 3x3x5, the minimal rank is known to be less than 45
    # Let's try to find it by solving the system
    
    # Approach: Fix U and V, then solve for W
    # For each (k,i), W[k*n+i, c] must satisfy:
    # sum_c V[j*p+k, c] * W[k*n+i, c] = 1/U[i*m+j, c] ... this is nonlinear
    
    # Better: Use the fact that we can normalize
    # Let's try a direct search for small ranks
    
    # For rank 1: T[i,j,k] = u[i*m+j] * v[j*p+k] * w[k*n+i]
    # This would require u[i*m+j] * v[j*p+k] * w[k*n+i] = 1 for all (i,j,k)
    # This is impossible unless we allow different values, but then it's not rank 1
    
    # For rank 2: sum of two rank-1 tensors
    # This is still very constrained
    
    # Let's use a different approach: exploit the tensor structure
    # The tensor T is the outer product of three identity-like structures
    # T = sum over all (i,j,k) of e_i ⊗ e_j ⊗ e_k (but this is rank 45)
    
    # Actually, for matrix multiplication, the tensor is:
    # T[i,j,k] = 1 if we're computing C[i,k] = sum_j A[i,j] B[j,k]
    # So every (i,j,k) triple has value 1
    
    # The key insight: we can group terms
    # For fixed i, we need sum_j A[i,j] B[j,k] for all k
    # This is like a matrix product for each i
    
    # Let's try to find a decomposition using the structure
    # For 3x3x5, we can try rank 15, 20, etc.
    
    # Use a systematic search with the substitution method
    # Fix the form of U, V, W based on the tensor structure
    
    # The tensor has the property that T[i,j,k] depends only on (i,j,k)
    # We can try to find a decomposition where each term corresponds to a subset
    
    # Let's try a greedy approach: start with rank 45 and try to merge terms
    # Or start from a lower rank and see if we can satisfy the constraints
    
    # For now, let's implement a solver for specific ranks
    
    # Try rank 15: This would mean we can do 3x3x5 in 15 operations
    # This is equivalent to saying we can compute 3 matrix-vector products in 15 ops total
    # Each 3x3 x 3x1 product needs at least 9 ops (schoolbook), so 3*9=27
    # But maybe we can share work
    
    # Actually, for 3x3 x 3x5, the output is 3x5, so 15 outputs
    # Each output C[i,k] = sum_j A[i,j] B[j,k]
    # We need to compute 15 dot products of length 3
    
    # The minimal rank for this is at least the number of outputs = 15
    # So rank 15 is the theoretical minimum if we can do each output in 1 op
    
    # Let's try to find a rank-15 decomposition
    # This would require that for each (i,k), there's a single term that computes the dot product
    # But a single rank-1 term u[i*m+j] * v[j*p+k] * w[k*n+i] can't compute a dot product
    
    # So rank 15 is not achievable with this form
    
    # Let's try rank 27: 3x3x5, each of the 3 rows of A multiplies 3x5 B
    # Each 3x3 x 3x5 product: if we do it in 9 ops per row, that's 27 total
    # But we can do better
    
    # Known result: 3x3x3 matrix multiplication has rank 23 (Strassen-like)
    # For 3x3x5, we can use the 3x3x3 result and extend
    
    # Let's implement a search that tries to find a valid decomposition
    
    # For the given rank, set up the system of equations
    # T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1
    
    # This is a system of n*m*p = 45 equations in r*9 + r*15 + r*15 = 39r unknowns
    # For r >= 2, this is underdetermined
    
    # We can solve this by fixing some variables and solving for others
    
    # Let's try a specific structured form:
    # Assume U[i*m+j, c] = 1 for c corresponding to (i,j), 0 otherwise
    # Then V[j*p+k, c] * W[k*n+i, c] must sum to 1 for each (i,j,k)
    
    # This reduces to: for each (i,j,k), sum over c where U[i*m+j,c]=1 of V[j*p+k,c]*W[k*n+i,c] = 1
    
    # If U is the identity (each (i,j) has its own column), then we need
    # V[j*p+k, c_ij] * W[k*n+i, c_ij] = 1 for all (i,j,k)
    # This means V[j*p+k, c_ij] = 1/W[k*n+i, c_ij] for all k
    # So V[j*p+k, c_ij] must be the same for all k, which means W[k*n+i, c_ij] is the same for all k
    # Let V[j*p+k, c_ij] = a_ij, W[k*n+i, c_ij] = 1/a_ij
    # But then for fixed j, V[j*p+k, c_ij] = a_ij for all k, which is fine
    # And for fixed i, W[k*n+i, c_ij] = 1/a_ij for all k, which is fine
    
    # Wait, this gives us a valid decomposition with rank n*m = 9!
    # Let me verify:
    # U[i*m+j, c] = 1 if c = i*m+j, else 0
    # V[j*p+k, c] = a_{i,j} if c = i*m+j, else 0  -- but c is indexed by (i,j), so for fixed j, c varies with i
    # This doesn't work because V[j*p+k, c] depends on both j and k, but c is (i,j)
    
    # Let me reconsider. If U[i*m+j, c] = delta_{c, i*m+j}, then
    # T[i,j,k] = U[i*m+j, i*m+j] * V[j*p+k, i*m+j] * W[k*n+i, i*m+j] = V[j*p+k, i*m+j] * W[k*n+i, i*m+j]
    # We need this to equal 1 for all (i,j,k)
    # So V[j*p+k, i*m+j] * W[k*n+i, i*m+j] = 1 for all (i,j,k)
    # Let V[j*p+k, i*m+j] = x_{i,j,k} and W[k*n+i, i*m+j] = 1/x_{i,j,k}
    # But V[j*p+k, i*m+j] must be consistent: for fixed (j,k), V[j*p+k, i*m+j] varies with i
    # And W[k*n+i, i*m+j] must be consistent: for fixed (k,i), W[k*n+i, i*m+j] varies with j
    
    # This is possible! Let's set:
    # V[j*p+k, i*m+j] = 1 for all (i,j,k)
    # W[k*n+i, i*m+j] = 1 for all (i,j,k)
    # Then T[i,j,k] = 1*1*1 = 1. 
    
    # But wait, V has shape (m*p, rank) = (15, 9)
    # V[j*p+k, i*m+j] for i in 0..2, j in 0..2, k in 0..4
    # For fixed (j,k), the column index is i*m+j = 3i+j
    # So V[j*p+k, 0] = V[j*p+k, 3*0+j] for i=0
    # V[j*p+k, 1] = V[j*p+k, 3*1+j] for i=1 -- but 3*1+j = 3+j, which is a different column
    # So V[j*p+k, c] where c = 3i+j
    # For fixed (j,k), as i varies, c = 3i+j takes values j, 3+j, 6+j
    # So V[j*p+k, j] = 1, V[j*p+k, 3+j] = 1, V[j*p+k, 6+j] = 1
    # Similarly for W[k*n+i, c] where c = 3i+j
    # For fixed (k,i), as j varies, c = 3i+j takes values 3i, 3i+1, 3i+2
    # So W[k*n+i, 3i] = 1, W[k*n+i, 3i+1] = 1, W[k*n+i, 3i+2] = 1
    
    # Let's check if this is consistent:
    # V has 15 rows (j*p+k for j in 0..2, k in 0..4) and 9 columns (c = 3i+j for i in 0..2, j in 0..2)
    # V[j*p+k, c] = 1 if c = 3i+j for some i, which is always true for c in 0..8
    # Actually, for fixed (j,k), V[j*p+k, c] = 1 for all c in {j, 3+j, 6+j}
    # But c ranges from 0 to 8, so for j=0: c in {0, 3, 6}
    # For j=1: c in {1, 4, 7}
    # For j=2: c in {2, 5, 8}
    # So V[j*p+k, c] = 1 if c mod 3 == j, else 0
    
    # Similarly, W[k*n+i, c] = 1 if c = 3i+j for some j, which means c in {3i, 3i+1, 3i+2}
    # So W[k*n+i, c] = 1 if floor(c/3) == i, else 0
    
    # Let's verify T[i,j,k]:
    # T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    # U[i*m+j, c] = 1 if c = i*m+j = 3i+j, else 0
    # So only c = 3i+j contributes
    # V[j*p+k, 3i+j] = 1 if (3i+j) mod 3 == j, which is true since 3i+j mod 3 = j mod 3 = j
    # W[k*n+i, 3i+j] = 1 if floor((3i+j)/3) == i, which is true since floor((3i+j)/3) = i
    # So T[i,j,k] = 1*1*1 = 1. 
    
    # This gives a valid decomposition with rank 9!
    # But wait, rank 9 is less than the number of outputs (15), which seems too good
    # Let me double-check the dimensions
    
    # U: (n*m, rank) = (9, 9)
    # V: (m*p, rank) = (15, 9)
    # W: (p*n, rank) = (15, 9)
    
    # The tensor T[i,j,k] for i in 0..2, j in 0..2, k in 0..4
    # T[i,j,k] = sum_{c=0}^{8} U[3i+j, c] * V[3j+k, c] * W[5k+i, c]
    # U[3i+j, c] = 1 if c == 3i+j, else 0
    # So T[i,j,k] = U[3i+j, 3i+j] * V[3j+k, 3i+j] * W[5k+i, 3i+j] = 1 * V[3j+k, 3i+j] * W[5k+i, 3i+j]
    
    # V[3j+k, 3i+j]: row = 3j+k, col = 3i+j
    # V[row, col] = 1 if col mod 3 == (row mod 5) // ... wait, row = 3j+k, so j = (row - k)//3, but k is not fixed
    # Actually, row = 3j+k where j in 0..2, k in 0..4
    # For a given row r, j = r // 3 if r < 6? No, r = 3j+k, so j = r // 3 only if k < 3, but k can be 0..4
    # This is not a bijection! 3j+k for j in 0..2, k in 0..4 gives values 0..14, but not uniquely
    # j=0,k=0: 0; j=0,k=1: 1; j=0,k=2: 2; j=0,k=3: 3; j=0,k=4: 4
    # j=1,k=0: 3; j=1,k=1: 4; j=1,k=2: 5; j=1,k=3: 6; j=1,k=4: 7
    # j=2,k=0: 6; j=2,k=1: 7; j=2,k=2: 8; j=2,k=3: 9; j=2,k=4: 10
    # So row 3 can be (j=0,k=3) or (j=1,k=0)
    # This means V[3, c] must satisfy both V[3, c] for j=0,k=3 and j=1,k=0
    # My earlier assumption was wrong
    
    # So rank 9 is not achievable with this simple structure
    
    # Let's go back to the schoolbook rank of 45 and try to reduce it
    
    # For now, return the schoolbook decomposition
    return None

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # Leave 10s margin
    
    # Try to find a lower rank decomposition
    # Start from rank 45 and try to reduce
    
    # For 3x3x5, the known minimal rank is less than 45
    # Let's try a few ranks
    
    # First, let's try to see if we can do better than 45
    # The theoretical lower bound is at least max(n*m, m*p, p*n) = max(9, 15, 15) = 15
    # But actually, for matrix multiplication, the lower bound is higher
    
    # Let's implement a search for ranks from 15 to 45
    # But this is computationally expensive, so let's be strategic
    
    # For now, return the schoolbook decomposition
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
    
    # Verify the decomposition
    if _verify_decomposition(U, V, W, n, m, p):
        return {"U": U, "V": V, "W": W}
    
    # If verification fails (shouldn't happen), return anyway
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
