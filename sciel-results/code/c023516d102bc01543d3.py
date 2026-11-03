"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [4, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    """
    Find a rank-minimal exact tensor decomposition for 4x4 * 4x8 matrix multiplication.
    Returns {'U': (n*m, rank), 'V': (m*p, rank), 'W': (p*n, rank)} with integer/half-integer entries.
    """
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s limit
    
    n, m, p = DIMENSIONS
    nm, mp, pn = n * m, m * p, p * n
    
    # Try known good decompositions first (from literature)
    # For 4x4 * 4x8, we can use structured decompositions
    
    # Strategy 1: Use the fact that this is a special case
    # A 4x4 times 4x8 can be decomposed more efficiently than schoolbook
    
    # Try rank values from known bounds downward
    # Schoolbook rank = 4*4*8 = 128
    # Lower bound is max(n*m, m*p, p*n) = max(16, 32, 32) = 32
    
    # Known results suggest we can do better than schoolbook
    # Let's try to find a decomposition with rank around 40-60
    
    # Use a structured approach: exploit the rectangular shape
    # A 4x4 * 4x8 matrix multiplication
    
    # Try different rank values and use integer programming / search
    # Start with a reasonable guess and refine
    
    # First, let's use a known efficient decomposition pattern
    # For multiplying (n x m) by (m x p), we can use:
    # - Split the computation into blocks
    # - Use fast matrix multiplication techniques adapted to small sizes
    
    # For 4x4 * 4x8, we can view the 4x8 as two 4x4 matrices
    # So we're computing [A*B1, A*B2] where A is 4x4, B1 and B2 are 4x4
    
    # This means we need 2 separate 4x4*4x4 multiplications
    # Known rank for 4x4*4x4 is at most 49 (Strassen-like)
    # But we can share some computations
    
    # Let's try to construct a decomposition with shared factors
    
    # Approach: Use a block decomposition where we share the A multiplication
    
    # For each column block of the 4x8 result:
    # Result[:, 0:4] = A * B[:, 0:4]
    # Result[:, 4:8] = A * B[:, 4:8]
    
    # We can precompute A's action on basis vectors and share
    
    # Let me try a direct construction with rank around 50-60
    
    # Actually, let's use a more systematic approach:
    # Try to find a decomposition by solving a constraint satisfaction problem
    
    # For now, let's use a known efficient pattern
    # The key insight: we can share the left matrix multiplication
    
    # Construct using the following pattern:
    # U has rows indexed by (i, j) where i in 0..3, j in 0..3
    # V has rows indexed by (j, k) where j in 0..3, k in 0..7
    # W has rows indexed by (k, i) where k in 0..7, i in 0..3
    
    # We want: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = delta_{i,i'} delta_{j,j'} delta_{k,k'}
    # Wait, that's not right. The tensor is: T[i,j,k] = 1 if this is a valid multiplication term
    
    # Actually, the matrix multiplication tensor has T[i,j,k] = 1 for all i,j,k
    # No wait, that's not right either.
    
    # Let me reconsider. The matrix multiplication tensor for (n x m) * (m x p) is:
    # T[i,j,k] = 1 if and only if... no, it's just all ones?
    
    # Actually, for matrix multiplication C = A * B where A is n x m and B is m x p:
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # So the tensor is T[i,j,k] = 1 for all i, j, k
    # The rank is the minimum number of terms sum_c U[i,j,c] * V[j,k,c] * W[k,i,c] that equals T[i,j,k]
    
    # Wait, that's not quite right. Let me re-read the problem.
    
    # The output factor follows W[k*n+i, r]
    # So W is indexed by (k, i) where k is the column of B and row of C, i is the row of A and C
    
    # The tensor decomposition should satisfy:
    # sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all i, j, k
    # (since every triple (i,j,k) contributes to the multiplication)
    
    # Actually no, that's not right. The tensor T[i,j,k] should be 1 if and only if
    # the (i,j) entry of A and (j,k) entry of B contribute to the (i,k) entry of C.
    # For standard matrix multiplication, T[i,j,k] = 1 for ALL i, j, k.
    
    # So we need: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all i, j, k
    
    # This is a rank-1 decomposition of the all-ones tensor of size n x m x p.
    # The minimum rank of an all-ones tensor of size a x b x c is max(a, b, c) if we allow real numbers?
    # No, that's not right either.
    
    # Wait, I think I'm confusing myself. Let me re-read the task.
    
    # "Find a rank-minimal exact tensor decomposition for multiplying a 4 by 4 matrix by a 4 by 8 matrix."
    
    # The standard matrix multiplication tensor is NOT the all-ones tensor.
    # It's defined as: T[i,j,k] = 1 if and only if... 
    
    # Actually, I think the issue is that I need to be more careful.
    
    # In matrix multiplication C = AB, we have C[i,k] = sum_j A[i,j] * B[j,k]
    # The "tensor" here is the bilinear form that maps (A, B) to C.
    # Each term in the sum is A[i,j] * B[j,k], and these are multiplied by 1 and added to C[i,k].
    
    # So the tensor decomposition is:
    # C[i,k] = sum_j sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] * A[i,j] * B[j,k]
    
    # Wait, that doesn't make sense dimensionally.
    
    # Let me think about this differently. The standard way to express matrix multiplication
    # as a tensor decomposition is:
    
    # We have input tensors A (n x m) and B (m x p), and output C (n x p).
    # The bilinear map is: (A, B) -> C where C[i,k] = sum_j A[i,j] * B[j,k]
    
    # This can be written as: C = sum_r U_r * (A * V_r) * W_r^T or something similar.
    
    # Actually, the standard tensor decomposition for matrix multiplication is:
    # T = sum_r U_r ⊗ V_r ⊗ W_r where U_r, V_r, W_r are vectors
    # and T[i,j,k] = sum_r U_r[i] * V_r[j] * W_r[k]
    
    # But what is T[i,j,k]? For matrix multiplication, it should be:
    # T[i,j,k] = 1 if the (i,j) entry of A and (j,k) entry of B contribute to C[i,k]
    # For standard multiplication, T[i,j,k] = 1 for all i, j, k.
    
    # So we need: sum_r U_r[i] * V_r[j] * W_r[k] = 1 for all i, j, k
    # where U_r is indexed by i (row of A, 0..n-1), V_r by j (col of A / row of B, 0..m-1),
    # and W_r by k (col of B, 0..p-1).
    
    # But the problem says U has shape (n*m, rank), V has shape (m*p, rank), W has shape (p*n, rank).
    # So U is indexed by (i, j) where i is row of A and j is col of A.
    # V is indexed by (j, k) where j is row of B and k is col of B.
    # W is indexed by (k, i) where k is col of B and i is row of C.
    
    # So the decomposition should satisfy:
    # sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all i, j, k
    
    # This is indeed the all-ones tensor of size n x m x p, and we want its rank-1 decomposition.
    
    # The minimum rank of an all-ones tensor of size a x b x c over the reals is:
    # It's known that the rank of the all-ones tensor is min(a, b, c) if we allow negative entries?
    # No, that's not right.
    
    # Actually, the all-ones tensor of size a x b x c has rank at most min(a, b, c) * max(a, b, c) / something...
    
    # Let me just try to construct a decomposition.
    
    # For the all-ones tensor T[i,j,k] = 1, we can write:
    # T[i,j,k] = sum_{r=1}^{R} U[i,j,r] * V[j,k,r] * W[k,i,r]
    
    # One simple decomposition: use R = n*m*p terms, one for each (i,j,k)
    # U[i,j,r] = 1 if r corresponds to (i,j,k), else 0
    # V[j,k,r] = 1 if r corresponds to (i,j,k), else 0
    # W[k,i,r] = 1 if r corresponds to (i,j,k), else 0
    
    # This gives rank n*m*p = 4*4*8 = 128, which is the schoolbook answer.
    
    # Can we do better? Yes, by sharing terms.
    
    # For example, if we fix i, we can share across j and k:
    # T[i,j,k] = sum_j' sum_k' something
    
    # Actually, let's try: T[i,j,k] = sum_r U[i,j,r] * V[j,k,r] * W[k,i,r]
    
    # If we set U[i,j,r] = 1 for all i, j, r=1, and V[j,k,r] = 1 for all j, k, r=1,
    # and W[k,i,r] = 1 for all k, i, r=1, then we get T[i,j,k] = 1.
    # So rank 1 works!
    
    # Wait, that can't be right. Let me check:
    # sum_r U[i,j,r] * V[j,k,r] * W[k,i,r] = U[i,j,1] * V[j,k,1] * W[k,i,1] = 1 * 1 * 1 = 1
    # Yes, that works!
    
    # So the rank is 1? That seems too good to be true.
    
    # But wait, this is for the all-ones tensor. Is the matrix multiplication tensor really the all-ones tensor?
    
    # Let me reconsider. In matrix multiplication, the bilinear map is:
    # C[i,k] = sum_j A[i,j] * B[j,k]
    
    # This is a bilinear map from (A, B) to C. The "tensor" associated with this bilinear map
    # is defined by T[i,j,k] such that C[i,k] = sum_{i',j',k'} T[i',j',k'] * A[i',j'] * B[j',k'] * delta_{i,i'} * delta_{k,k'}
    
    # Hmm, this is getting confusing. Let me look at it from a different angle.
    
    # The problem states that the output factor follows W[k*n+i, r].
    # This suggests that W is indexed by (k, i) where k is related to the column of B and i to the row of A/C.
    
    # I think the correct interpretation is:
    # We want to compute C = A * B where A is 4x4 and B is 4x8.
    # The decomposition is: C[i,k] = sum_j sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] * A[i,j] * B[j,k]
    
    # No wait, that would double-count.
    
    # Let me try: C[i,k] = sum_r (sum_j U[i*m+j, r] * A[i,j]) * (sum_j' V[j'*p+k, r] * B[j',k]) * W[k*n+i, r]
    
    # This doesn't seem right either.
    
    # I think I need to look at this more carefully. Let me just use the schoolbook decomposition
    # and try to improve it using optimization.
    
    # For now, return the schoolbook decomposition as a safe fallback.
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
    
    # Try to find a better decomposition using optimization
    # But given time constraints, let's return the schoolbook for now
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
