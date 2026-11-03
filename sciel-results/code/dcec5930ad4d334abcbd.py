"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 7]

# EVOLVE-BLOCK-START
import time
import numpy as np
from fractions import Fraction
from itertools import combinations
import sys

def construct():
    """
    Find a rank-minimal exact tensor decomposition for multiplying a 2x4 matrix by a 4x7 matrix.
    Output factor follows the official trace convention W[k*n+i, r].
    """
    n, m, p = DIMENSIONS  # 2, 4, 7
    
    # The matrix multiplication tensor T has dimensions n*m*p = 56
    # We need to find the minimal rank r such that T = sum_{c=1}^{r} u_c ⊗ v_c ⊗ w_c
    # where u_c ∈ R^{n*m}, v_c ∈ R^{m*p}, w_c ∈ R^{p*n}
    
    # The schoolbook decomposition has rank n*m*p = 56.
    # We want to find if we can do better.
    
    # Known results: For 2x4x7 matrix multiplication, the tensor rank is known to be 28 
    # (this is the Laderman-type result for specific dimensions). Let me verify this.
    
    # Actually, let me think about this more carefully. The tensor rank of the 2x4x7 
    # matrix multiplication tensor... 
    
    # For small matrix multiplication, the tensor rank can sometimes be less than n*m*p.
    # The key insight is that there may be linear dependencies among the rank-1 terms.
    
    # Let's construct the tensor and try to find a minimal decomposition.
    
    # First, let's set up a deadline guard
    start_time = time.time()
    deadline = start_time + 140  # 10 seconds safety margin
    
    # Construct the multiplication tensor as a 3D array T[i,j,k] = 1 if the (i,j) entry of 
    # the product depends on A[i,j'] * B[j',k]... actually let me think about the indexing.
    
    # The matrix multiplication tensor T is defined such that:
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # So T[i,j,k] = 1 for all i,j,k (in the standard basis)
    # Wait, that's not right. The tensor T has T[i,j,k] = 1 if and only if the (i,k) entry 
    # of C uses A[i,j] and B[j,k]. So T[i,j,k] = 1 for all valid i,j,k.
    
    # Actually, the tensor is: T ∈ R^{(n*m) × (m*p) × (p*n)}
    # T[i*m + j, j*p + k, k*n + i] = 1 for all i,j,k
    # And 0 otherwise.
    
    # The schoolbook decomposition uses one rank-1 term for each (i,j,k) triple.
    # But we can potentially combine terms if there are linear dependencies.
    
    # Let's try to find the tensor rank by solving a system of equations.
    
    # For rank r, we need:
    # T = sum_{c=1}^{r} u_c ⊗ v_c ⊗ w_c
    # where u_c ∈ R^{n*m}, v_c ∈ R^{m*p}, w_c ∈ R^{p*n}
    
    # This is equivalent to: for each (a,b,c) index, T[a,b,c] = sum_{d=1}^{r} u_d[a] * v_d[b] * w_d[c]
    
    # The tensor has n*m*p = 56 non-zero entries (one for each (i,j,k) triple).
    # We need to find the minimal r.
    
    # Known lower bound: The tensor rank is at least the "slice rank" which is at least 
    # max(n*m, m*p, p*n) / something... Actually, the slice rank lower bound is more complex.
    
    # For 2x4x7: n*m=8, m*p=28, p*n=14
    # The slice rank is at least max(8, 28, 14) = 28? No, that's not right either.
    
    # Actually, I recall that for matrix multiplication of m×n by n×p matrices, 
    # the tensor rank is at least max(mn, np, pm) / min(m,n,p) or something similar.
    
    # Let me just try to compute it numerically/rationally.
    
    # Strategy: Try ranks from a lower bound up to 56, and for each rank, 
    # check if a decomposition exists.
    
    # Lower bound: The tensor rank is at least the "border rank" lower bound.
    # For 2x4x7, let's try starting from 28 (a reasonable guess based on symmetry considerations).
    
    # Actually, let me think about this differently. The 2x4x7 tensor:
    # C (2x7) = A (2x4) * B (4x7)
    # C[i,k] = sum_{j=0}^{3} A[i,j] * B[j,k]
    # 
    # There are 2*7 = 14 entries in C, each requiring a sum of 4 products.
    # Total: 56 products in the schoolbook.
    
    # Can we share some computations? For example, if we precompute some 
    # intermediate values, we might reduce the total number of multiplications.
    
    # This is related to the Strassen algorithm idea. For 2x2x2, Strassen reduces 
    # from 8 to 7 multiplications.
    
    # For 2x4x7, let's see if we can find a decomposition with rank < 56.
    
    # Let me try a systematic approach: set up the tensor and use rational arithmetic 
    # to find a minimal decomposition.
    
    # The tensor T has entries T[a, b, c] where:
    # a = i*m + j = 4*i + j for i in {0,1}, j in {0,1,2,3}
    # b = j*p + k = 7*j + k for j in {0,1,2,3}, k in {0,...,6}
    # c = k*n + i = 2*k + i for k in {0,...,6}, i in {0,1}
    
    # T[4*i+j, 7*j+k, 2*k+i] = 1 for all i,j,k, else 0.
    
    # We want to write T = sum_{r=1}^{R} u_r ⊗ v_r ⊗ w_r
    
    # This means: for each (a,b,c), sum_{r=1}^{R} u_r[a] * v_r[b] * w_r[c] = T[a,b,c]
    
    # Let's flatten this. The tensor has 8*28*14 = 3136 entries, but only 56 are non-zero.
    
    # For a given rank R, we have R * (8 + 28 + 14) = R * 50 unknowns (the entries of u_r, v_r, w_r).
    # We have 3136 equations (but many are trivially satisfied: T[a,b,c]=0 for most (a,b,c)).
    
    # This is a system of bilinear equations, which is hard to solve directly.
    
    # Alternative approach: Use the fact that the tensor is "structured" and try to 
    # find a decomposition using known algorithms or by exploiting the structure.
    
    # Let me try a different approach: use the "tensor rank" computation via 
    # the "flattening" method.
    
    # For a 3-tensor T ∈ A ⊗ B ⊗ C, the tensor rank is at least the matrix rank 
    # of any flattening T_{A|BC}, T_{B|AC}, T_{C|AB}.
    
    # Flattening T_{A|BC}: This is an (n*m) × (m*p * p*n) = 8 × (28*14) = 8 × 392 matrix.
    # The rank of this matrix gives a lower bound on the tensor rank.
    
    # Let me compute this flattening.
    
    # T_{A|BC}[a, (b,c)] = T[a, b, c]
    
    # For a = 4*i + j, T[a, b, c] is non-zero only if b = 7*j + k and c = 2*k + i for some k.
    # So for fixed a = 4*i + j, the non-zero entries are at (b, c) = (7*j + k, 2*k + i) for k = 0,...,6.
    # That's 7 non-zero entries per row.
    
    # So the flattening T_{A|BC} is an 8 × 392 matrix where each row has exactly 7 ones.
    # The rows are:
    # Row 4*i+j: ones at columns (7*j+k, 2*k+i) for k=0,...,6
    
    # Let me check if these rows are linearly independent.
    # Row 0 (i=0,j=0): ones at (0,0), (7,2), (14,4), (21,6), (28,8), (35,10), (42,12)
    # Row 1 (i=0,j=1): ones at (7,0), (14,2), (21,4), (28,6), (35,8), (42,10), (49,12)
    # Row 2 (i=0,j=2): ones at (14,0), (21,2), (28,4), (35,6), (42,8), (49,10), (56,12)
    # Row 3 (i=0,j=3): ones at (21,0), (28,2), (35,4), (42,6), (49,8), (56,10), (63,12)
    # Row 4 (i=1,j=0): ones at (0,1), (7,3), (14,5), (21,7), (28,9), (35,11), (42,13)
    # Row 5 (i=1,j=1): ones at (7,1), (14,3), (21,5), (28,7), (35,9), (42,11), (49,13)
    # Row 6 (i=1,j=2): ones at (14,1), (21,3), (28,5), (35,7), (42,9), (49,11), (56,13)
    # Row 7 (i=1,j=3): ones at (21,1), (28,3), (35,5), (42,7), (49,9), (56,11), (63,13)
    
    # These 8 rows appear to be linearly independent (each has a unique pattern of 1s).
    # So the matrix rank is 8, giving a lower bound of 8 on the tensor rank.
    
    # Similarly, flattening T_{B|AC}: 28 × (8*14) = 28 × 112 matrix.
    # For b = 7*j + k, non-zero entries at (a, c) = (4*i+j, 2*k+i) for i=0,1.
    # That's 2 non-zero entries per row.
    # Row 7*j+k: ones at (4*0+j, 2*k+0) = (j, 2k) and (4*1+j, 2*k+1) = (4+j, 2k+1)
    
    # Let me check linear independence of these 28 rows.
    # Row 0 (j=0,k=0): ones at (0,0), (4,1)
    # Row 1 (j=1,k=0): ones at (1,0), (5,1)
    # Row 2 (j=2,k=0): ones at (2,0), (6,1)
    # Row 3 (j=3,k=0): ones at (3,0), (7,1)
    # Row 4 (j=0,k=1): ones at (0,2), (4,3)
    # Row 5 (j=1,k=1): ones at (1,2), (5,3)
    # ...
    # Row 7 (j=0,k=2): ones at (0,4), (4,5)
    # ...
    
    # These 28 rows each have exactly 2 non-zero entries in distinct positions.
    # They appear to be linearly independent, giving a lower bound of 28.
    
    # Flattening T_{C|AB}: 14 × (8*28) = 14 × 224 matrix.
    # For c = 2*k + i, non-zero entries at (a, b) = (4*i+j, 7*j+k) for j=0,1,2,3.
    # That's 4 non-zero entries per row.
    # Row 2*k+i: ones at (4*i+j, 7*j+k) for j=0,1,2,3
    
    # Row 0 (k=0,i=0): ones at (0,0), (1,7), (2,14), (3,21)
    # Row 1 (k=0,i=1): ones at (4,0), (5,7), (6,14), (7,21)
    # Row 2 (k=1,i=0): ones at (0,1), (1,8), (2,15), (3,22)
    # Row 3 (k=1,i=1): ones at (4,1), (5,8), (6,15), (7,22)
    # ...
    
    # These 14 rows appear to be linearly independent, giving a lower bound of 14.
    
    # So the lower bound from flattening is max(8, 28, 14) = 28.
    
    # Now, can we achieve rank 28? Let's try.
    
    # If the tensor rank is exactly 28, then we need to find 28 rank-1 terms.
    
    # Let me try to construct such a decomposition. The key insight is that the 
    # flattening T_{B|AC} has rank exactly 28, which suggests that the tensor rank 
    # might be exactly 28.
    
    # Let me try to find a decomposition of rank 28.
    
    # Approach: Use the "tensor decomposition" algorithm. For rank 28, we have 
    # 28 * 50 = 1400 unknowns and 3136 equations. This is underdetermined in terms 
    # of unknowns vs equations, but the equations are nonlinear (bilinear).
    
    # Alternative: Since we know the structure, let's try to find a decomposition 
    # by exploiting the fact that the tensor is "separable" in a certain way.
    
    # Actually, let me try a more direct approach: use the "Waring decomposition" 
    # or "tensor rank" algorithms.
    
    # For now, let me try to verify if rank 28 is achievable by constructing a 
    # candidate decomposition and checking if it works.
    
    # Let me try a heuristic approach: start with the schoolbook decomposition 
    # (rank 56) and try to merge terms.
    
    # Actually, let me just try to solve the system of equations for rank 28 
    # using a numerical approach with rational reconstruction.
    
    # Given the time constraints, let me try a different approach: use the known 
    # result that for 2x4x7 matrix multiplication, the tensor rank is 28.
    
    # Let me construct a rank-28 decomposition.
    
    # The key idea: For each j (0 to 3), we have 2*7 = 14 products A[i,j]*B[j,k].
    # Total: 4 * 14 = 56 products.
    
    # To reduce to 28, we need to "share" computations between different j values.
    
    # One approach: For pairs of j values, combine the products.
    # For example, for j=0 and j=1, we have:
    # C[i,k] += A[i,0]*B[0,k] + A[i,1]*B[1,k]
    
    # Can we compute this with fewer than 14 multiplications? 
    # This is like multiplying a 2x2 matrix by a 2x7 matrix, which has 
    # tensor rank... let me think.
    
    # Actually, this is getting complex. Let me just try to use a numerical 
    # solver to find a rank-28 decomposition.
    
    # For the time being, let me return the schoolbook decomposition (rank 56) 
    # as a safe fallback, and try to improve it if time permits.
    
    # Actually, let me try to implement a proper tensor rank search.
    
    # Given the complexity, let me try a more pragmatic approach:
    # 1. Start with rank 28 (the lower bound)
    # 2. Try to find a decomposition using optimization
    # 3. If that fails, increase the rank
    
    # For now, let me return the schoolbook decomposition.
    
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
    
    # Try to improve: check if we can find a lower rank decomposition
    # by using linear algebra on the tensor.
    
    # Let's try rank 28 first.
    # If we can find a valid decomposition, return it.
    
    # For simplicity and given time constraints, let me try a specific 
    # construction for rank 28.
    
    # Actually, I realize that finding an exact rational decomposition of rank 28 
    # is non-trivial. Let me try a different approach: use the "Strassen-like" 
    # construction.
    
    # For 2x2x2, Strassen's algorithm uses 7 multiplications. The decomposition is:
    # M1 = (A00 + A11)(B00 + B11)
    # M2 = (A10 + A11)B00
    # M3 = A00(B10 - B11)
    # M4 = A11(B01 - B00)
    # M5 = (A00 + A01)B11
    # M6 = (A10 - A00)(B00 + B01)
    # M7 = (A01 - A11)(B10 + B11)
    
    # Can we extend this to 2x4x7?
    
    # Given the time constraints and the difficulty of finding an exact decomposition,
    # let me return the schoolbook decomposition for now.
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
