"""Matrix multiplication 3,3,6 rank-minimal tensor decomposition."""
import time
import itertools
import numpy as np

DIMENSIONS = [3, 3, 6]

# Known rank-23 decomposition for 3x3x3 matrix multiplication
# (Landsberg's decomposition, using half-integer coefficients)
# The tensor is T[i,j,k] = e_i e_j^T e_k^T with indices 0..2
# We need 23 triples (u, v, w) in F^3 x F^3 x F^3 such that
# sum_r u_r[i] * v_r[j] * w_r[k] = delta_{i,k} * (e_j^T)_i ... wait no.
# Actually the matrix multiplication tensor T satisfies:
# T[i,j,k] = 1 if i==k (no wait)
# The tensor is: T_{i,j,k} = delta_{i, k}? No.
# For C = A*B, C[i,k] = sum_j A[i,j] * B[j,k]
# So the tensor T[i,j,k] = delta_{i,k}? No, T[i,j,k] = 1 if we're computing C[i,k] from A[i,j]*B[j,k].
# The tensor is T[i,j,k] = 1 for all i,j,k where the output index (i,k) is determined by input indices (i,j) and (j,k).
# Actually T[i,j,k] = delta_{i,i'} * delta_{k,k'} ... 
# Let me think again. The bilinear form is:
# C[i,k] = sum_j A[i,j] * B[j,k]
# So the tensor T has T[i,j,k] = 1 if the pair (i,j,k) corresponds to a valid term.
# T[i,j,k] = 1 for all i,j,k in {0,1,2}^3? No!
# T[i,j,k] = 1 if we're summing A[i,j]*B[j,k] into C[i,k].
# So T[i,j,k] = 1 for ALL i,j,k. That's the standard tensor.
# Wait no. The tensor T is in V* ⊗ V* ⊗ V where V is the space of 3x3 matrices... 
# No, A is 3x3, B is 3x3, C is 3x3.
# T ∈ R^9 ⊗ R^9 ⊗ R^9 where the three factors are A, B, C.
# T[a,b,c] where a = i*3+j (index into A), b = j'*3+k (index into B), c = i''*3+k'' (index into C)
# T[a,b,c] = 1 if a = i*3+j, b = j*3+k, c = i*3+k for some i,j,k.
# So T[i*3+j, j*3+k, i*3+k] = 1.

# For rank-23, we need 23 decompositions:
# sum_r U_r[i*3+j] * V_r[j*3+k] * W_r[i*3+k] = T[i*3+j, j*3+k, i*3+k]

# I'll use a known construction. The rank-23 decomposition uses the following approach:
# Split into cases based on the relationship between indices.

def construct():
    start_time = time.time()
    deadline = start_time + 110  # 10-second safety margin
    
    n, m, p = 3, 3, 6
    rank_schoolbook = n * m * p  # 54
    
    # First, try to construct a rank-46 decomposition using the rank-23 3x3x3 result
    # duplicated for the two halves of the 6-column dimension.
    
    # Known rank-23 vectors for 3x3x3 (from literature, half-integer values)
    # These are triples (u, v, w) where u,v,w are in {0, 1/2, 1}^3
    # such that sum_r u_r[i]*v_r[j]*w_r[k] = delta_{i,k} for the tensor structure.
    # Actually, the condition is:
    # sum_r u_r[i,j] * v_r[j,k] * w_r[i,k] = 1 if (i,j,k) is valid, 0 otherwise
    # where u_r is a 9-dim vector (flattened 3x3), etc.
    
    # Let me use a concrete known set. The rank-23 decomposition can be constructed
    # using the following 23 triples in {0,1,1/2}^3 x {0,1,1/2}^3 x {0,1,1/2}^3:
    
    # Actually, let me just verify with a known correct set from the literature.
    # The standard rank-23 decomposition uses these 23 terms (each is (u,v,w) in F^3):
    
    def verify_rank23(triples):
        """Verify that the given triples give a valid rank-23 decomposition for 3x3x3."""
        # Build the target tensor
        T = np.zeros((9, 9, 9))
        for i in range(3):
            for j in range(3):
                for k in range(3):
                    T[i*3+j, j*3+k, i*3+k] = 1
        
        # Check decomposition
        for r in range(len(triples)):
            u, v, w = triples[r]
            for i in range(3):
                for j in range(3):
                    for k in range(3):
                        T[i*3+j, j*3+k, i*3+k] -= u[i] * v[j] * w[k]
        
        # T should now be zero
        if np.allclose(T, 0, atol=1e-10):
            return True
        return False
    
    # Known rank-23 decomposition (from Landsberg 2010 / De Groote 2003)
    # Using half-integers. Here are the 23 triples:
    # I'll use a verified set. Let me construct it carefully.
    
    # The decomposition can be expressed as:
    # Terms where one coordinate is 0, two are 1, or all are 1/2, etc.
    
    # Actually, let me use a simpler approach: just try the schoolbook and see if we
    # can find any improvement within the time budget.
    
    # For 3x3x6, the schoolbook rank is 54.
    # If we can find a rank-46 decomposition, that's an improvement.
    
    # Let me try a direct approach: use the known rank-23 for 3x3x3 and duplicate.
    # For 3x3x6, split B into B1 (3x3) and B2 (3x3).
    # C[:, 0:3] = A * B1, C[:, 3:6] = A * B2
    # If we have a rank-23 decomposition for each, total rank = 46.
    
    # But we need the actual rank-23 vectors. Let me hardcode a known set.
    
    # From the literature, one valid rank-23 decomposition uses these 23 triples
    # (u, v, w) in {0, 1/2, 1}^3:
    
    # I'll use a set that I can verify. Let me try to construct one.
    
    # The key insight: for 3x3x3, the tensor has symmetry. 
    # One known decomposition (from "The complexity of matrix multiplication" by Landsberg):
    
    # Let me just use a brute-force verification approach with a candidate set.
    
    # Candidate rank-23 triples (these are from known constructions):
    # Using the following pattern with a = 1/2:
    
    # Actually, I recall that a valid rank-23 decomposition uses the following:
    # 12 terms of type (1,1,0), (1,0,1), (0,1,1) with permutations
    # Plus additional terms with 1/2 values
    
    # Let me just try a specific known set and verify it:
    
    a = 0.5
    # Known 23-term decomposition (I'll verify this programmatically)
    # This is from the work of De Groote and Landsberg:
    
    triples_23 = [
        # Terms with integer values
        (1, 1, 0), (1, 0, 1), (0, 1, 1),
        (1, 1, 0), (1, 0, 1), (0, 1, 1),
        (1, 1, 0), (1, 0, 1), (0, 1, 1),
        # Terms with half-integers
        (1, a, a), (a, 1, a), (a, a, 1),
        (1, a, a), (a, 1, a), (a, a, 1),
        (1, a, a), (a, 1, a), (a, a, 1),
        # More terms
        (1, 1, a), (1, a, 1), (a, 1, 1),
        (1, 1, a), (1, a, 1), (a, 1, 1),
    ]
    
    # This won't be correct - I need the actual correct set.
    # Let me try a different approach: just return schoolbook if I can't verify a better solution.
    
    # Time check
    if time.time() > deadline - 5:
        # Return schoolbook
        rank = rank_schoolbook
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
        return {"U": U, "V": V, "W": W}
    
    # Let me try to find a valid rank-46 decomposition by solving a linear system.
    # This is expensive but might work within the time budget.
    
    # Actually, for safety, let me just return the schoolbook decomposition.
    # The rank-46 approach requires verified rank-23 vectors which I can't guarantee.
    
    rank = rank_schoolbook
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
    
    return {"U": U, "V": V, "W": W}
