"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 4, 7]

# EVOLVE-BLOCK-START
import time
import numpy as np
from fractions import Fraction
from itertools import product

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Leave 10s margin
    
    n, m, p = DIMENSIONS  # 3, 4, 7
    
    # Build the matrix multiplication tensor T[i,j,k] for (i,j) in [0,n*m), (k) in [0,p)
    # T maps (A, B) -> C where C[i,k] = sum_j A[i,j] * B[j,k]
    # Tensor T_{(i,j), j, k} = delta_{j,j}
    # Actually: T[(i,j), (j,k)] = 1 if the middle index matches
    # More precisely: T[i*m+j, j*p+k] contributes to C[i,k]
    
    # The tensor is: T[a, b, c] where a = i*m+j (row of A, col of A),
    # b = j*p+k (row of B, col of B), c = i*p+k (row of C, col of C)
    # T[a, b, c] = 1 if a = i*m+j, b = j*p+k, c = i*p+k for some i,j,k
    
    # Build the 3-mode tensor as a list of rank-1 terms
    # Each term: u * v * w where u[i*m+j]=1, v[j*p+k]=1, w[i*p+k]=1
    
    rank_schoolbook = n * m * p  # 84
    
    # Store as list of (u_idx, v_idx, w_idx) for each rank-1 term
    terms = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = i * p + k
                terms.append((a, b, c))
    
    # Now try to find a lower-rank decomposition
    # Use the flattening approach: the tensor T can be seen as a matrix
    # T_{(a,b), c} or T_{a, (b,c)} etc.
    
    # The key insight: we can use the structure to find cancellations
    # Group by the output index c = i*p+k
    # For each c, the set of (a,b) pairs that contribute
    
    # Try a structured decomposition:
    # Split the 4 middle indices into groups and use Strassen-like tricks
    
    # For M_{3,4,7}, we can think of it as 4 blocks of M_{3,1,7}
    # But we can do better by using the bilinear structure
    
    # Approach: Use the known fact that R(M_{n,m,p}) can be bounded
    # by combining smaller decompositions.
    
    # Let's try to find the actual rank by checking linear independence
    # of the 84 rank-1 tensors in the space of 3-tensors.
    
    # The space of 3-tensors of size (n*m) x (m*p) x (p*n) has dimension
    # (n*m)*(m*p)*(p*n) = (n*m*p)^2 = 84^2 = 7056
    # But our tensor is sparse.
    
    # Better approach: use the flattening T_{(a,b), c} which is a matrix
    # of size (n*m * m*p) x (p*n) = (12*28) x 21 = 336 x 21
    # The rank of this flattening gives a lower bound on tensor rank.
    
    # But we need the actual tensor rank, not just flattening rank.
    
    # Let me try a direct approach: use Gaussian elimination on the tensor
    # entries to find if any rank-1 term is a linear combination of others.
    
    # Represent each rank-1 term as a vector in R^{(n*m)*(m*p)*(p*n)}
    # That's 7056-dimensional, which is large but manageable.
    
    # Actually, let's use a smarter approach. The tensor T satisfies:
    # T[a, b, c] = 1 if a = i*m+j, b = j*p+k, c = i*p+k
    # = sum over i,j,k of delta_{a, i*m+j} * delta_{b, j*p+k} * delta_{c, i*p+k}
    
    # The 84 terms are linearly independent in general, but there might be
    # relations due to the specific structure.
    
    # Let me try the known decomposition for small cases:
    # For M_{2,2,2}, Strassen gives rank 7 instead of 8.
    # For M_{3,4,7}, we can try to use Strassen on 2x2x2 subblocks.
    
    # Strategy: Partition the 3x4x7 into overlapping 2x2x2 blocks and
    # apply Strassen's algorithm to each, then combine.
    
    # But this is complex. Let me try a simpler approach:
    # Use the fact that R(M_{n,m,p}) <= R(M_{n,m',p'}) * (m/m') * (p/p')
    # for certain divisors.
    
    # Actually, the most reliable approach given time: return the schoolbook
    # rank-84 decomposition, but try to find a few cancellations.
    
    # Let me check: is there a known rank for M_{3,4,7}?
    # Lower bound: max(n*m, m*p, p*n) = max(12, 28, 21) = 28
    # Upper bound: 84 (schoolbook)
    # Best known: likely around 28-40
    
    # Let me try to construct a rank-28 decomposition using the flattening rank.
    
    # The flattening T_{(a,b), c} has rank at most 28 (since c ranges over 21 values,
    # but the structure limits it).
    
    # Actually, let me just compute the rank of the flattening matrices.
    
    # Flattening 1: T_{a, (b,c)} -> matrix of size (n*m) x (m*p * p*n) = 12 x (28*21) = 12 x 588
    # Flattening 2: T_{(a,b), c} -> matrix of size (12*28) x 21 = 336 x 21
    # Flattening 3: T_{a, (b,c)} differently...
    
    # The tensor rank is at least the max of all flattening ranks.
    
    # Let me build the flattening matrix T_{(a,b), c} and compute its rank.
    
    # T_{(a,b), c} = 1 if there exist i,j,k such that a=i*m+j, b=j*p+k, c=i*p+k
    # For fixed c = i*p+k, the valid (a,b) pairs are:
    # a = i*m+j, b = j*p+k for j in range(m)
    # So for each c, there are m=4 entries.
    
    # The flattening matrix is 336 x 21, with each column having 4 ones.
    # The rank of this matrix is at most 21 (number of columns).
    
    # Similarly, T_{a, (b,c)} is 12 x 588, rank at most 12.
    # T_{(a,c), b} is (12*21) x 28 = 252 x 28, rank at most 28.
    
    # So the lower bound from flattening is max(12, 21, 28) = 28.
    
    # Can we achieve rank 28? That would be optimal.
    
    # Let me try to construct a rank-28 decomposition.
    # The idea: use the 28 independent "directions" from the flattening.
    
    # For each (i,k) pair (there are n*p = 21 of them) and each j (m=4),
    # we have a term. But 21*4 = 84, not 28.
    
    # The 28 comes from the (a,c) flattening: there are 12*21 = 252 possible (a,c)
    # but only 28 are "active" in a certain sense.
    
    # Actually, let me think differently. The tensor T can be written as:
    # T = sum_{i,j,k} e_{i*m+j} * e_{j*p+k} * e_{i*p+k}
    
    # This is 84 terms. To reduce rank, we need to find coefficients c_{i,j,k}
    # such that sum c_{i,j,k} * e_{i*m+j} * e_{j*p+k} * e_{i*p+k} = T
    # with fewer nonzero c's, OR find a different set of rank-1 tensors.
    
    # The standard approach: use bilinear algorithms.
    
    # For now, let me return the schoolbook decomposition as a baseline,
    # and add a time check.
    
    U = np.zeros((n * m, rank_schoolbook))
    V = np.zeros((m * p, rank_schoolbook))
    W = np.zeros((p * n, rank_schoolbook))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = i * p + k
                U[a, col] = 1
                V[b, col] = 1
                W[c, col] = 1
                col += 1
    
    best_U, best_V, best_W = U, V, W
    best_rank = rank_schoolbook
    
    # Try to reduce rank by finding linear dependencies among the 84 terms
    # Use rational arithmetic for exactness
    
    # Represent each rank-1 term as a sparse vector
    # Term (i,j,k) has 1 at position (i*m+j, j*p+k, i*p+k)
    
    # Check if any term is a linear combination of others
    # This is equivalent to checking if the 84 vectors in R^7056 are linearly independent
    
    # Use a simpler test: check the flattening matrices for rank deficiency
    
    # Build flattening matrix T_{(a,b), c} of size (n*m*m*p) x (p*n) = 336 x 21
    flatten_ab_c = np.zeros((n * m * m * p, p * n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = i * p + k
                ab_idx = a * (m * p) + b
                flatten_ab_c[ab_idx, c] = 1
    
    # Compute rank of this flattening
    rank_flatten = np.linalg.matrix_rank(flatten_ab_c, tol=1e-10)
    
    # The tensor rank is at least rank_flatten
    # If rank_flatten < 84, we might be able to reduce
    
    # For now, return the schoolbook decomposition
    def to_list(mat):
        return mat.tolist()
    
    if time.time() > deadline:
        return {"U": to_list(best_U), "V": to_list(best_V), "W": to_list(best_W)}
    
    return {"U": to_list(best_U), "V": to_list(best_V), "W": to_list(best_W)}
# EVOLVE-BLOCK-END
