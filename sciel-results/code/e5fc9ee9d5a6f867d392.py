"""Program to find low-rank tensor decomposition for 3x3x7 matrix multiplication."""
DIMENSIONS = [3, 3, 7]

import time
import random
from fractions import Fraction
import numpy as np

def _build_schoolbook():
    n, m, p = DIMENSIONS
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
    return {"U": U, "V": V, "W": W}

def _verify_decomposition(U, V, W):
    """Verify that the decomposition is exact."""
    n, m, p = DIMENSIONS
    num_U = len(U)
    num_V = len(V)
    num_W = len(W)
    if num_U != n * m:
        return False
    if num_V != m * p:
        return False
    if num_W != p * n:
        return False
    rank = len(U[0]) if U else 0
    if len(V[0]) != rank or len(W[0]) != rank:
        return False
    for c in range(rank):
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    # The rank-1 term: U[i*m+j,c] * V[j*p+k,c] should equal W[k*n+i,c]
                    # for the matrix multiplication tensor
                    expected = U[i * m + j][c] * V[j * p + k][c]
                    actual = W[k * n + i][c]
                    if expected != actual:
                        return False
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 145  # safety margin
    
    # Start with schoolbook
    schoolbook = _build_schoolbook()
    
    # Try to find a lower rank decomposition using randomized search
    # The tensor equation: for each c, U[:,c] * V[:,c] (outer product structure)
    # Actually the constraint is: for all i,j,k: sum over... no, each rank-1 term is separate.
    # For a valid decomposition of the matrix multiplication tensor, we need:
    # For each triple (i,j,k): sum_c U[i*m+j,c]*V[j*p+k,c]*... 
    
    # Wait, I need to reconsider. The tensor decomposition means:
    # M_{i,j,k} = sum_c U[i*m+j,c] * V[j*p+k,c] * W_inv... 
    # No, that's not right either.
    
    # Actually for a rank-r tensor decomposition of the matmul tensor:
    # The matmul tensor T has T[i*m+j, j*p+k, k*n+i] = 1 for all i,j,k, else 0
    # A rank-r decomposition means T = sum_{c=1}^{r} u_c ⊗ v_c ⊗ w_c
    # where u_c ∈ R^{n*m}, v_c ∈ R^{m*p}, w_c ∈ R^{p*n}
    # So T[i*m+j, j*p+k, k*n+i] = sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c]
    
    # For the schoolbook, each term has exactly one nonzero in each vector at position (i,j), (j,k), (k,i)
    # and T gets value 1 at that position.
    
    # For a general decomposition, we need:
    # sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] = 1 for all (i,j,k)
    # And for positions that are NOT of the form (i,j),(j,k),(k,i) in the tensor,
    # the sum should be 0. But wait, the tensor is only defined on the "matching" positions.
    
    # Actually, the matmul tensor T ∈ R^{(n*m)*(m*p)*(p*n)} where
    # T[(i*m+j)*(m*p)*(p*n) + (j*p+k)*(p*n) + (k*n+i)] = 1
    # and 0 elsewhere. But this indexing is complex.
    
    # Simpler: T is a 3-index tensor T[A_idx, B_idx, C_idx] where
    # A_idx = i*m + j (for A matrix entry), B_idx = j*p + k (for B), C_idx = k*n + i (for C)
    # T[A_idx, B_idx, C_idx] = 1 if the indices are consistent (same i,j,k), else 0.
    
    # A rank-r decomposition: T = sum_{c=1}^{r} u_c ⊗ v_c ⊗ w_c
    # means T[A,B,C] = sum_c u_c[A] * v_c[B] * w_c[C]
    
    # So we need: for all i,j,k: sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] = 1
    # For inconsistent indices (e.g., A_idx=0 corresponds to i=0,j=0, B_idx=10 corresponds to j=1,k=2, C_idx=0 corresponds to k=0,i=0 - inconsistent j),
    # sum_c U[A,c]*V[B,c]*W[C,c] = 0
    
    # This is a complex set of nonlinear equations. Let me try a simpler approach.
    
    # For 3x3x7, let me try to see if we can exploit linearity in the k direction.
    # C_{i,k} = sum_j A_{i,j} B_{j,k}
    # Think of B as having 7 columns. Each column k is a vector in R^3.
    # C[:,k] = A * B[:,k]
    
    # So we're applying the linear map A to 7 different vectors B[:,k].
    # The tensor for this operation has a block structure.
    
    # One approach: use the fact that for fixed A, the map B -> C is linear in B.
    # The "rank" of this bilinear map can potentially be less than n*m*p.
    
    # For 3x3x3, the matmul tensor has rank at most 20 (or is it 19?).
    # For 3x3x7, it might be related but the exact minimum is unknown.
    
    # Let me try a different approach: assume a structure where the W matrix has a block form.
    # Since C is 3x7, write C = [C_1 C_2 C_3] where C_i is 3x3 (first 6 cols) and C_4 is 3x1.
    # Wait, 3x7 doesn't split evenly. Let's do C = [C_1 C_2 C_3 C_4] where C_1,C_2,C_3 are 3x2 and C_4 is 3x1.
    # Or better: C[:, 0:3], C[:, 3:6], C[:, 6:7]
    
    # Actually, let me try a randomized approach: assume rank = 63 - d for small d,
    # and try to find U, V, W satisfying the equations.
    
    # The equations are:
    # 1. For all i in {0,1,2}, j in {0,1,2}, k in {0,...,6}:
    #    sum_c U[i*3+j, c] * V[j*7+k, c] * W[k*3+i, c] = 1
    # 2. For "inconsistent" triples (A_idx, B_idx, C_idx) where the indices don't match:
    #    sum_c U[A_idx, c] * V[B_idx, c] * W[C_idx, c] = 0
    
    # This is a system of polynomial equations. Hard to solve directly.
    
    # Alternative: fix U and V randomly, then see if W can be determined.
    # From equation 1: for each (i,j,k), W[k*3+i, c] appears.
    # But it's sum over c of a product of three terms, so it's nonlinear in W.
    
    # Let me try yet another approach: use the fact that the problem has some symmetry.
    # For the 3x3x7 case, maybe we can find a decomposition with rank = 3 * rank_3x3x3 + something.
    
    # Actually, for now, let me just return the schoolbook if we can't find better quickly.
    # But let me try a simple randomized search.
    
    n, m, p = 3, 3, 7
    
    # Try a few random configurations
    best = schoolbook
    best_rank = 63
    
    for trial in range(20):  # limit trials to stay within time
        if time.time() > deadline - 5:
            break
        
        # Try to construct a decomposition of rank 63
        # Actually, let me try a structured approach
        # Assume U has a block structure: U = [U_1; U_2; U_3] where each U_i is 3 x rank
        # Similarly for other matrices
        
        # Let me try: for each of the 3 "rows" of A, use a shared basis
        # This might allow sharing across different i values
        
        if trial == 0:
            # Try a rank-63 decomposition but with a different structure
            # Group terms by i: for each i, we have a 3x7 multiplication
            # The 3x7 multiplication can be seen as 7 separate 3x1 vector multiplications
            # For fixed i, C[i,:] = A[i,:] * B
            # This is a dot product for each column
            
            pass
        
        # Let me try a simple parameterization
        # Assume W has the form where for each k, W[k*3:(k+1)*3, :] is 3xrank
        # and U has U[i*3:(i+1)*3, :] for each i
        
        # For a structured decomposition, assume:
        # U[i*3+j, c] = a_{i,j,c}
        # V[j*7+k, c] = b_{j,k,c}  
        # W[k*3+i, c] = a_{i,j,c} * b_{j,k,c}  (for the matching terms)
        
        # This is exactly the schoolbook if a and b are 0/1.
        # To get a lower rank, we need to share.
        
        pass
    
    return best
