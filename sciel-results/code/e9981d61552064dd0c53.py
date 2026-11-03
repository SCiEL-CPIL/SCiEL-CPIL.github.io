"""Program for matrix multiplication 2x3 by 3x5 using tensor code / Berlekamp-Massey approach."""
import numpy as np
from itertools import product

DIMENSIONS = [2, 3, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    U_dim = n * m  # 6
    V_dim = m * p  # 15
    W_dim = p * n  # 10

    # The matrix multiplication tensor T[i,j,k] = 1 for all valid (i,j,k)
    # We seek rank-r decomposition: T = sum_c u_c ⊗ v_c ⊗ w_c
    # where u_c ∈ R^{n*m}, v_c ∈ R^{m*p}, w_c ∈ R^{p*n}
    
    # Key insight: The tensor is "all ones" in the sense that every valid
    # (i,j,k) tuple contributes 1. This is a structured tensor.
    #
    # For matmul_{n,m,p}, the tensor can be written as:
    #   T[i,j,k] = 1 for all i∈[0,n), j∈[0,m), k∈[0,p)
    #
    # This is the tensor of the bilinear map (A,B) -> C where C[i,k] = sum_j A[i,j]B[j,k]
    #
    # The schoolbook rank is n*m*p = 30.
    #
    # However, we can exploit the structure. The tensor is a Kronecker-like product
    # of three "ones" tensors in a specific way. Let's think about it differently.
    #
    # Actually, the matmul tensor M_{n,m,p} has entries:
    #   M[i,j,k] = 1 if we interpret it as the coefficient of A[i,j]*B[j,k] in C[i,k]
    #
    # This tensor is NOT all ones in a simple tensor product sense. It's the
    # "identity" tensor in the sense that it picks out the j-index that links A and B.
    #
    # Let's use a different approach: we can decompose the matmul as a sum of
    # rank-1 tensors by exploiting linearity in each factor.
    #
    # For the 2x3 * 3x5 case:
    # C[0,k] = A[0,0]*B[0,k] + A[0,1]*B[1,k] + A[0,2]*B[2,k]
    # C[1,k] = A[1,0]*B[0,k] + A[1,1]*B[1,k] + A[1,2]*B[2,k]
    #
    # Each output C[i,k] is a sum of m=3 products. There are n*p = 10 outputs.
    # So schoolbook gives 10*3 = 30 terms.
    #
    # Can we share computations? The key is that B[j,k] appears in multiple
    # output equations. Specifically, B[0,k] appears in C[0,k] and C[1,k].
    #
    # Let's try to find a lower rank by solving the multilinear system.
    # We'll use a greedy approach: try rank r from a lower bound upward,
    # and solve for U, V, W using iterative linearization.
    #
    # Lower bound: The tensor rank is at least max(rank of any flattening).
    # Flattening (i,j) x k: 6x5 matrix with all 1s -> rank 1
    # Flattening i x (j,k): 2x15 matrix with all 1s -> rank 1  
    # Flattening (i,k) x j: 10x3 matrix with all 1s -> rank 1
    # So lower bound is 1, which is very weak.
    #
    # A better lower bound: the tensor rank of M_{n,m,p} is at least 
    # max(n*m, m*p, p*n) = max(6,15,10) = 15? No, that's not right either.
    #
    # Actually, for the standard matmul tensor, it is known that the rank
    # equals n*m*p for most small cases, but there are exceptions.
    #
    # Let's try to construct a rank-15 decomposition by exploiting the
    # structure that B has 15 entries and A has 6 entries.
    #
    # Alternative: Use the fact that M_{2,3,5} can be embedded in a larger
    # structure. For example, M_{2,3,5} is a sub-tensor of M_{2,3,3} ⊗ M_{1,1,2}
    # or similar. But this is getting complicated.
    #
    # Let's try a direct approach: assume rank r and solve.
    # We'll use the method of "fixing one matrix and solving for the others."
    #
    # Strategy: 
    # 1. Initialize W randomly (or with a structured pattern)
    # 2. For each column c, we need U[:,c] ⊗ V[:,c] ⊗ W[:,c] to contribute
    #    to the tensor.
    # 3. The constraint is: for all (i,j,k), sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] = 1
    #
    # This is a system of 30 equations in r*31 unknowns (but with multilinear structure).
    #
    # Let's try a structured initialization:
    # Set W[k*n+i, c] = 1 for c = k*n+i (i.e., W is identity-like)
    # Then the constraint becomes: sum_c U[i*m+j,c]*V[j*p+k,c]*delta_{c, k*n+i} = 1
    # Which means: U[i*m+j, k*n+i] * V[j*p+k, k*n+i] = 1 for all (i,j,k)
    #
    # This requires U[i*m+j, k*n+i] = 1/V[j*p+k, k*n+i] for all (i,j,k)
    # But U[i*m+j, c] must be consistent across different k for the same (i,j).
    # For fixed (i,j), U[i*m+j, c] appears in constraints for all k.
    # U[i*m+j, k*n+i] * V[j*p+k, k*n+i] = 1 for each k.
    # So U[i*m+j, k*n+i] = 1/V[j*p+k, k*n+i]
    #
    # If we set V[j*p+k, c] = 1 for all j,k,c, then U[i*m+j, c] = 1 for all i,j,c.
    # But then the decomposition is: sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c]
    # = sum_c 1*1*W[k*n+i,c]
    # = sum_c W[k*n+i,c]
    # This must equal 1 for all (i,j,k), but it doesn't depend on j!
    # So this doesn't work unless the sum is 1 and independent of j, which it is,
    # but we need it to be 1 for each (i,j,k). Since it's independent of j,
    # we just need sum_c W[k*n+i,c] = 1 for all (i,k).
    #
    # With W being identity-like (W[c,c]=1, W[c,d]=0 for c≠d), sum_c W[k*n+i,c] = 1.
    # So this works! But the rank is W_dim = 10.
    #
    # Wait, let me check: if W is the identity matrix of size 10x10, and U and V
    # are all-ones matrices, then:
    # sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] = sum_c 1*1*delta_{c,k*n+i} = 1
    # Yes! This gives a rank-10 decomposition!
    #
    # Let me verify: U is 6x10 all ones, V is 15x10 all ones, W is 10x10 identity.
    # For any (i,j,k): sum_{c=0}^{9} 1*1*W[k*2+i,c] = W[k*2+i, k*2+i] = 1. ✓
    #
    # So rank 10 works! That's a huge improvement over 30.
    
    rank = W_dim  # 10
    U = [[1] * rank for _ in range(U_dim)]
    V = [[1] * rank for _ in range(V_dim)]
    W = [[0] * rank for _ in range(W_dim)]
    for c in range(rank):
        W[c][c] = 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
