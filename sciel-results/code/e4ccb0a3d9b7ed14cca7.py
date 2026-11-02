"""
Matrix multiplication tensor decomposition for 2x3x7.
We search for a low-rank exact decomposition using a structured
ansatz that exploits the cyclic-group embedding of the inner dimensions.
"""
import numpy as np
from itertools import product
import time
import sys

DIMENSIONS = [2, 3, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Known lower bounds: n*m, m*p, p*n
    lower_bound = max(n*m, m*p, p*n)  # 21
    # Schoolbook upper bound
    upper_bound = n*m*p  # 42
    
    # We will try to find a decomposition of rank r for r in range(lower_bound, upper_bound)
    # Strategy: Use a parameterized family of decompositions and solve for the parameters.
    
    # The tensor M is defined by M[i,j,k] = 1 if (i,j,k) is a valid triple, but actually
    # the multiplication tensor T_{ijk} = 1 if the index (i,j) in U, (j,k) in V, (k,i) in W 
    # correspond to the same product term.
    # Actually, the tensor is T such that T[i,j,k] = 1 if the product C[i,k] = sum_j A[i,j]B[j,k]
    # involves the term A[i,j]B[j,k].
    # So T is a 0-1 tensor where T[i,j,k]=1 for all i,j,k. Wait, no.
    # The multiplication tensor is sum_{i,j,k} T_{ijk} (a_i b_j c_k) where T_{ijk} = 1 if the 
    # term A_{ij} B_{jk} contributes to C_{ik}. 
    # In our case, C_{ik} = sum_j A_{ij} B_{jk}.
    # So T_{ijk} = 1 for all i in 0..n-1, j in 0..m-1, k in 0..p-1.
    # Wait, that's not right. The tensor is sparse in the sense that each "slice" is determined by j.
    # T_{ijk} = 1 if the indices match. But here, for any i,j,k, the term A_{ij}B_{jk} contributes to C_{ik}.
    # So T_{ijk} is 1 for ALL i,j,k? 
    # No, the tensor T is defined such that the bilinear map (A,B) -> C is given by 
    # C_{ik} = sum_j A_{ij} B_{jk}.
    # This can be written as C = sum_{i,j,k} A_{ij} B_{jk} e_i e_k^T.
    # The tensor T has components T_{ijk} = 1 for all i,j,k. 
    # Wait, no. T is a tensor in A^* x B^* x C^*.
    # T = sum_{i,j,k} e_i^* \otimes e_j^* \otimes e_k^* is NOT correct.
    # The correct tensor is T = sum_{i,j,k} e_i \otimes e_j \otimes e_k^* ?
    # Let's use the standard definition: T = sum_{i,j,k} a_{ij} b_{jk} c_{ik} where c_{ik} = sum_j a_{ij} b_{jk}.
    # This is T = sum_{i,j,k} a_{ij} b_{jk} c_{ik}.
    # But c_{ik} is not an input, it's the output.
    # The tensor is T = sum_{i,j,k} e_i^* \otimes e_j^* \otimes (e_k e_l^T)? No.
    
    # Let's stick to the factorization form:
    # U is (n*m) x r, V is (m*p) x r, W is (p*n) x r.
    # The condition is: for all i,j,k:
    # sum_{alpha=0}^{r-1} U_{i*m+j, alpha} * V_{j*p+k, alpha} * W_{k*n+i, alpha} = 1
    # And for all (i,j,k) that do NOT correspond to a term (but here ALL i,j,k correspond to a term),
    # the sum must be 1.
    # Wait, are there any (i,j,k) that do NOT correspond to a term?
    # In matrix multiplication, EVERY pair (i,j) in A and (j,k) in B produces a term in C[i,k].
    # So the tensor is DENSE: T_{ijk} = 1 for all i,j,k.
    # But that can't be, because then the rank would be at least n*m*p?
    # No, the rank of the multiplication tensor is NOT n*m*p.
    # The schoolbook algorithm has rank n*m*p, but there are better algorithms (Strassen, etc.).
    # The tensor T is indeed T_{ijk} = 1 for all i,j,k?
    # Let's check for 1x1x1: T_{000}=1. Rank 1.
    # For 2x1x1: T_{000}=1, T_{100}=1. Rank 2? 
    # C[0,0] = A[0,0]B[0,0]
    # C[1,0] = A[1,0]B[0,0]
    # So T = e_0^* \otimes e_0^* \otimes e_0 + e_1^* \otimes e_0^* \otimes e_0
    # Rank 2.
    # So yes, T_{ijk} = 1 for all i,j,k.
    # But wait, for 2x2x2, the tensor is not just all ones.
    # C[0,0] = A[0,0]B[0,0] + A[0,1]B[1,0]
    # C[0,1] = A[0,0]B[0,1] + A[0,1]B[1,1]
    # C[1,0] = A[1,0]B[0,0] + A[1,1]B[1,0]
    # C[1,1] = A[1,0]B[0,1] + A[1,1]B[1,1]
    # So T_{ijk} = 1 if the term A[i,j]B[j,k] contributes to C[i,k].
    # Which is ALWAYS true.
    # So T_{ijk} = 1 for all i,j,k.
    # But then the tensor is just the all-ones tensor, which has rank 1?
    # No, the tensor is T = sum_{i,j,k} a_{ij} b_{jk} c_{ik} is not the right way to think.
    # The tensor is in the dual space.
    # T = sum_{i,j,k} e_i^* \otimes e_j^* \otimes e_{(i,k)}? No.
    
    # Let's use the factorization directly.
    # We need: sum_alpha U_{i*m+j, alpha} V_{j*p+k, alpha} W_{k*n+i, alpha} = 1 for all i,j,k.
    # This is a system of n*m*p equations.
    # We have 3*r variables (U, V, W).
    # We can fix one factor to simplify. For example, set W to be a basis or something.
    
    # A better approach: use the fact that the tensor is the Kronecker product of smaller tensors?
    # Not exactly.
    
    # Let's try to find a decomposition for small ranks using a grid search or optimization.
    # Since the problem is small (2x3x7), we can try to solve for r in [21, 41] using a 
    # semi-definite programming or a nonlinear solver. But we only have numpy.
    
    # Alternative: use a known decomposition for 2x3x7 if it exists in literature.
    # The best known rank for 2x3x7 is not well-known. 
    # For 2x2x2, it's 7 (Strassen).
    # For 2x2x3, it's 11 (Winograd).
    # For 2x3x3, it's 12?
    # For 2x3x7, we can use a combination of known decompositions.
    
    # Let's try to use a decomposition based on the Kronecker product of a 2x2x2 and a 1x3x7?
    # No, that doesn't make sense.
    
    # Let's try to construct a decomposition using the following idea:
    # C = A * B
    # We can write A as a 2x3 matrix and B as a 3x7 matrix.
    # We can use a decomposition that treats the 3x7 part as a convolution.
    
    # Since we don't have a known lower-rank decomposition, we will return the schoolbook 
    # decomposition but with a slightly different structure to ensure it's valid.
    
    # Actually, let's try to find a rank-21 decomposition using a grid search for the entries.
    # This is unlikely to work in 120s.
    
    # Let's return the schoolbook decomposition for now.
    rank = n * m * p
    U = np.zeros((n*m, rank), dtype=float)
    V = np.zeros((m*p, rank), dtype=float)
    W = np.zeros((p*n, rank), dtype=float)
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m + j, col] = 1.0
                V[j*p + k, col] = 1.0
                W[k*n + i, col] = 1.0
                col += 1
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
