"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np

DIMENSIONS = [2, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin before 150s kill

    n, m, p = DIMENSIONS
    # Schoolbook rank
    rank_school = n * m * p  # 48

    # We want to find a lower rank decomposition.
    # The tensor is T[i,j,k] = e_{i,j} ⊗ e_{j,k} ⊗ e_{k,i} (with appropriate indices)
    # Actually, the matrix multiplication tensor T has entries T[i,j,k] where:
    # U[i*m+j, r], V[j*p+k, r], W[k*n+i, r]
    # The condition is: for all i,j,k, sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
    # And for all other combinations (i',j',k') where (i',j',k') != (i,j,k) in the "diagonal" sense... 
    # Wait, the matrix multiplication tensor is:
    # C[i,k] = sum_j A[i,j] B[j,k]
    # So the tensor T[i,j,k] = 1 if we're computing C[i,k] from A[i,j] and B[j,k]
    # The standard tensor is T[i,j,k] = delta_{i,i'} delta_{j,j'} delta_{k,k'} ... no.
    # Let's think carefully.
    # A is n x m, B is m x p, C is n x p.
    # C[i,k] = sum_{j=1}^m A[i,j] B[j,k]
    # The multiplication tensor T is a 3-tensor of dimensions n x m x p:
    # T[i,j,k] = 1 for all i,j,k? No.
    # Actually, the tensor for matrix multiplication is:
    # T[i,j,k] = 1 if the entry (i,j) of A and (j,k) of B contribute to (i,k) of C.
    # So T[i,j,k] = 1 for all valid i,j,k. That's just the all-ones tensor? No.
    # Let me re-read the task.
    # "exactly decompose the matrix multiplication tensor"
    # The matrix multiplication tensor T has T[i,j,k] = 1 for all i in [n], j in [m], k in [p].
    # Wait, that can't be right because then the rank would be 1.
    # 
    # Actually, the matrix multiplication tensor is defined as:
    # T = sum_{i,j,k} e_i ⊗ e_j ⊗ e_k * (contribution to C[i,k])
    # C[i,k] = sum_j A[i,j] B[j,k]
    # So the bilinear form is: sum_{i,j,k} C[i,k] A[i,j] B[j,k] = sum_{i,j,k} (sum_{j'} A[i,j'] B[j',k]) A[i,j] B[j,k]
    # This doesn't make sense as a tensor decomposition.
    #
    # The standard formulation: The matrix multiplication tensor T_{n,m,p} is a tensor in F^n ⊗ F^m ⊗ F^p
    # defined by T_{i,j,k} = 1 if i,j,k are such that the (i,j) entry of A and (j,k) entry of B
    # multiply to contribute to the (i,k) entry of C.
    # So T_{i,j,k} = 1 for all i ∈ {1,...,n}, j ∈ {1,...,m}, k ∈ {1,...,p}.
    # But that's the all-ones tensor, which has rank 1.
    #
    # I think I'm confusing myself. Let me look at the schoolbook decomposition in the code.
    # U[i*m+j, column] = 1, V[j*p+k, column] = 1, W[k*n+i, column] = 1
    # So for each (i,j,k), we have a rank-1 term: e_{i*m+j} ⊗ e_{j*p+k} ⊗ e_{k*n+i}
    # The sum of these over all (i,j,k) gives the tensor.
    # So the tensor T has T[i*m+j, j*p+k, k*n+i] = 1 for all i,j,k, and 0 otherwise?
    # No, the tensor is in F^{n*m} ⊗ F^{m*p} ⊗ F^{p*n}.
    # The entry T[a,b,c] = sum_r U[a,r] V[b,r] W[c,r]
    # For the schoolbook: T[i*m+j, j*p+k, k*n+i] = 1 for all i,j,k.
    # And T[a,b,c] = 0 for all other (a,b,c).
    #
    # So the tensor is: T[a,b,c] = 1 if a = i*m+j, b = j*p+k, c = k*n+i for some i,j,k, else 0.
    # This is the "matrix multiplication tensor" in the sense that it encodes the multiplication.
    #
    # Now, we want to find a lower rank decomposition.
    # The rank of this tensor is what we want to minimize.
    #
    # For n=2, m=4, p=6, the schoolbook rank is 48.
    # We want to find if there's a lower rank.
    #
    # Let's try a simple approach: use the fact that the tensor has a lot of structure.
    # The tensor T is defined by: T[i*m+j, j*p+k, k*n+i] = 1
    # This can be written as: T = sum_{i,j,k} e_{i*m+j} ⊗ e_{j*p+k} ⊗ e_{k*n+i}
    #
    # One idea: group terms by i. For fixed i, we have sum_{j,k} e_{i*m+j} ⊗ e_{j*p+k} ⊗ e_{k*n+i}
    # = sum_{j,k} e_{i*m+j} ⊗ e_{j*p+k} ⊗ e_{k*n+i}
    # Note that e_{k*n+i} depends on k and i.
    #
    # Another idea: use the fact that the tensor is a "slice" of a larger structure.
    #
    # Let's try to use a numerical optimization approach to find a low-rank decomposition.
    # We can try to minimize the rank by using a soft rank minimization or by trying specific ranks.

    # Let's try to verify a decomposition of rank r.
    # We need to find U (n*m x r), V (m*p x r), W (p*n x r) such that
    # sum_r U[a,r] V[b,r] W[c,r] = T[a,b,c] for all a,b,c.
    
    # The tensor T has T[i*m+j, j*p+k, k*n+i] = 1 for all i,j,k.
    # Let's list the non-zero entries.
    # There are n*m*p = 48 non-zero entries, each equal to 1.
    
    # Let's try to find a decomposition of rank < 48.
    # We can use a simple iterative method or try to exploit structure.
    
    # One approach: use the fact that the tensor is separable in some way.
    # T[i*m+j, j*p+k, k*n+i] = 1
    # Let's write this as: for each (i,j,k), the term is e_{i*m+j} ⊗ e_{j*p+k} ⊗ e_{k*n+i}
    
    # Can we combine terms? For example, if we fix i and vary j,k:
    # sum_{j,k} e_{i*m+j} ⊗ e_{j*p+k} ⊗ e_{k*n+i}
    # = sum_j (e_{i*m+j} ⊗ (sum_k e_{j*p+k} ⊗ e_{k*n+i}))
    # The inner sum over k: sum_k e_{j*p+k} ⊗ e_{k*n+i}
    # This is a vector in F^{m*p} ⊗ F^{p*n}... no, it's a matrix.
    
    # Let's try a different approach: use a greedy or optimization method.
    # Since the time limit is tight, let's try a few specific strategies.
    
    # Strategy 1: Try to find a decomposition using a small number of random vectors.
    # Strategy 2: Use the structure to find a decomposition.
    
    # Let's first try to see if rank 24 or so is possible by using a simple heuristic.
    
    # Actually, let's just try to improve the schoolbook by merging terms.
    # For example, if two terms share the same U index, we can potentially combine them.
    
    # Let's try a simple approach: use the fact that the tensor is a sum of rank-1 terms,
    # and try to find linear dependencies among the terms.
    
    # The terms are: for each (i,j,k), t_{i,j,k} = e_{i*m+j} ⊗ e_{j*p+k} ⊗ e_{k*n+i}
    # We want to find a subset of these terms (or linear combinations) that span the same tensor.
    
    # Let's try to use a numerical rank finding approach.
    # We can form a matrix whose columns are the "flattened" rank-1 terms and find its rank.
    
    # Flatten each rank-1 term into a vector of length (n*m)*(m*p)*(p*n).
    # But that's 48*24*12 = 13824 dimensions, which is manageable.
    
    # Actually, the tensor has dimensions (n*m) x (m*p) x (p*n) = 48 x 24 x 12.
    # The total number of entries is 48*24*12 = 13824.
    # Only 48 of them are non-zero (equal to 1).
    
    # Let's form a matrix M where each column corresponds to a rank-1 term,
    # and the entries are the tensor values.
    # M has shape (13824, 48).
    # The rank of M is the rank of the tensor (in the sense of the span of the rank-1 terms).
    # But we want the tensor rank, which is the minimum number of rank-1 terms needed.
    
    # Actually, the tensor rank is the minimum r such that T = sum_{r'=1}^r u_{r'} ⊗ v_{r'} ⊗ w_{r'}.
    # This is not the same as the matrix rank of the flattened tensor.
    
    # Let's just try to use a simple optimization to find a low-rank decomposition.
    # We'll try ranks from 47 down to some lower bound.
    
    # For a given rank r, we need to solve:
    # sum_{r'=1}^r U[:,r'] ⊙ V[:,r'] ⊙ W[:,r'] = T
    # This is a system of equations.
    
    # Let's try a simple approach: use random initialization and gradient descent.
    # But this might be too slow.
    
    # Let's try a different approach: exploit the structure.
    # T[i*m+j, j*p+k, k*n+i] = 1
    # Note that the indices are coupled: j appears in all three, i appears in first and third, k appears in second and third.
    
    # Let's try to write T as a sum of terms that factor in a different way.
    # For example, can we write T as sum_i (something depending on i) ⊗ (something) ⊗ (something)?
    
    # For fixed i, the slice T[i*m+j, j*p+k, k*n+i] = 1 for all j,k.
    # This slice is: sum_{j,k} e_{i*m+j} ⊗ e_{j*p+k} ⊗ e_{k*n+i}
    # = (sum_j e_{i*m+j} ⊗ e_{j*p}) ⊗ ... no, this doesn't factor simply.
    
    # Let's try: for fixed i, define A_i = sum_j e_{i*m+j} ⊗ e_{j*p} (a vector in F^{n*m} ⊗ F^{m*p})
    # and B_i = sum_k e_{k*n+i} (a vector in F^{p*n})
    # But A_i ⊗ B_i = sum_{j,k} e_{i*m+j} ⊗ e_{j*p+k} ⊗ e_{k*n+i}
    # Wait, e_{j*p} is not the same as e_{j*p+k}.
    
    # Let's be more careful.
    # For fixed i, the slice is: sum_{j,k} e_{i*m+j} ⊗ e_{j*p+k} ⊗ e_{k*n+i}
    # Let's write e_{j*p+k} = sum_k e_{j*p+k} (this is just the basis vector).
    # Actually, for fixed j, sum_k e_{j*p+k} is a vector in F^{m*p} with 1s in positions j*p, j*p+1, ..., j*p+p-1.
    # And for fixed i, sum_k e_{k*n+i} is a vector in F^{p*n} with 1s in positions i, i+n, i+2n, ..., i+(p-1)n.
    
    # Let u_j = sum_k e_{j*p+k} (a vector in F^{m*p})
    # Let v_i = sum_k e_{k*n+i} (a vector in F^{p*n})
    # Then for fixed i,j: e_{i*m+j} ⊗ u_j ⊗ v_i = sum_k e_{i*m+j} ⊗ e_{j*p+k} ⊗ e_{k*n+i}
    # So T = sum_{i,j} e_{i*m+j} ⊗ u_j ⊗ v_i
    
    # This gives a decomposition of rank n*m = 8!
    # Let's verify:
    # T[a,b,c] = sum_{i,j} (e_{i*m+j})[a] * u_j[b] * v_i[c]
    # = sum_{i,j} delta_{a, i*m+j} * u_j[b] * v_i[c]
    # For a = i*m+j, this is sum_j u_j[b] * v_i[c]
    # u_j[b] = 1 if b is in [j*p, (j+1)*p), else 0
    # v_i[c] = 1 if c ≡ i (mod n), else 0
    # So T[i*m+j, b, c] = sum_j u_j[b] * v_i[c]
    # But we need T[i*m+j, b, c] = 1 if b = j*p+k and c = k*n+i for some k, else 0.
    # 
    # Let's check: if b = j*p+k and c = k*n+i, then u_j[b] = 1 and v_i[c] = 1, so the sum is 1 (since only j' = j contributes to u_{j'}[b]).
    # If b != j*p+k for any k, then u_j[b] = 0, so the sum is 0.
    # If c != k*n+i for any k, then v_i[c] = 0, so the sum is 0.
    # 
    # Wait, but what if b is in a different block? Say b = j'*p+k' with j' != j.
    # Then u_j[b] = 0, so no contribution.
    # 
    # So it seems like T[i*m+j, b, c] = 1 if b = j*p+k and c = k*n+i for some k, else 0.
    # This matches the original tensor!
    # 
    # So we have a decomposition of rank n*m = 8.
    # 
    # Let's write it out:
    # For each i in range(n), j in range(m):
    #   U[i*m+j, r] = 1 where r = i*m+j
    #   V[b, r] = u_j[b] = 1 if b in [j*p, (j+1)*p) else 0
    #   W[c, r] = v_i[c] = 1 if c % n == i else 0
    
    rank = n * m
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for i in range(n):
        for j in range(m):
            r = i * m + j
            U[i * m + j][r] = 1
            for k in range(p):
                V[j * p + k][r] = 1
            for k in range(p):
                W[k * n + i][r] = 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
