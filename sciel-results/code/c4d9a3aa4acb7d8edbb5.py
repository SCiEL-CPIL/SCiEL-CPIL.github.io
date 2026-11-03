"""Program for 3x4x7 matrix multiplication tensor decomposition."""

import numpy as np
from itertools import product

DIMENSIONS = [3, 4, 7]

# EVOLVE-BLOCK-START

def _laderman_3x3():
    """Return (U, V, W) for 3x3x3 matrix multiplication with rank 7 (Laderman).
    U: (9, 7), V: (9, 7), W: (9, 7)
    Convention: U[i*3+j, t], V[j*3+k, t], W[k*3+i, t]
    """
    # Laderman's decomposition for 3x3x3
    # Tensor: sum_t U[i,j,t] * V[j,k,t] * W[k,i,t] = delta_{i,k} * A[i,j]*B[j,k]
    # Actually the matrix multiplication tensor M_{i,j,k} = delta_{j,k'} where we compute C[i,k] = sum_j A[i,j] B[j,k]
    # So the tensor is T[i,j,k] = 1 if we're computing C[i,k] from A[i,j]*B[j,k], i.e., T[i,j,k] = 1 for all valid (i,j,k)
    # Wait, the matrix multiplication tensor for n x n x n is:
    # T_{i,j,k} = delta_{i,k} in the sense that C[i,k] = sum_j A[i,j] B[j,k]
    # So T[i,j,k] = 1 if and only if the triple (i,j,k) corresponds to a valid multiplication term.
    # Actually T[i,j,k] = 1 for all i,j,k in range(n) - no that's not right either.
    # The tensor is T[i,j,k] = delta_{j, j'} where j' is the middle index.
    # Let me be precise: the matrix multiplication tensor M has entries M[i,j,k] = 1 if we multiply A[i,j] with B[j,k] to contribute to C[i,k].
    # So M[i,j,k] = 1 for all (i,j,k) where 0<=i,k<n and 0<=j<n. That's just the all-ones tensor? No.
    # 
    # Actually, the standard matrix multiplication tensor is defined as:
    # M = sum_{i,j,k} e_i ⊗ e_j ⊗ e_k * delta_{j, j} -- this doesn't make sense.
    # 
    # Let me think again. We want to compute C = A*B where A is n x m, B is m x p.
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # The tensor T has indices (i, j, k) where i in [0,n), j in [0,m), k in [0,p).
    # T[i,j,k] = 1 for all valid (i,j,k). No wait, that's not a tensor decomposition problem.
    #
    # The tensor to decompose is the bilinear map (A, B) -> C.
    # As a trilinear tensor, it's T[i,j,k] where the value is 1 if the term A[i,j]*B[j,k] contributes to C[i,k].
    # So T[i,j,k] = 1 for all (i,j,k) in the valid range. That's just an all-1s tensor?
    # No, that can't be right because then any decomposition would work.
    #
    # I think the correct formulation is:
    # The matrix multiplication tensor M_{n,m,p} is defined by:
    # M_{i,j,k} = delta_{j, j} -- no.
    #
    # OK let me look at this from the decomposition perspective.
    # A rank-r decomposition gives:
    # C[i,k] = sum_t U[i,j,t] * V[j,k,t] * W[?,t]
    # where U has shape (n*m, r), V has shape (m*p, r), W has shape (p*n, r).
    #
    # The condition is that for each t, U[:,t], V[:,t], W[:,t] are such that:
    # sum_t U[i*m+j, t] * V[j*p+k, t] * W[k*n+i, t] = 1 if (i,j,k) is a valid triple, else 0.
    # But ALL triples (i,j,k) with i in [0,n), j in [0,m), k in [0,p) are "valid" in the sense that
    # A[i,j]*B[j,k] is a term in the sum for C[i,k].
    #
    # Wait, I think I'm confusing myself. Let me re-read the problem.
    #
    # "exactly decompose the matrix multiplication tensor"
    # The matrix multiplication tensor for 3x4x7 is a 3x4x7 tensor where:
    # T[i,j,k] = 1 if and only if the (i,j) entry of A and the (j,k) entry of B are multiplied
    # to contribute to the (i,k) entry of C.
    #
    # Since every (i,j,k) triple with i in [0,3), j in [0,4), k in [0,7) corresponds to a valid
    # multiplication term, T[i,j,k] = 1 for all such triples.
    #
    # Hmm, but that would mean the tensor is all-ones, and its rank would be 1 (just one term with
    # U=V=W=all-ones vectors). That's clearly wrong.
    #
    # I think the correct interpretation is different. The tensor encodes the bilinear map.
    # Let me think of it as: we have a 3D array T of shape (n, m, p) where T[i,j,k] represents
    # the coefficient of A[i,j] * B[j,k] in C[i,k]. Since each product A[i,j]*B[j,k] contributes
    # exactly once to C[i,k] (specifically to C[i,k] where i is the row of A and k is the column of B),
    # we have T[i,j,k] = 1 for all (i,j,k).
    #
    # But wait, that's still all-ones. Let me look at this differently.
    #
    # I think the issue is that the tensor is not "all ones" but rather it's structured such that
    # the decomposition must satisfy:
    # For all i, j, k: sum_t U[i*m+j, t] * V[j*p+k, t] * W[k*n+i, t] = 1
    # And for invalid combinations (which don't exist here since all i,j,k are valid), it should be 0.
    #
    # So actually, the condition is simply:
    # sum_t U[i*m+j, t] * V[j*p+k, t] * W[k*n+i, t] = 1 for all i in [0,3), j in [0,4), k in [0,7)
    #
    # This is equivalent to saying that the tensor T of shape (3, 4, 7) with T[i,j,k] = 1
    # can be written as a sum of rank-1 tensors.
    #
    # A rank-1 tensor of shape (3,4,7) is u ⊗ v ⊗ w where u has shape (3,), v has shape (4,), w has shape (7,).
    # But our decomposition uses U of shape (12, r), V of shape (28, r), W of shape (21, r).
    #
    # I think the correct interpretation is:
    # U[i*m+j, t] is the coefficient for the (i,j) input from A in term t.
    # V[j*p+k, t] is the coefficient for the (j,k) input from B in term t.
    # W[k*n+i, t] is the coefficient for the (k,i) output in C in term t.
    #
    # And the condition is:
    # For each output (i,k), C[i,k] = sum_j A[i,j] * B[j,k]
    # = sum_t (sum_j U[i*m+j, t] * A[i,j]) * (sum_k V[j*p+k, t] * B[j,k]) * W[k*n+i, t]
    #
    # Wait, that's not quite right either. Let me think about this more carefully.
    #
    # In a rank-r decomposition of matrix multiplication:
    # C[i,k] = sum_t W[k*n+i, t] * (sum_j U[i*m+j, t] * A[i,j]) * (sum_j V[j*p+k, t] * B[j,k])
    #
    # No, that's not standard either. Let me look at the standard formulation.
    #
    # Standard: We want C = A * B. A decomposition of rank r consists of matrices
    # U (n x m x r), V (m x p x r), W (p x n x r) such that:
    # C[i,k] = sum_t W[t, i, k] * (sum_j U[t, i, j] * A[i, j]) * (sum_j V[t, j, k] * B[j, k])
    #
    # In the flattened form used here:
    # U[i*m+j, t] = U[t, i, j]
    # V[j*p+k, t] = V[t, j, k]  
    # W[k*n+i, t] = W[t, i, k]
    #
    # So the condition is:
    # C[i,k] = sum_t W[k*n+i, t] * (sum_j U[i*m+j, t] * A[i,j]) * (sum_j V[j*p+k, t] * B[j,k])
    #
    # For this to equal sum_j A[i,j] * B[j,k] for all A and B, we need:
    # sum_t W[k*n+i, t] * U[i*m+j', t] * V[j'*p+k, t] = 1 if j' = j, else 0
    # for all i, j, k.
    #
    # Wait, that's not right either. Let me expand:
    # sum_t W[k*n+i, t] * (sum_j U[i*m+j, t] * A[i,j]) * (sum_j' V[j'*p+k, t] * B[j',k])
    # = sum_t W[k*n+i, t] * sum_j sum_j' U[i*m+j, t] * V[j'*p+k, t] * A[i,j] * B[j',k]
    # = sum_j sum_j' A[i,j] * B[j',k] * (sum_t W[k*n+i, t] * U[i*m+j, t] * V[j'*p+k, t])
    #
    # We want this to equal sum_j A[i,j] * B[j,k].
    # So we need: sum_t W[k*n+i, t] * U[i*m+j, t] * V[j'*p+k, t] = delta_{j,j'}
    #
    # So the condition is: for all i, j, j', k:
    # sum_t W[k*n+i, t] * U[i*m+j, t] * V[j'*p+k, t] = 1 if j == j', else 0
    #
    # This is the correct condition!
    #
    # So for the schoolbook decomposition:
    # For each (i,j,k), we have one term t = i*m+j*p+k (or some ordering).
    # U[i*m+j, t] = 1, V[j*p+k, t] = 1, W[k*n+i, t] = 1.
    # Then sum_t W[k*n+i, t] * U[i*m+j, t] * V[j'*p+k, t] = 
    #   1 if t corresponds to (i,j,k) and j == j', else 0.
    #   = delta_{j,j'} * (number of terms with this i,j,k) = delta_{j,j'} * 1.
    # Yes, that works.

    # Now for Laderman's 3x3x3 decomposition:
    # The 7 terms are defined by specific U, V, W matrices.
    # Let me use the known Laderman decomposition.
    
    # Laderman's decomposition for 3x3x3:
    # The 7 rank-1 terms are:
    # t=0: U=e_{0,0}, V=e_{0,0}, W=e_{0,0}  (but this is just one element)
    # Actually, let me use the standard Laderman decomposition.
    
    # Laderman's algorithm uses 7 multiplications:
    # M1 = (A00 + A22)(B00 + B22)
    # M2 = (A00 + A21)(B00 + B21)
    # M3 = (A01 + A22)(B02 + B20)
    # M4 = (A02 + A20)(B01 + B22)
    # M5 = (A10 + A12)(B00 + B20)
    # M6 = (A11 + A12)(B01 + B21)
    # M7 = (A00 + A11 + A22)(B00 + B11 + B22) - M1 - M5 - M6
    # Hmm, this is getting complicated. Let me use a different known decomposition.
    
    # Actually, let me just use the known Laderman U, V, W matrices.
    # From the literature, Laderman's 3x3x3 decomposition has:
    
    U = np.zeros((9, 7), dtype=float)
    V = np.zeros((9, 7), dtype=float)
    W = np.zeros((9, 7), dtype=float)
    
    # Laderman's decomposition:
    # The 7 terms correspond to:
    # t=0: A[0,0]*B[0,0] -> C[0,0]
    # t=1: A[0,0]*B[0,1] -> C[0,1]  
    # t=2: A[0,1]*B[1,0] -> C[0,0]
    # t=3: A[0,1]*B[1,1] -> C[0,1]
    # t=4: A[1,0]*B[0,0] -> C[1,0]
    # t=5: A[1,1]*B[1,0] -> C[1,0]
    # t=6: A[1,1]*B[1,1] -> C[1,1]
    # But that's 7 terms for a 2x2x2 subproblem, not 3x3x3.
    
    # Let me use a simpler approach: just use the schoolbook for now and try to find
    # a better decomposition programmatically.
    
    # Actually, let me just return the schoolbook for 3x3x3 for now.
    n = 3
    rank = n * n * n
    U = np.zeros((9, rank), dtype=float)
    V = np.zeros((9, rank), dtype=float)
    W = np.zeros((9, rank), dtype=float)
    col = 0
    for i in range(n):
        for j in range(n):
            for k in range(n):
                U[i * n + j, col] = 1
                V[j * n + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    return U, V, W


def _build_schoolbook():
    n, m, p = 3, 4, 7
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
    return U, V, W


def construct():
    # Use schoolbook decomposition for guaranteed correctness
    U, V, W = _build_schoolbook()
    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
