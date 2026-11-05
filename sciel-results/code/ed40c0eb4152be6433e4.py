"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [3, 3, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 3, 3, 7
    deadline = time.time() + 135  # safety margin before 150s limit
    
    # Build the matrix multiplication tensor T[i, j, k] = delta_{j,k} for the (n,m,p) case
    # Actually, the tensor for matrix multiplication C = A @ B where A is n x m and B is m x p:
    # C[i, k] = sum_j A[i, j] * B[j, k]
    # The tensor T has T[i, j, k] = 1 if and only if this is a valid term, else 0.
    # In the decomposition, we need sum_c U[i,j,c] * V[j,k,c] * W[k,i,c] = T[i,j,k]
    # Wait, the standard form is: C[i,k] = sum_j A[i,j] B[j,k] = sum_c (sum_j U[i,j,c] A[i,j]) (sum_k V[j,k,c] B[j,k]) ... no.
    # The correct tensor decomposition form is:
    # C[i,k] = sum_c U_{i,k,c} * (sum_j U'_{i,j,c} A[i,j]) ... 
    # Let me use the standard: A[i,j] * B[j,k] coefficient is T[i,j,k] = delta_{jk}... no, T[i,j,k]=1 for all i,j,k is wrong.
    # T[i,j,k] = 1 if the term A[i,j]*B[j,k] appears in C[i,k], which is always true.
    # So T is the all-ones tensor of shape (n, m, p)? No!
    # C[i,k] = sum_j A[i,j] B[j,k]. The coefficient of A[i,j] B[j,k] in C[i,k] is 1.
    # The tensor T_{i,j,k} for the multiplication is such that T_{i,j,k} = 1.
    # Wait, that's not right either. The tensor is defined by:
    # <T, A ⊗ B> = C, where C[i,k] = sum_j A[i,j] B[j,k]
    # So T[i,j,k] = 1 for all i,j,k? No, T is a 3-tensor with T[i,j,k] = 1.
    # Actually, the bilinear form is sum_{i,j,k} T[i,j,k] A[i,j] B[j,k] = sum_{i,k} C[i,k] ... 
    # Let me just build it directly. T[i,j,k] = 1 if the product A[i,j]*B[j,k] contributes to C[i,k].
    # It always does, so T[i,j,k] = 1 for all i,j,k.
    # But that's the all-ones tensor, which has rank 1, which is clearly wrong.
    
    # Let me reconsider. The matrix multiplication tensor T is in V^* ⊗ V^* ⊗ V where
    # the three spaces are indexed by (i,j), (j,k), (i,k) respectively.
    # T is defined by T[(i,j), (j,k), (i,k)] = 1 if the indices match, else 0.
    # But (i,j) and (j,k) must share j, and (i,k) is determined by i,k.
    # So T has shape (n*m, m*p, p*n) and T[(i,j), (j,k), (k,n_idx)] = 1 
    # where the third index is (k, i) in the trace convention W[k*n+i].
    # So T[a, b, c] = 1 where a = i*m+j, b = j*p+k, c = k*n+i.
    # This is nonzero only when the indices are consistent.
    
    # For the decomposition: C[i,k] = sum_c U[i*m+j... 
    # The standard form is:
    # For each rank term c, we have matrices A_c (n x m), B_c (m x p), C_c (n x p)
    # such that A[i,j] * B[j,k] * ... no.
    # The correct form: C = sum_c (A_c * diag(x_c) * B_c) where x_c are the "intermediates"
    # Or equivalently: C[i,k] = sum_c sum_j U_{i,j,c} * A[i,j] * V_{j,k,c} * B[j,k] ... 
    # I'm overcomplicating. Let me use the standard tensor rank definition.
    
    # The matrix multiplication tensor T ∈ R^{n*m} ⊗ R^{m*p} ⊗ R^{p*n}
    # T[(i,j), (j,k), (k,i)] = 1 (using 0-indexed: T[i*m+j, j*p+k, k*n+i] = 1)
    # All other entries are 0.
    
    # We want to write T = sum_{c=1}^{rank} u_c ⊗ v_c ⊗ w_c
    # where u_c ∈ R^{n*m}, v_c ∈ R^{m*p}, w_c ∈ R^{p*n}
    
    # The schoolbook decomposition uses rank = n*m*p:
    # For each (i,j,k), u_{ijk} has 1 at position i*m+j, v_{ijk} has 1 at position j*p+k,
    # w_{ijk} has 1 at position k*n+i.
    
    # This is correct. Now, can we do better?
    # For 3x3x3, the rank is at most 23 (Bini et al.), and it's known to be exactly 23 or less.
    # For 3x3x7, we can potentially use a decomposition that handles the 3x3x3 part efficiently
    # and the remaining columns separately.
    
    # Let me try a hybrid approach:
    # - Use a known low-rank decomposition for the 3x3x3 part (if I can encode it)
    # - Use schoolbook for the remaining 4 columns
    
    # Actually, I don't have a reliable 3x3x3 rank-23 decomposition encoded.
    # Let me try a different approach: use Strassen-like ideas.
    
    # For 3x3 matrix multiplication, one can partition into 2x2 blocks and use Strassen.
    # A = [[A11, a12], [a21, A22]] where A11 is 2x2, a12 is 2x1, a21 is 1x2, A22 is 1x1
    # This is messy.
    
    # Let me try yet another approach: since we have 135 seconds, I can try to use
    # numerical methods to find a low-rank decomposition.
    
    # Build the tensor T
    T = np.zeros((n*m, m*p, p*n), dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m+j, j*p+k, k*n+i] = 1.0
    
    # Reshape to a matrix of size (n*m) x (m*p * p*n)
    T_mat = T.reshape(n*m, m*p * p*n)
    
    # The rank of T_mat gives a lower bound, but we want the tensor rank.
    # Tensor rank is NP-hard in general, but for small tensors we can try.
    
    # For now, let me try a greedy approach or just use schoolbook.
    # Given time constraints, I'll use schoolbook which is guaranteed to work.
    
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
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
