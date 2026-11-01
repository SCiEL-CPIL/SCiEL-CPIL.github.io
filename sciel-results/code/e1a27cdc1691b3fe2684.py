"""Matrix multiplication 2x2 by 2x6 - Lazard-Teitelbaum bilinear circuit synthesis."""
import numpy as np
from scipy import linalg

DIMENSIONS = [2, 2, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    # For 2x2 * 2x6 matrix multiplication, the tensor T is:
    # T[i*m+j, j*p+k, k*n+i] = 1 for all valid indices
    # All other entries are 0.
    
    # The schoolbook rank is n*m*p = 24.
    # We can do better by sharing multiplications across the k dimension.
    
    # Key insight: For fixed (i,j), the entries A[i,j] are multiplied with
    # B[j,0], B[j,1], ..., B[j,5]. The output C[i,k] involves A[i,0]*B[0,k] + A[i,1]*B[1,k].
    
    # Let's think of this differently. The bilinear map is:
    # C[i,k] = A[i,0]*B[0,k] + A[i,1]*B[1,k]
    
    # We can group by the A-indices. For each i, we need to compute:
    # C[i,:] = A[i,0] * B[0,:] + A[i,1] * B[1,:]
    
    # This is a linear combination of rows of B. The "multiplication" here is
    # scalar-vector multiplication, which can be done with 1 multiplication per scalar.
    
    # Actually, let's use the Lazard-Teitelbaum approach more carefully.
    # The idea is to find a rank decomposition of the tensor.
    
    # The tensor T has 24 non-zero entries (one for each (i,j,k) triple).
    # We want to write T = sum_r U_r ⊗ V_r ⊗ W_r with minimal r.
    
    # Let's try to find a decomposition with rank less than 24.
    
    # Approach: Use the fact that for each (i,k), the output C[i,k] depends on
    # A[i,0]*B[0,k] + A[i,1]*B[1,k]. We can use a 2x2 matrix multiplication trick.
    
    # For 2x2 * 2x1, the rank is 4 (schoolbook). But can we do better?
    # Actually, 2x2 * 2x1 is just two dot products of length 2, which is rank 4.
    
    # However, we can share across the k dimension. Let's define:
    # For each k, we compute C[:,k] = A @ B[:,k]
    
    # The key is that A is the same for all k. So we can precompute combinations of A.
    
    # Let's use a Strassen-like approach. Define:
    # M1 = A[0,0] + A[1,1]
    # M2 = A[1,0]
    # M3 = A[0,1]
    # M4 = A[0,0] + A[0,1]
    
    # Then for B[:,k] = [B[0,k], B[1,k]]:
    # C[0,k] = A[0,0]*B[0,k] + A[0,1]*B[1,k]
    # C[1,k] = A[1,0]*B[0,k] + A[1,1]*B[1,k]
    
    # This is 4 multiplications per k, so 4*6 = 24 total. Same as schoolbook.
    
    # Can we do better? Let's try:
    # P1 = (A[0,0] + A[1,1]) * (B[0,k] + B[1,k])
    # P2 = (A[0,0] - A[1,1]) * (B[0,k] - B[1,k])
    # P3 = A[0,1] * B[1,k]
    # P4 = A[1,0] * B[0,k]
    
    # Then:
    # A[0,0]*B[0,k] = (P1 + P2)/4 + (P1 - P2)/4... this doesn't work cleanly.
    
    # Let me try a different approach. The minimum rank for 2x2*2x6 is known to be 24
    # based on the fact that each output column requires a 2x2*2x1 multiplication of rank 4,
    # and these are independent.
    
    # However, let's verify this more carefully. The tensor T for M_{2,2,6} can be written as:
    # T = sum_{i=0}^{1} sum_{j=0}^{1} sum_{k=0}^{5} e_{i,j} ⊗ e_{j,k} ⊗ e_{i,k}
    
    # This is a sum of 24 rank-1 tensors. Can we combine them?
    
    # For fixed (i,j), we have sum_k e_{i,j} ⊗ e_{j,k} ⊗ e_{i,k} = e_{i,j} ⊗ (sum_k e_{j,k}) ⊗ (sum_k e_{i,k})
    # This is a single rank-1 tensor!
    
    # So T = sum_{i=0}^{1} sum_{j=0}^{1} e_{i,j} ⊗ (sum_k e_{j,k}) ⊗ (sum_k e_{i,k})
    # This is a sum of 4 rank-1 tensors!
    
    # Let's verify: For a given (i,j,k), the coefficient in the expansion is:
    # sum_{i',j'} [i==i' and j==j'] * 1 * 1 = 1 if i==i' and j==j', else 0.
    # But we need the coefficient to be 1 for all (i,j,k). This works!
    
    # Wait, let me be more careful. The tensor is:
    # T = sum_{i,j,k} e_{i,j} ⊗ e_{j,k} ⊗ e_{i,k}
    # = sum_i sum_j sum_k e_{i,j} ⊗ e_{j,k} ⊗ e_{i,k}
    # = sum_i sum_j (e_{i,j} ⊗ (sum_k e_{j,k}) ⊗ (sum_k e_{i,k}))
    
    # So T is a sum of 4 rank-1 tensors. The rank is at most 4.
    
    # Let's construct the decomposition:
    # For (i,j) in {(0,0), (0,1), (1,0), (1,1)}:
    # U_r[i*m+j] = 1, all other U_r entries = 0
    # V_r[j*p+k] = 1 for all k, all other V_r entries = 0
    # W_r[k*n+i] = 1 for all k, all other W_r entries = 0
    
    # Let's verify this gives the correct result.
    # C[i,k] = sum_r U_r[i*m+j_r] * V_r[j_r*p+k] * W_r[k*n+i]
    # where the sum is over r, and for each r, j_r is determined by r.
    
    # Actually, the formula is:
    # C[i,k] = sum_r (sum_{i',j'} U_r[i'*m+j'] * A[i',j']) * (sum_{j',k'} V_r[j'*p+k'] * B[j',k']) * W_r[k*n+i]
    
    # For r corresponding to (i_r, j_r):
    # sum_{i',j'} U_r[i'*m+j'] * A[i',j'] = A[i_r, j_r]
    # sum_{j',k'} V_r[j'*p+k'] * B[j',k'] = sum_{k'} B[j_r, k']
    # W_r[k*n+i] = 1 if i == i_r, else 0
    
    # So C[i,k] = sum_{r: i_r == i} A[i_r, j_r] * (sum_{k'} B[j_r, k'])
    # = sum_{j_r} A[i, j_r] * (sum_{k'} B[j_r, k'])
    
    # This is NOT sum_j A[i,j] * B[j,k]! It's sum_j A[i,j] * (sum_k B[j,k]).
    
    # So this decomposition is wrong. The issue is that the tensor decomposition
    # doesn't directly give the bilinear algorithm in the way I'm thinking.
    
    # Let me reconsider. The correct approach is to use the fact that the tensor
    # T = sum_{i,j,k} e_{i,j} ⊗ e_{j,k} ⊗ e_{i,k} can be decomposed as:
    # T = sum_{i,j} (e_{i,j} ⊗ sum_k e_{j,k} ⊗ sum_k e_{i,k})
    
    # But this gives a bilinear form, not the matrix multiplication map.
    
    # The matrix multiplication map is:
    # mu(A,B)[i,k] = sum_j A[i,j] * B[j,k]
    
    # The tensor for this map is in (R^{n*m})^* ⊗ (R^{m*p})^* ⊗ R^{n*p}:
    # T = sum_{i,j,k} e_{i,j}^* ⊗ e_{j,k}^* ⊗ e_{i,k}
    
    # A rank-R decomposition is T = sum_r f_r ⊗ g_r ⊗ h_r where:
    # f_r ∈ (R^{n*m})^*, g_r ∈ (R^{m*p})^*, h_r ∈ R^{n*p}
    
    # Then mu(A,B) = sum_r f_r(A) * g_r(B) * h_r
    
    # For the schoolbook: f_r(A) = A[i,j], g_r(B) = B[j,k], h_r = e_{i,k}
    
    # For a better decomposition, we need to find f_r, g_r, h_r such that:
    # sum_j A[i,j] * B[j,k] = sum_r f_r(A) * g_r(B) * h_r[i,k]
    
    # This is a hard problem. Let's just use the schoolbook for now.
    
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
# EVOLVE-BLOCK-END
