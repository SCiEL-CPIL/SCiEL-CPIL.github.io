"""Winograd-style decomposition for 3x4 * 4x4 matrix multiplication."""
DIMENSIONS = [3, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 3, 4, 4
    
    # Use a structured approach:
    # Matrix A is 3x4, Matrix B is 4x4, Result C is 3x4
    # C[i,j] = sum_k A[i,k] * B[k,j]
    
    # Strategy: Use rank-1 updates with shared structure
    # For 3x4*4x4, schoolbook is 3*4*4 = 48
    # Try to find a decomposition with rank < 48
    
    # Key insight: We can factor through intermediate transformations
    # C = A @ B where A is 3x4, B is 4x4
    
    # Let's build explicit factors for a decomposition
    # The tensor T[i,j,k] = delta such that C[i,j] = sum_k A[i,k]*B[k,j]
    
    # U has shape (n*m, rank) = (12, rank)
    # V has shape (m*p, rank) = (16, rank)  
    # W has shape (p*n, rank) = (12, rank)
    
    # W convention: W[k*n + i, c] - so row index is k*n + i where k is from B's row, i is from C's row
    
    # Let's try a direct construction using the algebraic structure
    # We need: C[i,j] = sum_c U[i*m+j, c] * V[j*p+k... 
    
    # Actually, let me re-read the convention:
    # U[i*m + j, c] - i from A row, j from A col (=B row)
    # V[j*p + k, c] - j from B row, k from B col
    # W[k*n + i, c] - k from B row... wait
    
    # The tensor decomposition: T_{a,b,c} where a indexes A (n*m), b indexes B (m*p), c indexes output (p*n)
    # Standard: C[i,j] = sum_k A[i,k] B[k,j]
    # So T[i*m+k, k*p+j, j*n+i] should be 1 for each (i,k,j)
    
    # Wait, let me reconsider. The output index is k*n+i per the convention W[k*n+i, c]
    # This means the output tensor has indices (k, i) where k ranges over p and i ranges over n
    # So C has shape p x n? But we want C to be n x p = 3 x 4
    
    # Re-reading: "The output factor follows the official trace convention W[k*n+i,r]"
    # So W has p*n rows, indexed as k*n + i, where k is 0..p-1 and i is 0..n-1
    # This suggests the output C is arranged as C[k, i] = sum_j A[i, j] * B[j, k]
    # i.e., C is p x n, not n x p!
    
    # So actually: C[k, i] = sum_j A[i, j] * B[j, k]
    # A is n x m = 3 x 4
    # B is m x p = 4 x 4
    # C is p x n = 4 x 3
    
    # U[i*m + j, c]: i in 0..n-1, j in 0..m-1 -> indexes A[i,j]
    # V[j*p + k, c]: j in 0..m-1, k in 0..p-1 -> indexes B[j,k]
    # W[k*n + i, c]: k in 0..p-1, i in 0..n-1 -> indexes C[k,i]
    
    # So the decomposition is: C[k,i] = sum_c U[i*m+j, c] * V[j*p+k, c] for some j... 
    # No, it's: sum over c of U[i*m+j, c] * V[j*p+k, c] summed over j too?
    
    # The tensor is: T_{(i,j), (j',k), (k',i')} = delta_{j,j'} * delta_{k,k'} * delta_{i,i'}
    # Wait, that's just the schoolbook.
    
    # For the decomposition: C[k,i] = sum_j A[i,j] B[j,k]
    # = sum_j sum_c U[i*m+j, c] V[j*p+k, c]  (if we can decompose the product)
    
    # Actually the bilinear decomposition is:
    # C[k,i] = sum_c (sum_j U[i*m+j, c] * V[j*p+k, c])
    # where the inner sum is over j (the shared index between A and B)
    
    # So we need: for each c, define alpha_c(i,j) = U[i*m+j, c], beta_c(j,k) = V[j*p+k, c], gamma_c(k,i) = W[k*n+i, c]
    # And: sum_c alpha_c(i,j) * beta_c(j,k) * gamma_c(k,i) ... no that's not right
    
    # Let me think again. The matrix multiplication tensor T has T_{a,b,c} where:
    # a = i*m + j (A index), b = j'*p + k (B index), c = k'*n + i' (C index)
    # T_{a,b,c} = 1 if j=j' and k=k' and i=i', else 0
    # Because C[i',k'] = sum_j A[i',j] * B[j,k']
    # So T_{i*m+j, j*p+k, k*n+i} = 1
    
    # Rank decomposition: T_{a,b,c} = sum_r U[a,r] * V[b,r] * W[c,r]
    
    # Schoolbook: one term per (i,j,k) triple
    # We want fewer terms using algebraic tricks
    
    # For 3x4 * 4x4 -> 4x3 output:
    # Let's try to use the structure. n=3, m=4, p=4
    
    # One approach: group columns of A and rows of B
    # Since m=p=4, we can use 2x2 block structure on the shared dimension
    
    # Let's try a concrete construction with rank 36 (savings from 48)
    # Using 2x2 Winograd on pairs of the 4 shared indices
    
    # Split j into j = 2*s + t where s in {0,1}, t in {0,1}
    # Apply Winograd to each (i, s) pair over t and corresponding k
    
    # This is getting complex. Let me just implement a verified construction.
    
    # Simple approach: Use the fact that we can share computations
    # For now, let's return the schoolbook to ensure correctness,
    # then we can optimize.
    
    rank = n * m * p  # 48
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
