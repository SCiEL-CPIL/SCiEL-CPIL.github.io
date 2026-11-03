"""Matrix multiplication 4,4,7: exact tensor decomposition."""
import numpy as np

DIMENSIONS = [4, 4, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # n=4 rows of A, m=4 cols of A / rows of B, p=7 cols of B
    # Output C is n x p = 4 x 7
    # Tensor: T[i, j, k] = delta (matrix mult)
    # C[i,k] = sum_j A[i,j] B[j,k]
    # U shape: (n*m, rank) = (16, rank)
    # V shape: (m*p, rank) = (28, rank)  
    # W shape: (p*n, rank) = (28, rank)
    # Convention: W[k*n+i, r] (k is output col, i is output row)
    # C[i,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    #         = sum_j,k,r ... wait, let me re-read.
    # Actually: C[i,k] = sum_j A[i,j] B[j,k]
    # In tensor decomposition: C[i,k] = sum_r U[i,j,r] V[j,k,r] W[i,k,r]
    # But the interface says U has shape (n*m, rank), V has shape (m*p, rank), W has shape (p*n, rank)
    # So U is indexed by (i,j), V by (j,k), W by (k,i) with W[k*n+i, r]
    # C[i,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] summed over j? 
    # No: C[i,k] = sum_j sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # Wait, that doesn't work dimensionally. Let me think again.
    # Standard tensor decomposition for matmul:
    # C[i,k] = sum_j A[i,j] B[j,k] = sum_r (sum_j U[i*m+j, r] A[i,j]) (sum_k V[j*p+k, r] B[j,k]) W[k*n+i, r]
    # Hmm, that's not right either.
    # Correct form: C[i,k] = sum_r U_{i,j,r} A[i,j] V_{j,k,r} B[j,k] W_{k,i,r}
    # where sums over j and r.
    # So: C[i,k] = sum_r W[k*n+i, r] * (sum_j U[i*m+j, r] * A[i,j]) * (sum_j V[j*p+k, r] * B[j,k])
    # Wait, the j index appears in both U and V. So:
    # C[i,k] = sum_r W[k*n+i, r] * (sum_j U[i*m+j, r] A[i,j]) * (sum_j V[j*p+k, r] B[j,k])
    # But this has two separate sums over j, which is a product of two sums = sum over j1 and j2.
    # That's not right. The correct tensor decomposition is:
    # C[i,k] = sum_{r} U[i,j,r] A[i,j] V[j,k,r] B[j,k] W[i,k,r]  (Einstein summation over j and r)
    # So U has indices (i,j,r), V has (j,k,r), W has (i,k,r) or (k,i,r).
    # Flattened: U[i*m+j, r], V[j*p+k, r], W[k*n+i, r]
    # C[i,k] = sum_{j, r} U[i*m+j, r] * A[i,j] * V[j*p+k, r] * B[j,k] * W[k*n+i, r]
    # = sum_r W[k*n+i, r] * (sum_j U[i*m+j, r] A[i,j] V[j*p+k, r] B[j,k])
    # This is the correct form.
    
    # Schoolbook: rank = n*m*p = 4*4*7 = 112
    # For each (i,j,k): U[i*m+j, r] = 1, V[j*p+k, r] = 1, W[k*n+i, r] = 1
    
    # Try to do better. Let me think about block structure.
    # Split B into B1 (4x4) and B2 (4x3).
    # C = A*B1 + A*B2
    # A*B1: 4x4x4 matmul, rank 64 schoolbook
    # A*B2: 4x4x3 matmul, rank 48 schoolbook
    # Total: 112. Same as before.
    
    # Can we use Winograd for the 4x4x4 part?
    # Winograd F(4,4) has rank 24.
    # But I need to construct the factors.
    
    # Let me try a different approach: use the fact that we can share computations.
    # Actually, let me just implement schoolbook correctly and return rank 112.
    # The seed already does this.
    
    # Let me try to get a better rank. 
    # Known: 4x4x4 matmul has rank 24 (Winograd).
    # 4x4x3 matmul: not sure of minimal rank.
    
    # Let me implement Winograd 4x4x4.
    # The Winograd algorithm for 4x4 matrix multiplication:
    # A (4x4) * B (4x4) = C (4x4)
    # Transform A: A' = U_A * A * V_A (element-wise products)
    # Transform B: B' = U_B * B * V_B
    # C' = A' * B' (element-wise, 4x4)
    # C = U_C * C' * V_C
    # The number of multiplications is 16 (element-wise) + overhead.
    # Actually Winograd F(4,4) uses 24 multiplications total.
    
    # Let me look up the Winograd 4x4x4 factors.
    # The factors are based on the 8x8 DFT or similar.
    
    # Alternative: use the rank-24 decomposition from literature.
    # I'll construct it using the FFT approach for 4x4 convolution, 
    # but adapted for matrix multiplication.
    
    # Actually, matrix multiplication is NOT convolution. 
    # But there's a trick: A*B where A is 4x4 and B is 4x4 can be 
    # related to convolution via appropriate padding and rearrangement.
    
    # Let me just use schoolbook for now and return rank 112.
    # The seed does this, so I need to do at least as well.
    
    # Actually, let me try a simple improvement: 
    # For the 4x4x3 part, can I reduce the rank?
    # 4x4x3: C[i,k] = sum_j A[i,j] B[j,k] for k=0,1,2
    # Schoolbook: 4*4*3 = 48
    # Can I do better? Maybe using Winograd-like transforms on A.
    
    # Let me just return schoolbook rank 112.
    
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
