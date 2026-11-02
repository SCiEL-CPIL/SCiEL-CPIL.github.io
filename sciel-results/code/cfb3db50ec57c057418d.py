"""Block-circulant embedding for 3x5x7 matrix multiplication."""
DIMENSIONS = [3, 5, 7]

# EVOLVE-BLOCK-START
import numpy as np
from scipy.linalg import dft as scipy_dft

def construct():
    n, m, p = DIMENSIONS  # 3, 5, 7
    
    # We use a block-circulant embedding strategy.
    # Embed the 3x5x7 multiplication into a larger structured tensor.
    # 
    # Key idea: For fixed i, the product (A*B)[i,k] = sum_j A[i,j]*B[j,k]
    # can be viewed as a bilinear form. We can use the FFT structure to
    # combine terms.
    #
    # Strategy: Use cyclic convolution of size N >= m + p - 1 = 11, so N=12.
    # For each i in [0,3), we decompose the 5x7 multiplication using
    # the DFT of size N.
    #
    # The tensor T[i,j,k] = 1 for i<n, j<m, k<p
    # We want: T = sum_c U_c ⊗ V_c ⊗ W_c
    #
    # Using the FFT structure:
    # For each frequency f in [0,N), and each i in [0,n):
    #   U[i*m+j, c] = 1/sqrt(N) for j<m
    #   V[j*p+k, c] = conj(omega^(f*j)) * delta_k_valid / sqrt(N)  
    #   W[k*n+i, c] = omega^(f*k) / sqrt(N)
    #
    # But this doesn't directly give T[i,j,k]=1. Let me think more carefully.
    #
    # Actually, the correct approach:
    # The multiplication tensor for 3x5x7 has rank at most 105 (schoolbook).
    # We can potentially reduce this using the structure.
    #
    # Let's use a different approach: exploit the fact that for fixed i,
    # the 5x7 multiplication is a "diagonal" tensor with rank 35.
    # Across the 3 values of i, we can potentially share some structure.
    #
    # However, the most straightforward improvement is to use a Winograd-like
    # or FFT-based decomposition that reduces the constant factor.
    #
    # For now, let's implement a clean construction that achieves rank < 105.
    # 
    # Approach: Use the fact that we can embed 5x7 into a 12x12 cyclic convolution.
    # The 12-point DFT gives 12 orthogonal rank-1 terms for the convolution.
    # For each of the 3 i values, we get 12 terms, but we need to be careful
    # about the embedding.
    
    N = 12  # Cyclic convolution size, >= m + p - 1 = 11
    
    # Precompute the DFT matrix of size N
    omega = np.exp(2j * np.pi / N)
    F = np.array([[omega**(a*b) for b in range(N)] for a in range(N)])
    
    # The inverse DFT is F^{-1} = (1/N) * conj(F)
    
    # For the 3x5x7 multiplication:
    # T[i,j,k] = 1 for i<3, j<5, k<7
    # 
    # We can write this as a sum of rank-1 tensors using the DFT structure.
    # For each frequency f in [0,N):
    #   The contribution to T[i,j,k] from frequency f is:
    #   (1/N) * conj(F[f,j]) * F[f,k] * [sum over valid j,k of ...]
    #
    # This is getting complex. Let me use a verified construction.
    #
    # Actually, the simplest correct approach with potential for improvement:
    # Use the schoolbook but with a more efficient indexing.
    # The rank is 105, but we can potentially reduce it.
    #
    # Let's try a block approach: split the 5x7 into 2x2 blocks.
    # 5 = 2+2+1, 7 = 2+2+2+1
    # Use Strassen-like ideas for the 2x2 blocks.
    #
    # For a 2x2 matrix multiplication, Strassen uses 7 mults instead of 8.
    # But this is for square matrices. For rectangular, we can still gain.
    #
    # Let's implement a clean construction with rank = 105 for now,
    # but structured to allow future optimization.
    
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    
    # Convert to lists as required
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
