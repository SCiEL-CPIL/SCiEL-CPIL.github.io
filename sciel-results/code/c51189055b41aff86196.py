"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 5, 7]

# EVOLVE-BLOCK-START
import numpy as np
from scipy.linalg import dft

def construct():
    n, m, p = DIMENSIONS  # 3, 5, 7
    
    # Embed into larger dimensions for circulant structure
    # Use cyclic convolution size that accommodates the multiplication
    # For 3x5x7, embed into 4x8x8 and use circulant blocks
    
    # Strategy: Use the fact that matrix multiplication can be embedded
    # in a tensor product structure. We'll use a block-circulant approach.
    
    # The multiplication tensor T[i,j,k] = 1 if i in [0,n), j in [0,m), k in [0,p)
    # We want to decompose: T = sum_c U_c ⊗ V_c ⊗ W_c
    
    # Use FFT-based decomposition of the embedded circulant tensor
    # Embed size: use N = lcm or next power of 2 for clean DFT
    N = 8  # Size of cyclic convolution (>= max(m,p) = 7, and >= n=3)
    
    # The 3x5x7 multiplication can be seen as part of a larger structure.
    # We'll construct the decomposition using the DFT of size N.
    
    # For a circulant matrix of size N, the eigendecomposition gives N rank-1 terms.
    # We use this to decompose the tensor.
    
    # Construct the tensor for 3x5x7 multiplication
    # T[i, j, k] = 1 for valid indices
    # We'll use the standard approach: for each (i,j,k), we have a rank-1 term
    # but we want to group them using the circulant structure.
    
    # Alternative approach: Use the Strassen-like or Winograd-like idea
    # but with FFT structure.
    
    # Let's use a direct but optimized construction:
    # The minimal rank for 3x5x7 is unknown, but we can do better than 105.
    
    # Use the embedding: map i -> i, j -> j, k -> k in Z_8
    # The tensor is supported on {0,1,2} x {0,1,2,3,4} x {0,1,2,3,4,5,6}
    
    # We can decompose using the DFT basis of Z_8.
    # The DFT matrix F of size 8 has entries F[a,b] = omega^(a*b) where omega = exp(2*pi*i/8)
    # But we need integer/half-integer entries, so we work over reals/rationals.
    
    # Use the real DFT (cosine and sine components) to get rational approximations,
    # or better: use the number-theoretic transform over a suitable field.
    
    # For simplicity and correctness, let's use a known lower-rank construction.
    # The rank of 3x5x7 multiplication is at most 3*5*7 = 105 (schoolbook).
    # We can potentially do better with structured decompositions.
    
    # Let's try a block approach:
    # Split the 5x7 multiplication into blocks and use smaller known decompositions.
    
    # Actually, let's use a clean FFT-based construction over the reals.
    # The key: the tensor T can be written as a sum of rank-1 tensors using
    # the eigendecomposition of the associated circulant matrices.
    
    # Construct using DFT of size N=8
    # omega = exp(2*pi*i/N)
    # We'll use the fact that sum over b of omega^(a*b) * omega^(-c*b) = N * delta_{a,c}
    
    # For the 3x5x7 case, we embed each dimension into Z_8.
    # The multiplication tensor in the embedded space has a circulant structure
    # in the j and k dimensions when i is fixed.
    
    # Let's construct explicitly:
    rank = 0
    U_list = []
    V_list = []
    W_list = []
    
    # For each "frequency" in the cyclic convolution, we get rank-1 terms.
    # The j and k dimensions form a cyclic convolution of size N=8.
    # For each i in [0,3), and each frequency f in [0,8), we get a term.
    
    # But we need to be careful: the actual multiplication is not a full cyclic
    # convolution, it's a truncated one. We need to handle the boundary conditions.
    
    # Let's use a different approach: direct construction with grouping.
    # Group the (j,k) pairs by their sum mod N to exploit circulant structure.
    
    # Actually, the cleanest approach for this specific case:
    # Use the standard bilinear decomposition but with the FFT structure.
    
    # Let me construct it properly using the DFT:
    # T[i,j,k] = 1 for i<3, j<5, k<7
    # = sum_f (1/N) * omega^(-f*j) * omega^(f*k) * [sum over valid (j,k) with j<5,k<7 of omega^(f*(j-k))]
    
    # This is getting complex. Let me use a verified lower-rank construction.
    
    # For 3x5x7, a known upper bound on rank is less than 105.
    # Let's use the approach of embedding in a 4x8x8 and using the 8-point DFT.
    
    # The 8-point DFT gives 8 orthogonal rank-1 terms for the cyclic convolution.
    # For each of the 3 values of i, we get 8 terms, but some may be redundant.
    
    # Let's just construct the schoolbook but with a twist: use the DFT to 
    # combine terms and reduce rank.
    
    # Final approach: Use the fact that we can represent the multiplication
    # using the Kronecker product structure and FFT.
    
    # Construct U, V, W with rank = n * N where N is the DFT size, but optimized.
    
    # Actually, let me use a simple but effective construction:
    # rank = 3 * 8 = 24 terms from the DFT structure, but this might not be exact.
    
    # Let me verify with a direct construction and use the minimal known rank.
    
    # For now, use a construction that's guaranteed to work:
    # The rank is at most n*m*p = 105, but we can do better.
    
    # Use the following: decompose the 5x7 part using a 2x2 block structure
    # and apply Strassen-like ideas.
    
    # Given the complexity, let me use a clean FFT-based construction:
    
    N = 8  # DFT size
    omega = np.exp(2j * np.pi / N)
    
    # Precompute DFT matrix
    F = np.array([[omega**(a*b) for b in range(N)] for a in range(N)])
    
    # The multiplication tensor T[i,j,k] for i in [0,3), j in [0,5), k in [0,7)
    # We can write: T[i,j,k] = sum_f (1/N) * conj(F[f,j]) * F[f,k] * C[i,f]
    # where C[i,f] = sum_{j<5, k<7, j-k = something} ... 
    
    # This is not quite right. Let me use a different formulation.
    
    # The correct approach: The tensor T can be decomposed as:
    # T[i,j,k] = sum_c U[i,c] * V[j,c] * W[k,c]
    
    # Using the FFT structure, we set:
    # For each frequency f in [0,N):
    #   U[i, c(i,f)] = 1 for i < n
    #   V[j, c(i,f)] = conj(F[f,j]) / sqrt(N) for j < m  
    #   W[k, c(i,f)] = F[f,k] / sqrt(N) for k < p
    
    # But this doesn't give T[i,j,k] = 1; it gives a convolution.
    
    # Let me use the standard bilinear form with FFT acceleration concept:
    # The product A*B where A is 3x5 and B is 5x7 can be computed as:
    # (A*B)[i,k] = sum_j A[i,j] * B[j,k]
    
    # We can write this using the DFT of B's rows and A's columns.
    
    # Given time constraints, let me use a verified construction:
    # The rank of 3x5x7 matrix multiplication is at most 105, and we can
    # achieve a lower rank using structured decompositions.
    
    # Use the following construction based on the FFT:
    rank = n * N  # 3 * 8 = 24
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    for i in range(n):
        for f in range(N):
            # U part: U[i*m + j, col] for all j
            for j in range(m):
                U[i * m + j][col] = 1.0
            # V part: V[j*p + k, col] = conj(omega^(f*j)) / sqrt(N)
            for j in range(m):
                for k in range(p):
                    V[j * p + k][col] = np.conj(omega ** (f * j)).real / np.sqrt(N)
            # W part: W[k*n + i, col] = omega^(f*k) / sqrt(N)
            for k in range(p):
                W[k * n + i][col] = np.real(omega ** (f * k)) / np.sqrt(N)
            col += 1
    
    # This construction may not be exact. Let me verify and fix.
    # The issue: this gives a convolution, not exact multiplication.
    
    # Let me use the schoolbook but with a small optimization:
    # Actually, for correctness, let me return the schoolbook decomposition
    # but I'll try a lower-rank approach.
    
    # Use the following: the rank is at most (n+m-1)*(m+p-1) for some structures,
    # but that's not directly applicable.
    
    # Given the requirement for exact decomposition, let me use the schoolbook
    # but with half-integer entries where possible to show structure.
    
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
