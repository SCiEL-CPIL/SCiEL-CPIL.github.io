"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 5, 7]

# EVOLVE-BLOCK-START
import numpy as np
from fractions import Fraction

def construct():
    n, m, p = DIMENSIONS  # 3, 5, 7
    
    # Block-circulant embedding strategy:
    # Embed the 3x5x7 multiplication into a cyclic convolution structure.
    # We use the fact that matrix multiplication can be decomposed using
    # circulant matrices and the DFT (Discrete Fourier Transform).
    
    # The key insight: if we can embed the multiplication into a larger
    # circulant structure, we can use the DFT to get a rank decomposition.
    
    # For the 3x5x7 case, we'll use the following approach:
    # 1. Embed the j and k dimensions into a cyclic group of size N
    # 2. Use the DFT of size N to diagonalize the circulant structure
    # 3. This gives us N rank-1 terms for each i in [0,3)
    
    # However, we need to be careful: the actual multiplication is not a
    # full cyclic convolution, but a truncated one. We need to handle the
    # boundary conditions properly.
    
    # Let's use N = 8 (next power of 2 >= max(m, p) = 7)
    N = 8
    
    # The DFT matrix F of size N has entries F[a,b] = omega^(a*b)
    # where omega = exp(2*pi*i/N)
    # But we need integer/half-integer entries, so we'll work with the
    # real and imaginary parts separately.
    
    # Actually, let's use a different approach: we'll construct the
    # decomposition explicitly using the structure of the multiplication tensor.
    
    # The multiplication tensor T[i,j,k] = 1 for i < n, j < m, k < p
    # We want to write T = sum_c U_c ⊗ V_c ⊗ W_c
    
    # Using the FFT structure, we can write:
    # T[i,j,k] = sum_f (1/N) * conj(omega^(f*j)) * omega^(f*k) * C[i,f]
    # where C[i,f] is a correction term that accounts for the truncation.
    
    # But this is getting complex. Let's use a simpler but effective approach:
    # We'll use the fact that the multiplication can be decomposed using
    # the Kronecker product structure and exploit the small dimensions.
    
    # For 3x5x7, we can use the following construction:
    # Split the 5x7 matrix into blocks and use smaller known decompositions.
    
    # Actually, let's use a direct construction based on the following insight:
    # The rank of the multiplication tensor is at most n*m*p = 105, but we can
    # do better by exploiting the structure.
    
    # Let's use the following approach: we'll construct a decomposition with
    # rank = n * (m + p - 1) = 3 * 11 = 33, using the structure of the
    # multiplication.
    
    # This is based on the fact that the multiplication can be viewed as a
    # convolution, and the rank of a convolution of length m and p is at
    # most m + p - 1.
    
    # However, this is not quite right. Let's use a different approach.
    
    # We'll use the following construction:
    # rank = n * m + m * p - m = 15 + 35 - 5 = 45
    # This is based on the fact that we can share some of the rank-1 terms.
    
    # Actually, let's use the simplest correct construction: the schoolbook
    # decomposition with rank = n * m * p = 105.
    
    # But wait, we can do better. Let's use the following insight:
    # The multiplication tensor has a certain structure that allows us to
    # reduce the rank. Specifically, we can use the fact that the tensor is
    # supported on a subset of the full space.
    
    # Let's use the following construction:
    # We'll group the (j,k) pairs by their sum j+k and use the structure
    # to reduce the rank.
    
    # For each i in [0,3), and each possible value of s = j+k in [0, m+p-2],
    # we can have a rank-1 term. But this doesn't directly give a decomposition.
    
    # Let's use a different approach: we'll use the fact that the multiplication
    # can be decomposed using the DFT of the embedded circulant structure.
    
    # Here's the key: we'll use N = 8 and construct the decomposition as follows:
    # For each frequency f in [0,8), we get a rank-1 term for each i in [0,3).
    # This gives us 3 * 8 = 24 rank-1 terms.
    
    # But we need to be careful: the DFT gives complex entries, and we need
    # integer/half-integer entries. We'll use the real and imaginary parts
    # separately, which doubles the rank to 3 * 8 * 2 = 48.
    
    # Actually, we can do better: we'll use the fact that the DFT matrix is
    # unitary, and we can normalize the entries to be half-integers.
    
    # Let's construct the decomposition explicitly:
    
    # omega = exp(2*pi*i/8) = cos(pi/4) + i*sin(pi/4) = sqrt(2)/2 + i*sqrt(2)/2
    # So omega^k for k = 0,1,...,7 are:
    # k=0: 1
    # k=1: sqrt(2)/2 + i*sqrt(2)/2
    # k=2: i
    # k=3: -sqrt(2)/2 + i*sqrt(2)/2
    # k=4: -1
    # k=5: -sqrt(2)/2 - i*sqrt(2)/2
    # k=6: -i
    # k=7: sqrt(2)/2 - i*sqrt(2)/2
    
    # These involve sqrt(2), which is not a half-integer. So we can't use
    # the complex DFT directly.
    
    # Let's use a different approach: we'll use the number-theoretic transform
    # over a suitable finite field, or we'll use a real DFT with rational entries.
    
    # Actually, let's use the following construction: we'll use the fact that
    # the multiplication can be decomposed using the structure of the indices.
    
    # Here's a simple but effective construction:
    # We'll use rank = n * m + m * p - m = 45
    # This is based on the following insight:
    # (A*B)[i,k] = sum_j A[i,j] * B[j,k]
    # We can write this as:
    # (A*B)[i,k] = sum_j A[i,j] * B[j,k]
    # = sum_j (A[i,j] - A[i,0]) * B[j,k] + A[i,0] * sum_j B[j,k]
    # This doesn't directly help.
    
    # Let's use a different approach: we'll use the following construction:
    # rank = n * (m + p) - n = 3 * 12 - 3 = 33
    # This is based on the fact that we can share the computation of certain terms.
    
    # Actually, I realize I'm overcomplicating this. Let's use the simplest
    # correct construction: the schoolbook decomposition with rank = 105.
    
    # But we can do a little better: we can use the fact that the multiplication
    # tensor has a certain symmetry to reduce the rank slightly.
    
    # For now, let's use the schoolbook decomposition, which is guaranteed to
    # be correct, and see if we can improve it in future iterations.
    
    rank = n * m * p  # 105
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
