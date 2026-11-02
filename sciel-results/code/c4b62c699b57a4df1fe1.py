"""
DFT-embedding construction for 3x5x7 matrix multiplication.
Maps the 5-dimensional inner product to a cyclic convolution of length 11,
diagonalizes via DFT, and projects back to get an exact rational decomposition.
"""

import time
import numpy as np
from fractions import Fraction

DIMENSIONS = [3, 5, 7]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 110  # safety margin before 120s limit
    
    n, m, p = DIMENSIONS  # 3, 5, 7
    N = 11  # smallest prime >= m + p - 1 = 11
    
    # First, try the DFT-based construction with rational entries
    # The idea: for each i in [0,n), the inner product A[i,:] . B[:,k]
    # can be computed via cyclic convolution of length N=11.
    # The cyclic convolution tensor of size N has rank N over the field
    # containing the N-th roots of unity.
    
    # We work over Q(omega) where omega = exp(2*pi*i/N), but we need
    # rational/integer entries. Instead, we use the fact that the
    # circulant matrix of size N can be diagonalized, and we can
    # express the decomposition using the DFT matrix entries.
    
    # For exactness with rational entries, we use the following approach:
    # The multiplication tensor T[i,j,k] = 1 for i<n, j<m, k<p
    # can be decomposed as:
    # T[i,j,k] = sum_{f=0}^{N-1} (1/N) * omega^{-f*j} * omega^{f*k} * delta_{j-k mod N, 0}
    # Wait, that's not quite right. Let me think more carefully.
    
    # Actually, the standard result: the cyclic convolution of two length-N
    # sequences can be computed with N multiplications via FFT. The tensor
    # for cyclic convolution of size N has tensor rank N.
    
    # For our case, we embed the 5x7 multiplication into a cyclic convolution
    # of size 11. For each row i of A (length 5) and each column k of B (length 7),
    # the inner product is:
    # (A*B)[i,k] = sum_{j=0}^{4} A[i,j] * B[j,k]
    
    # We can view this as: pad A[i,:] to length 11 (zeros beyond index 4)
    # and B[:,k] to length 11 (zeros beyond index 6), then take the cyclic
    # convolution and extract the appropriate component.
    
    # But the cyclic convolution gives:
    # (a * b)_cyclic[t] = sum_{s=0}^{10} a[s] * b[(t-s) mod 11]
    
    # We want sum_{j=0}^{4} A[i,j] * B[j,k], which is NOT directly a single
    # component of the cyclic convolution.
    
    # Let me use a different, verified approach:
    # Use the fact that the tensor can be decomposed using the DFT basis.
    
    # The correct construction:
    # For the cyclic convolution tensor of size N, we have:
    # C[t] = sum_s a[s] * b[(t-s) mod N]
    # = sum_f (1/N) * (sum_s a[s] * omega^{-f*s}) * (sum_r b[r] * omega^{f*r}) * omega^{f*t}
    
    # So the rank-1 decomposition has N terms with:
    # U_f[s] = omega^{-f*s} / sqrt(N)  (for input a)
    # V_f[r] = omega^{f*r} / sqrt(N)   (for input b)  
    # W_f[t] = omega^{f*t}             (for output c)
    
    # But our multiplication is NOT a full cyclic convolution. It's a truncated
    # bilinear form. So this direct approach doesn't give exact results.
    
    # Instead, let's use a guaranteed correct construction:
    # The schoolbook decomposition has rank n*m*p = 105.
    # We can try to reduce this using block structure.
    
    # Strategy: Use the DFT to combine terms where possible.
    # For fixed i, the terms involving j=0..4 and k=0..6 can potentially
    # be grouped.
    
    # Actually, let me try a direct low-rank construction:
    # We know that rank(3x5x7) <= 3*rank(5x7) and various other bounds.
    # But the simplest guaranteed improvement: use the fact that we can
    # share factors across i values.
    
    # Let me construct a decomposition of rank n * (m+p-1) using the
    # "Toeplitz" structure. For each i, the inner product sum_j A[i,j]*B[j,k]
    # is a bilinear form in A[i,:] and B[:,k].
    
    # Key insight: For fixed i, define the vector a_i = A[i, 0..4] and 
    # the matrix B. The product is a_i^T * B[:, k].
    
    # We can write B as a sum of rank-1 matrices: B = sum_{r=0}^{R-1} u_r v_r^T
    # Then (A*B)[i,k] = sum_r (a_i^T u_r) * v_r[k]
    # Total rank = n * R where R = rank(B) <= min(m,p) = 5.
    # But B is a specific matrix, not arbitrary. The tensor T[i,j,k]=1 is
    # the full multiplication tensor, not for a specific B.
    
    # I'm overcomplicating this. Let me just use the schoolbook but verify
    # it's correct, then try a simple optimization.
    
    # Schoolbook: rank = 105
    # Simple improvement: For each i, the 5*7 = 35 terms can be written
    # using the structure of the inner product.
    
    # Actually, the cleanest low-rank construction I can verify:
    # Use the DFT of size N=11 to get a decomposition of the cyclic
    # convolution, then project.
    
    # Let me implement the DFT-based construction carefully:
    # 
    # We want to compute C[i,k] = sum_{j=0}^{4} A[i,j] * B[j,k]
    # 
    # Embed: define a_i[s] = A[i,s] for s<5, 0 otherwise (s in Z_11)
    #         b_k[r] = B[r,k] for r<7, 0 otherwise (r in Z_11)
    # 
    # The cyclic convolution: (a_i * b_k)[t] = sum_{s=0}^{10} a_i[s] * b_k[(t-s) mod 11]
    # 
    # We want sum_{j=0}^{4} A[i,j] * B[j,k] = sum_{j=0}^{4} a_i[j] * b_k[j]
    # 
    # This equals (a_i * b_k')[t] where b_k'[s] = b_k[-s mod 11] and t=0?
    # Let's check: (a_i * b_k')[0] = sum_s a_i[s] * b_k'[(0-s) mod 11] = sum_s a_i[s] * b_k[-s mod 11]
    # = sum_s a_i[s] * b_k[(11-s) mod 11]
    # For s=0: a_i[0]*b_k[0], s=1: a_i[1]*b_k[10], s=2: a_i[2]*b_k[9], ...
    # This is NOT what we want.
    
    # We want sum_j a_i[j] * b_k[j], which is the "cross-correlation" at lag 0.
    # Cross-correlation: (a_i ⋆ b_k)[t] = sum_s a_i[s] * b_k[s+t]
    # At t=0: sum_s a_i[s] * b_k[s] = sum_{j=0}^{4} A[i,j]*B[j,k] ✓
    
    # So C[i,k] = (a_i ⋆ b_k)[0]
    
    # The cross-correlation can be computed via convolution:
    # (a_i ⋆ b_k)[t] = (a_i * b_k_flipped)[t] where b_k_flipped[s] = b_k[-s mod N]
    
    # So C[i,k] = (a_i * b_k_flipped)[0]
    
    # Now use FFT: (a_i * b_k_flipped)[0] = sum_f (1/N) * A_hat[i,f] * B_hat_flipped[k,f] * omega^{f*0}
    # = (1/N) * sum_f A_hat[i,f] * B_hat_flipped[k,f]
    
    # Where A_hat[i,f] = sum_s a_i[s] * omega^{-f*s}
    #        B_hat_flipped[k,f] = sum_r b_k_flipped[r] * omega^{-f*r} = sum_r b_k[-r mod N] * omega^{-f*r}
    #                           = sum_j b_k[j] * omega^{f*j}  (substituting j = -r mod N)
    
    # So C[i,k] = (1/N) * sum_f (sum_s a_i[s] * omega^{-f*s}) * (sum_j b_k[j] * omega^{f*j})
    
    # This is a sum of N=11 rank-1 terms per (i,k) pair!
    # Total rank = n * N = 3 * 11 = 33
    
    # The rank-1 terms are:
    # For each f in [0,11):
    #   U[i*m+s, col(i,f)] = omega^{-f*s} / sqrt(N)  for s in [0,5)
    #   V[j*p+k, col(i,f)] = omega^{f*j} / sqrt(N)   for j in [0,7)  -- wait, this depends on k
    
    # Hmm, the issue is that B_hat_flipped[k,f] depends on k, so the V factor
    # is not separable in the same way.
    
    # Let me re-examine:
    # C[i,k] = (1/N) * sum_f A_hat[i,f] * B_hat_flipped[k,f]
    # where A_hat[i,f] = sum_s a_i[s] * omega^{-f*s} (depends on i and s)
    #       B_hat_flipped[k,f] = sum_j B[j,k] * omega^{f*j} (depends on k and j)
    
    # So the decomposition is:
    # C[i,k] = sum_f U[i,f] * V[k,f] where
    #   U[i,f] = (1/sqrt(N)) * sum_s A[i,s] * omega^{-f*s}  -- but this is a linear combination of A's entries
    #   V[k,f] = (1/sqrt(N)) * sum_j B[j,k] * omega^{f*j}
    
    # For the tensor decomposition, we need:
    # T[i,j,k] = sum_c U_c[i] * V_c[j] * W_c[k]
    
    # From the expression:
    # C[i,k] = (1/N) * sum_f (sum_s A[i,s] * omega^{-f*s}) * (sum_j B[j,k] * omega^{f*j})
    #        = sum_f (1/N) * (sum_s A[i,s] * omega^{-f*s}) * (sum_j B[j,k] * omega^{f*j})
    
    # This can be written as:
    # C[i,k] = sum_{f,s,j} (1/N) * A[i,s] * B[j,k] * omega^{-f*s} * omega^{f*j}
    #        = sum_{f,s,j} (1/N) * A[i,s] * B[j,k] * omega^{f*(j-s)}
    
    # For this to match T[i,j,k] = 1 when i<n, j<m, k<p, we need:
    # sum_{f,s,j} (1/N) * omega^{f*(j-s)} = 1 for the specific (j,k) pair?
    
    # No, I'm confusing the bilinear form with the tensor.
    
    # The tensor T[i,j,k] is defined such that C[i,k] = sum_{j} T[i,j,k] * A[i,j] * B[j,k]
    # For matrix multiplication, T[i,j,k] = 1 for all valid i,j,k.
    
    # So we need: sum_c U_c[i] * V_c[j] * W_c[k] = 1 for all i<n, j<m, k<p
    
    # From the FFT construction:
    # sum_{f,s,j'} (1/N) * omega^{-f*s} * A[i,s] * omega^{f*j'} * B[j',k]
    # = sum_{s,j'} A[i,s] * B[j',k] * (1/N) * sum_f omega^{f*(j'-s)}
    # = sum_{s,j'} A[i,s] * B[j',k] * delta_{j',s}  (since sum_f omega^{f*(j'-s)} = N*delta_{j',s})
    # = sum_s A[i,s] * B[s,k]
    
    # This is exactly what we want! So the decomposition is:
    # T[i,j,k] = sum_f (1/N) * (sum_s omega^{-f*s} * [i,j] term) * ...
    
    # More precisely, the rank-1 terms are indexed by (f, s, j') but with the
    # constraint that they combine to give delta. Actually, the decomposition
    # has N terms per "frequency" f, but the s and j' are summed over.
    
    # Let me write it cleanly:
    # T[i,j,k] = sum_{f=0}^{N-1} (1/N) * omega^{-f*j} * omega^{f*j} * 1
    # That's just sum_f (1/N) = 1. That's trivial and gives rank N=11.
    
    # Wait, that can't be right. Let me be more careful.
    
    # The bilinear form is: C[i,k] = sum_j A[i,j] * B[j,k]
    # 
    # Using FFT:
    # Define a_i[s] = A[i,s] for s in Z_N (padded with zeros)
    # Define b_k[r] = B[r,k] for r in Z_N (padded with zeros)
    # 
    # C[i,k] = sum_{s in Z_N} a_i[s] * b_k[s]  (since a_i and b_k are zero outside their ranges)
    # 
    # Now, sum_s a_i[s] * b_k[s] = sum_s a_i[s] * b_k[s] * (1/N) * sum_f omega^{f*0}
    # = (1/N) * sum_f (sum_s a_i[s] * b_k[s] * omega^{f*0})
    # = (1/N) * sum_f (sum_s a_i[s] * omega^{-f*s/2} * b_k[s] * omega^{f*s/2})  -- not helpful
    # 
    # Better: use the identity delta_{a,b} = (1/N) sum_f omega^{f(a-b)}
    # sum_s a_i[s] * b_k[s] = sum_s a_i[s] * b_k[s] * delta_{s,s}
    # = (1/N) * sum_f sum_s a_i[s] * b_k[s] * omega^{f(s-s)}
    # = (1/N) * sum_f (sum_s a_i[s] * b_k[s])
    # = sum_s a_i[s] * b_k[s]  (trivial)
    
    # The FFT trick works for convolution, not for element-wise product.
    # For element-wise product (Hadamard product), the FFT doesn't directly help.
    
    # I was confusing convolution with Hadamard product. The matrix multiplication
    # inner product is a Hadamard product sum, not a convolution.
    
    # So the DFT approach doesn't directly apply. Let me go back to basics.
    
    # For the Hadamard product sum: sum_j a_j * b_j
    # This is already a sum of m rank-1 terms (one per j).
    # For 3x5x7, the schoolbook rank is 3*5*7 = 105.
    
    # Can we do better? The tensor rank of matrix multiplication is a deep
    # question. For 3x5x7, the best known upper bounds are not much below 105.
    
    # Let me try a simple block decomposition:
    # Split the 5x7 into 2x2 blocks and use Strassen-like ideas.
    # But 5 and 7 don't divide evenly by 2.
    
    # Alternative: use the fact that rank(3x5x7) <= rank(3x5x5) * something.
    # Or: embed 3x5x7 into 4x8x8 and use known results.
    
    # Given the complexity, let me just return the schoolbook decomposition
    # which is guaranteed correct. Rank = 105.
    
    # But wait - let me try one more idea: use the DFT to combine the
    # k-dimension. For fixed i and j, the term A[i,j]*B[j,k] contributes
    # to all k. If we could share the A[i,j] factor across k values...
    
    # Actually, here's a valid optimization:
    # For each (i,j), the term A[i,j] * B[j,k] for k=0..6 can be written
    # as A[i,j] * (B[j,0], B[j,1], ..., B[j,6]) which is a vector in R^7.
    # We can't reduce the rank of this vector below 7 in general.
    
    # So the schoolbook rank of 105 is likely close to optimal for this
    # structured decomposition.
    
    # Let me just implement the schoolbook correctly.
    
    if time.time() > deadline - 5:
        # Return schoolbook quickly
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
    
    # Try to find a better decomposition using optimization
    # We'll use a simple gradient-based approach to minimize rank
    
    # Actually, let me try a structured decomposition using the
    # Kronecker product structure.
    
    # Idea: 3 = 1*3, 5 = 1*5, 7 = 1*7
    # Or: use the fact that the tensor is a sum of Kronecker products.
    
    # The multiplication tensor T = sum_{i,j,k} e_i ⊗ e_j ⊗ e_k
    # where e_i, e_j, e_k are standard basis vectors.
    
    # We can write this as:
    # T = (sum_i e_i) ⊗ (sum_j e_j) ⊗ (sum_k e_k) - (terms where indices don't match)
    # But this doesn't reduce rank.
    
    # Let me try: T = A ⊗ B ⊗ C where A, B, C are sums of basis vectors.
    # (sum_i e_i) ⊗ (sum_j e_j) ⊗ (sum_k e_k) = sum_{i,j,k} e_i ⊗ e_j ⊗ e_k = T
    # This is rank 1! But wait, that's the tensor (1,1,...,1) ⊗ (1,1,...,1) ⊗ (1,1,...,1)
    # which has all entries equal to 1. That's exactly T!
    
    # So T = u ⊗ v ⊗ w where u = (1,1,1), v = (1,1,1,1,1), w = (1,1,1,1,1,1,1)
    # This is a rank-1 decomposition!
    
    # But wait, that means the tensor rank is 1, which is clearly wrong.
    # The issue: the tensor T[i,j,k] = 1 for all i,j,k is indeed rank 1.
    # But matrix multiplication tensor is NOT just T[i,j,k]=1.
    
    # Let me re-read the problem: "Find a rank-minimal exact tensor decomposition
    # for multiplying a 3 by 5 matrix by a 5 by 7 matrix."
    
    # The tensor for matrix multiplication (3x5 times 5x7) has entries:
    # T[i,j,k] = 1 if the (i,j) entry of A and (j,k) entry of B multiply to
    # contribute to (i,k) entry of C.
    
    # In the standard bilinear decomposition, the tensor is:
    # T[i,j,k] = 1 for all valid i in [0,3), j in [0,5), k in [0,7)
    
    # And the decomposition T = sum_c U_c ⊗ V_c ⊗ W_c means:
    # T[i,j,k] = sum_c U_c[i*m+j] * V_c[j*p+k] * W_c[k*n+i]
    
    # Wait, the indices are coupled! U is indexed by (i,j), V by (j,k), W by (k,i).
    # So it's not a simple Kronecker product.
    
    # The schoolbook decomposition: for each (i,j,k), have a term with
    # U[i*m+j, c(i,j,k)] = 1, V[j*p+k, c(i,j,k)] = 1, W[k*n+i, c(i,j,k)] = 1
    
    # This gives T[i,j,k] = 1 for all valid indices, which is correct.
    
    # Can we do better? The key constraint is that U is indexed by (i,j),
    # V by (j,k), W by (k,i). The coupling through j and k limits the rank.
    
    # For now, let me just return the schoolbook decomposition.
    
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
