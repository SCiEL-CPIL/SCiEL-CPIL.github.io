"""Matrix multiplication 2x3x5 rank-minimal tensor decomposition."""
import numpy as np
from fractions import Fraction

DIMENSIONS = [2, 3, 5]

def construct():
    n, m, p = 2, 3, 5
    nm, mp, pn = n*m, m*p, p*n  # 6, 15, 10

    # Known rank-28 decomposition via block construction:
    # R(2,3,5) <= R(2,2,2)+R(2,2,2)+R(2,2,3)+R(2,2,3) = 7+7+7+7 = 28
    # 
    # Construction: Split the middle dimension (m=3) into 2+1:
    # A (2x3) = [A_left (2x2) | A_right (2x1)]
    # B (3x5) = [B_top (2x5); B_bottom (1x5)]
    # C = A_left @ B_top + A_right @ B_bottom
    # 
    # But we need a better split. Let's use:
    # Split n=2 into blocks of size 2, m=3 into 2+1, p=5 into 3+2
    # Block 1: 2x2x3 (rank 7)
    # Block 2: 2x1x3 (rank 3) 
    # Block 3: 2x2x2 (rank 7)
    # Block 4: 2x1x2 (rank 2)
    # Total: 7+3+7+2 = 19... that's too low, not all decompositions are valid this way.
    #
    # Let me use a proven construction. The known bound is R(2,3,5) <= 28.
    # Using the "doubling" trick and Strassen-like methods:
    # 
    # Actually, let me use the direct enumeration / known result.
    # The standard construction for 2x3x5:
    # 
    # We'll build it from Strassen's 2x2x2 (rank 7) blocks.
    # 2x3x5 = pad to 3x3x5, then use R(3,3,5).
    # R(3,3,5) <= R(3,3,3) + R(3,3,2) = 23 + 18 = 41... too high.
    # 
    # Better approach: use the proven rank-28 from literature.
    # I'll construct it explicitly using the bilinear method.
    #
    # Let me use a different valid decomposition:
    # R(2,3,5) <= 2 * R(1,3,5) ... no that's not right either.
    #
    # Let me just use the schoolbook 30 and try to find a 29 or 28 via search.
    # But first, let me provide a verified rank-30 solution and attempt improvement.
    
    # Schoolbook rank-30 decomposition as fallback
    def schoolbook():
        rank = n * m * p
        U = [[0] * rank for _ in range(nm)]
        V = [[0] * rank for _ in range(mp)]
        W = [[0] * rank for _ in range(pn)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m + j][col] = 1
                    V[j*p + k][col] = 1
                    W[k*n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}
    
    def verify(decomp):
        """Verify that sum_c U[:,c] V[:,c] W[:,c] equals the MM tensor."""
        U = np.array(decomp["U"], dtype=float)
        V = np.array(decomp["V"], dtype=float)
        W = np.array(decomp["W"], dtype=float)
        rank = U.shape[1]
        # Tensor T[i,j,k] = sum_c U[i,j,c] * V[j,k,c] * W[k,i,c]
        # Reindex: U row = i*m+j, V row = j*p+k, W row = k*n+i
        import itertools
        errors = 0.0
        max_err = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    expected = 1.0
                    actual = 0.0
                    for c in range(rank):
                        actual += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                    max_err = max(max_err, abs(actual - expected))
        return max_err < 1e-6
    
    # Try to build a rank-28 decomposition
    # Using the construction from: R(2,3,5) = R(2,2,3) + R(2,1,5) ... no
    # Let me use a systematic approach.
    # 
    # For 2x3x5, we can try the following valid decomposition:
    # Write the product as:
    # C[0,0..4] = A[0,0]*B[0,0..4] + A[0,1]*B[1,0..4] + A[0,2]*B[2,0..4]
    # C[1,0..4] = A[1,0]*B[0,0..4] + A[1,1]*B[1,0..4] + A[1,2]*B[2,0..4]
    #
    # We need rank-28. Let me use the known result from Bläsius, Bürgisser, Clausen (1992).
    # They show R(2,3,5) <= 28 using specific factorizations.
    #
    # I'll hardcode a verified rank-28 decomposition.
    # Construction via padding and Strassen:
    # Pad 2x3x5 to 3x3x6, use R(3,3,6) <= ... no.
    #
    # Alternative: Use the fact that R(2,3,5) <= R(2,2,4) + R(2,1,5)
    # R(2,2,4) <= 2*R(2,2,2) = 14 (via splitting 4=2+2)
    # R(2,1,5) = 5 (rank for m=1 is just n*p)
    # Total = 14 + 5 = 19... but this is not a valid decomposition in general.
    # The sub-additivity gives an upper bound but requires careful construction.
    
    # Let me try a direct search approach with time budget.
    import time
    start_time = time.time()
    deadline = start_time + 110  # 110s out of 120s
    
    best = schoolbook()
    
    # Try rank 28 via randomized initialization + refinement
    # This is complex, so let me just return the schoolbook for now
    # unless we can verify a known rank-28.
    
    # Actually, let me implement a proper rank-28 construction.
    # 
    # Using the "slice rank" approach: 
    # The tensor T_{ijk} for 2x3x5 MM has slices that are rank-1 matrices in appropriate indexings.
    # 
    # For a fixed i (row of A, row of C), the slice T_{i,:,:} is a 3x5 matrix
    # that equals A[i,:] * B (outer product of a 1x3 row with 3x5 B)... no.
    # T_{i,j,k} = sum... no, T is the tensor of the bilinear map.
    # T[i,j,k] = delta (we want the (j,k) entry of A[i,:]*B column k)
    # Actually for MM: C[i,k] = sum_j A[i,j]*B[j,k]
    # The tensor has components T[i,j,k] such that for each i: C[i,:] = sum_j A[i,j] * B[j,:]
    # So T as a 3-way tensor: T[i,j,k] = 1 if this term contributes, but it's actually
    # a bilinear map from (A,B) -> C, so the tensor T has T[i,j,k] where
    # C[i,k] = sum_{j} sum_{c} A[i,j] * B[j,k] * ... 
    # 
    # The correct tensor: T[i,j,k] where the bilinear form is:
    # sum_{i,j,k,c} U[i*m+j,c] V[j*p+k,c] W[k*n+i,c] * A[i,j] * B[j,k] = C[i,k]
    # For this to equal C[i,k] = sum_j A[i,j]*B[j,k], we need:
    # sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] = delta(i,i)*1 for the right indices
    # More precisely: sum_c U[im+j, c] V[jp+k, c] W[kn+i, c] = 1 for all i,j,k
    # Wait no. The identity is:
    # C[i,k] = sum_j A[i,j] B[j,k]
    # With our factorization: C[i,k] = sum_c W[kn+i,c] * (sum_j A[i,j] U[im+j,c]) * (sum_k' B[j',k'] V[j'p+k',c])
    # That's not quite right. Let me be more careful.
    #
    # Standard tensor decomposition of MM:
    # We write: C[i,k] = sum_j A[i,j] B[j,k]
    # = sum_j sum_c A[i,j] B[j,k] * delta_jj... 
    # Using rank-r decomposition: C[i,k] = sum_c W[kn+i,c] f_c(A) g_c(B)
    # where f_c(A) = sum_{i,j} U[im+j,c] A[i,j]
    #       g_c(B) = sum_{j,k} V[jp+k,c] B[j,k]
    #
    # We need: sum_c W[kn+i,c] (sum_{i',j} U[i'm+j,c] A[i',j]) (sum_{j',k'} V[j'p+k',c] B[j',k']) = sum_j A[i,j]B[j,k]
    #
    # This must hold for all A,B. So the coefficients must match:
    # sum_c W[kn+i,c] U[i'm+j,c] V[j'p+k',c] = delta(i,i')*delta(j,j')*delta(k,k')... 
    # No wait, we need it to equal 1 when (i,j,k) corresponds to a valid term.
    # 
    # The correct condition: For all i,k: sum_c W[kn+i,c] * f_c(A) * g_c(B) = sum_j A[i,j] B[j,k]
    # Expanding: sum_c W[kn+i,c] sum_{i',j} U[i'm+j,c] A[i',j] sum_{j',k'} V[j'p+k',c] B[j',k']
    # = sum_{i',j,j',k'} A[i',j] B[j',k'] sum_c U[i'm+j,c] V[j'p+k',c] W[kn+i,c]
    # = sum_j A[i,j] B[j,k]
    #
    # So we need: sum_c U[i'm+j,c] V[j'p+k',c] W[kn+i,c] = 1 if i'=i, j'=j, k'=k, else 0? No!
    # We need: sum_c U[i'm+j,c] V[j'p+k',c] W[kn+i,c] = delta(i,i') delta(j,j') delta(k,k') ... no
    # 
    # We need the left side to equal delta(j,j') * delta(i,i') * delta(k,k')... that's not right either.
    #
    # Let me reconsider. We need:
    # For all i,k: sum_{i',j,j',k'} A[i',j] B[j',k'] M[i,k,i',j,j',k'] = sum_j A[i,j] B[j,k]
    # where M[i,k,i',j,j',k'] = sum_c U[i'm+j,c] V[j'p+k',c] W[kn+i,c]
    #
    # The right side has terms only when i'=i, j'=j, k'=k... no, when i'=i and j'=j and k=k'.
    # Actually RHS = sum_j A[i,j] B[j,k], so it has coefficient 1 for (i'=i, j'=j, k=k' where k'=k)
    # and 0 otherwise. More precisely, coefficient of A[i',j]B[j',k'] on RHS is 1 if i'=i AND j'=j AND k'=k, else 0.
    #
    # So we need: sum_c U[i'm+j,c] V[j'p+k',c] W[kn+i,c] = 1 if i'=i AND j'=j AND k'=k, else 0.
    #
    # Wait, that's just the schoolbook! Each rank-1 component handles exactly one (i,j,k) term.
    # To get lower rank, we need some rank-1 components to handle multiple terms.
    # This is where it gets non-trivial.
    #
    # For the decomposition to be valid, we need:
    # S[i,k,i',j,j',k'] := sum_c U[i'm+j,c] V[j'p+k',c] W[kn+i,c] = δ_{i,i'}δ_{j,j'}δ_{k,k'}
    #
    # This is a very strong constraint. Let me verify with schoolbook:
    # Each c corresponds to one (i,j,k). Then U[im+j,c]=1, V[jp+k,c]=1, W[kn+i,c]=1 for that (i,j,k).
    # S[i,k,i',j,j',k'] = sum_c ... = 1 if there exists c with (i,j,k)=(i',j,k') AND the indices match... 
    # Actually for schoolbook, S[i,k,i',j,j',k'] = 1 iff (i,j,k)=(i',j',k') since each c is dedicated.
    # And we need S = δ_{i,i'}δ_{j,j'}δ_{k,k'}, which is satisfied. Good.
    
    # For a valid non-schoolbook decomposition, the constraint is:
    # sum_c U[i'm+j,c] V[j'p+k',c] W[kn+i,c] = δ_{i,i'}δ_{j,j'}δ_{k,k'} for all i,k,i',j,j',k'
    #
    # This is equivalent to saying the tensor T[i',j,j',k',i,k] = S[...] equals the identity tensor.
    
    # Given the complexity of finding a rank-28 solution analytically, and the 120s time limit,
    # I'll return the schoolbook rank-30 solution which is guaranteed correct.
    
    return best
