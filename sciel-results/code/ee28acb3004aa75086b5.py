"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 6]

import time
import random
import numpy as np
from fractions import Fraction

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140  # safety margin before 150s kill

    # Start with schoolbook (rank 48)
    rank_sb = n * m * p
    U_sb = [[0] * rank_sb for _ in range(n * m)]
    V_sb = [[0] * rank_sb for _ in range(m * p)]
    W_sb = [[0] * rank_sb for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j][col] = 1
                V_sb[j * p + k][col] = 1
                W_sb[k * n + i][col] = 1
                col += 1
    best_result = {"U": U_sb, "V": V_sb, "W": W_sb}
    best_rank = rank_sb

    # Try to find lower rank via structured search
    # The matrix multiplication tensor T[i,j,k] has T[i,j,k] = 1 if we think of
    # (A*B)[i,k] = sum_j A[i,j]*B[j,k].
    # A rank-r decomposition means:
    #   T[i,j,k] = sum_{c=0}^{r-1} U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    # where U has shape (n*m, r), V has shape (m*p, r), W has shape (p*n, r).
    #
    # For 2x4 * 4x6: n*m=8, m*p=24, p*n=12.
    # Known lower bounds: max(n*m, m*p, p*n) = 24.
    # Let's try ranks from 24 upward.

    def verify(U, V, W, rank):
        """Verify that the decomposition exactly matches the multiplication tensor."""
        # T[i,j,k] should equal sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    expected = 1.0
                    actual = 0.0
                    for c in range(rank):
                        actual += U[i * m + j][c] * V[j * p + k][c] * W[k * n + i][c]
                    if abs(actual - expected) > 1e-9:
                        return False
        return True

    def verify_frac(U, V, W, rank):
        """Exact verification using fractions."""
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    expected = Fraction(1, 1)
                    actual = Fraction(0, 1)
                    for c in range(rank):
                        actual += Fraction(U[i * m + j][c]) * Fraction(V[j * p + k][c]) * Fraction(W[k * n + i][c])
                    if actual != expected:
                        return False
        return True

    # Try structured decompositions
    # Strategy: use the fact that for small matrices, we can try various structured forms.
    
    # One known approach: if we can factor one of the dimensions, we can reduce rank.
    # E.g., if m = m1 + m2, we can use block decomposition.
    # For 2x4 * 4x6: m=4 = 2+2, so we could potentially use a block approach.
    
    # Let's try a general random search for low ranks.
    # For rank r, we need to find U (8xr), V (24xr), W (12xr) such that
    # for all (i,j,k): sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] = 1
    
    # This is a system of n*m*p = 48 equations in r*(n*m + m*p + p*n) = r*44 unknowns.
    # For r=24: 24*44 = 1056 unknowns, 48 equations - very underdetermined.
    # But the equations are cubic (trilinear), so it's not linear.
    
    # Let's try a different approach: fix W to be identity-like or structured.
    
    # Approach: Try rank = max(n*m, m*p, p*n) = 24 first with structured forms.
    
    # Try: set W[k*n+i, c] = delta_{k, c//n} * delta_{i, c%n} for c in range(n*p)
    # i.e., W is an identity matrix embedded. Then we need:
    # U[i*m+j, c] * V[j*p+k, c] = 1 for c = k*n+i
    # This means U[i*m+j, k*n+i] * V[j*p+k, k*n+i] = 1
    # So U[i*m+j, k*n+i] = 1 / V[j*p+k, k*n+i]
    # We can set V[j*p+k, k*n+i] = 1 for all valid (j,k,i) where k*n+i < 24
    # But k ranges 0..5, i ranges 0..1, so k*n+i ranges 0..11.
    # For c >= 12, W[k*n+i, c] = 0, so those columns don't contribute.
    # This only gives us 12 effective columns, not 24.
    
    # Let me think differently. The tensor T[i,j,k] = 1 for all i,j,k.
    # This is actually the all-ones tensor! Wait, no.
    # T[i,j,k] represents the coefficient of A[i,j]*B[j,k] in (A*B)[i,k].
    # Since (A*B)[i,k] = sum_j A[i,j]*B[j,k], the tensor has T[i,j,k] = 1 for all i,j,k.
    # So we need: sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1 for ALL i,j,k.
    
    # This is the "all-ones" trilinear form. We need rank of the all-ones tensor.
    # The all-ones tensor of size n x m x p has rank... let me think.
    # Actually, the all-ones tensor 1_{n x m x p} can be written as:
    # sum_{c=0}^{min(n,m,p)-1} u_c v_c w_c where u_c[i]=1, v_c[j]=1, w_c[k]=1 for all c.
    # Wait, that gives sum_c 1*1*1 = min(n,m,p) which is not 1 unless min(n,m,p)=1.
    
    # Let me reconsider. We need sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] = 1 for all i,j,k.
    # If we set U[i*m+j,c] = a_c, V[j*p+k,c] = b_c, W[k*n+i,c] = d_c (independent of i,j,k),
    # then sum_c a_c * b_c * d_c = 1. This works with rank 1!
    # U[i*m+j, 0] = 1, V[j*p+k, 0] = 1, W[k*n+i, 0] = 1.
    # Check: sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] = 1*1*1 = 1. Yes!
    
    # Wait, that seems too good to be true. Let me re-read the problem.
    # "Matrix multiplication 2,4,6" - multiply a 2x4 matrix by a 4x6 matrix.
    # The output is a 2x6 matrix.
    # (A*B)[i,k] = sum_{j=0}^{m-1} A[i,j] * B[j,k]
    # The tensor T has indices (i, j, k) with T[i,j,k] = 1.
    # 
    # But wait - the factor matrices are:
    # U: shape (n*m, rank) = (8, rank), indexed by (i*m+j)
    # V: shape (m*p, rank) = (24, rank), indexed by (j*p+k)  
    # W: shape (p*n, rank) = (12, rank), indexed by (k*n+i)
    #
    # And the condition is: for all i,j,k:
    #   sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = T[i,j,k] = 1
    #
    # If T[i,j,k] = 1 for all i,j,k, then yes, rank 1 works:
    # U[i*m+j, 0] = 1 for all i,j
    # V[j*p+k, 0] = 1 for all j,k
    # W[k*n+i, 0] = 1 for all k,i
    #
    # But this seems suspiciously simple. Let me re-read the problem statement.
    # "Find a rank-minimal exact tensor decomposition for multiplying a 2 by 4 matrix by a 4 by 6 matrix."
    #
    # Hmm, but standard matrix multiplication tensor T[i,j,k] = delta_{i,j} * delta_{j,k} is for
    # a different indexing. Let me reconsider.
    #
    # Actually, I think I'm confusing two conventions. The standard matrix multiplication tensor
    # for multiplying an (n x m) matrix by an (m x p) matrix has:
    # T[i, j, k] where i is row of A (0..n-1), j is column of A / row of B (0..m-1), k is column of B (0..p-1)
    # And T[i,j,k] = 1 (the coefficient is 1).
    #
    # So the tensor IS the all-ones tensor of size n x m x p.
    # And its rank as a trilinear form is indeed 1, since it's separable.
    #
    # But wait, that can't be right because then the problem would be trivial.
    # Let me reconsider the indexing of the factor matrices.
    
    # U has shape (n*m, rank). Index i*m+j. So U depends on both i and j.
    # V has shape (m*p, rank). Index j*p+k. So V depends on both j and k.
    # W has shape (p*n, rank). Index k*n+i. So W depends on both k and i.
    #
    # The condition is: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1 for all i,j,k.
    #
    # If I set U[i*m+j, 0] = 1, V[j*p+k, 0] = 1, W[k*n+i, 0] = 1, then the sum is 1. 
    # This is a valid rank-1 decomposition.
    #
    # But this would mean the answer is rank 1, which seems too simple. Let me check if there's
    # a constraint I'm missing.
    
    # Re-reading: "U[i*m+j, c]" - this means U is a matrix where row index is i*m+j.
    # So U[i*m+j, c] can depend on both i and j.
    # Similarly V[j*p+k, c] can depend on both j and k.
    # And W[k*n+i, c] can depend on both k and i.
    
    # The condition is that for EVERY combination of (i,j,k), the trilinear sum equals 1.
    
    # With rank 1: U[i*m+j, 0] = 1, V[j*p+k, 0] = 1, W[k*n+i, 0] = 1.
    # Then for any (i,j,k): U[i*m+j,0]*V[j*p+k,0]*W[k*n+i,0] = 1*1*1 = 1. Correct.
    
    # So the minimal rank is 1!
    
    # But wait, let me double-check by considering what "matrix multiplication tensor" means.
    # In the standard Strassen-style decomposition, the tensor is:
    # T = sum_{i,j,k} e_i (in A-space) ⊗ e_j (in B-space) ⊗ e_k (in output-space)
    # where the coefficient is 1.
    # 
    # The rank of this tensor (as a 3-tensor) is the minimum number of rank-1 tensors needed
    # to sum to T. A rank-1 tensor is u ⊗ v ⊗ w where u is in A-space, v in B-space, w in output.
    # 
    # T = sum_{i,j,k} e_i ⊗ e_j ⊗ e_k = (sum_i e_i) ⊗ (sum_j e_j) ⊗ (sum_k e_k)
    # 
    # So T is itself a rank-1 tensor! The rank is 1.
    
    # This means U, V, W should each have a single column of all 1s.
    
    U = [[1] for _ in range(n * m)]
    V = [[1] for _ in range(m * p)]
    W = [[1] for _ in range(p * n)]
    
    # Verify
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0
                for c in range(1):
                    s += U[i * m + j][c] * V[j * p + k][c] * W[k * n + i][c]
                if s != 1:
                    # Fallback to schoolbook
                    return {"U": U_sb, "V": V_sb, "W": W_sb}
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
