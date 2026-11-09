import time
import numpy as np
from fractions import Fraction

DIMENSIONS = [2, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 130  # safety margin before 150s kill
    
    # Schoolbook decomposition (rank = n*m*p = 50)
    def schoolbook():
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
    
    # Verify a decomposition is correct
    def verify(U, V, W):
        rank = len(U[0])
        # Check: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] == 1 for all i,j,k
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0
                    for c in range(rank):
                        s += U[i * m + j][c] * V[j * p + k][c] * W[k * n + i][c]
                    if s != 1:
                        return False
        return True
    
    # Try to find a rank-48 decomposition
    # Strategy: Start from schoolbook, try to merge pairs of terms
    # For M_{2,5,5}, rank 48 is known to be achievable.
    # Construction: For each j, the 2*p terms (for i=0,1 and all k) can potentially
    # be reduced. Specifically, we can use the identity that for the 2x p output,
    # we can save terms by combining.
    
    # Known construction for rank(M_{2,m,p}) = 2mp - 2:
    # Use schoolbook for all but two (j,k) pairs, and for those two, use a 2-term
    # decomposition instead of 4 terms.
    # 
    # More precisely: pick two k values, say k=0 and k=1, and one j, say j=0.
    # The 4 terms for (j=0, k=0, i=0), (j=0, k=0, i=1), (j=0, k=1, i=0), (j=0, k=1, i=1)
    # can be replaced by 2 terms using the fact that:
    # A[0,0]*B[0,0] contributes to C[0,0]
    # A[1,0]*B[0,0] contributes to C[0,1]
    # A[0,0]*B[0,1] contributes to C[1,0]
    # A[1,0]*B[0,1] contributes to C[1,1]
    #
    # We want 2 rank-1 terms to produce these 4 outputs.
    # Term 1: u1, v1, w1 such that u1[0*m+0]=a, u1[1*m+0]=b, v1[0*p+0]=c, v1[0*p+1]=d, w1[0*n+0]=e, w1[0*n+1]=f, w1[1*n+0]=g, w1[1*n+1]=h
    # This gives: C[0,0] += a*c*e, C[0,1] += b*c*f, C[1,0] += a*d*g, C[1,1] += b*d*h
    # We need: a*c*e=1, b*c*f=1, a*d*g=1, b*d*h=1
    # With 1 free variable (e.g., choose c=1), we have: a*e=1, b*f=1, a*d*g=1, b*d*h=1
    # Set a=1, e=1, b=1, f=1, then d*g=1, d*h=1, so g=h, and d*g=1.
    # Choose d=1, g=1, h=1. Then w1 = [1,1,1,1,0,0,0,0,0,0] (for indices 0..9)
    # u1 = [1,0,0,0,0,0,0,0,0,0] (only u1[0]=1, u1[5]=1 for i=0,j=0 and i=1,j=0)
    # Wait, u1[i*m+j]: u1[0*5+0]=u1[0]=1, u1[1*5+0]=u1[5]=1
    # v1[j*p+k]: v1[0*5+0]=v1[0]=1, v1[0*5+1]=v1[1]=1
    # w1[k*n+i]: w1[0*2+0]=w1[0]=1, w1[0*2+1]=w1[1]=1, w1[1*2+0]=w1[2]=1, w1[1*2+1]=w1[3]=1
    # So term 1: u1 has 1 at positions 0 and 5, v1 has 1 at positions 0 and 1, w1 has 1 at positions 0,1,2,3.
    # This contributes: for (i=0,j=0,k=0): u1[0]*v1[0]*w1[0] = 1*1*1 = 1 ✓
    # for (i=1,j=0,k=0): u1[5]*v1[0]*w1[1] = 1*1*1 = 1 ✓
    # for (i=0,j=0,k=1): u1[0]*v1[1]*w1[2] = 1*1*1 = 1 ✓
    # for (i=1,j=0,k=1): u1[5]*v1[1]*w1[3] = 1*1*1 = 1 ✓
    # But this also contributes to other (i,j,k) combinations!
    # For example, (i=0,j=1,k=0): u1[1]*v1[5]*w1[0] = 0*0*1 = 0 (ok, u1[1]=0)
    # Actually u1 only has non-zero at 0 and 5, so i*m+j must be 0 or 5, i.e., (i=0,j=0) or (i=1,j=0).
    # v1 only has non-zero at 0 and 1, so j*p+k must be 0 or 1, i.e., (j=0,k=0) or (j=0,k=1).
    # So the only (i,j,k) combos affected are: (0,0,0), (0,0,1), (1,0,0), (1,0,1). 
    # This is exactly what we want! So 1 term replaces 4 terms.
    # 
    # Wait, that saves 3 terms, not 2. Let me recheck.
    # Schoolbook has 4 terms for these 4 (i,j,k) combos.
    # My new term handles all 4 with 1 term.
    # So I save 3 terms, giving rank 47? That seems too good.
    # 
    # Let me verify: the term u1 ⊗ v1 ⊗ w1 has:
    # u1 = e_0 + e_5 (in R^{10})
    # v1 = e_0 + e_1 (in R^{25})
    # w1 = e_0 + e_1 + e_2 + e_3 (in R^{10})
    # The tensor u1 ⊗ v1 ⊗ w1 has entries:
    # T[0,0,0] = 1, T[0,0,1] = 1, T[0,1,0] = 1, T[0,1,1] = 1
    # T[5,0,0] = 1, T[5,0,1] = 1, T[5,1,0] = 1, T[5,1,1] = 1
    # Wait, T[i*m+j, j*p+k, k*n+i]... no, the tensor is T[a,b,c] where a indexes U, b indexes V, c indexes W.
    # T[a,b,c] = u1[a] * v1[b] * w1[c]
    # So T[0,0,0] = 1*1*1 = 1
    # T[0,0,1] = 1*1*1 = 1
    # T[0,0,2] = 1*1*1 = 1
    # T[0,0,3] = 1*1*1 = 1
    # T[5,0,0] = 1*1*1 = 1
    # T[5,0,1] = 1*1*1 = 1
    # T[5,0,2] = 1*1*1 = 1
    # T[5,0,3] = 1*1*1 = 1
    # T[0,1,0] = 1*1*1 = 1
    # T[0,1,1] = 1*1*1 = 1
    # T[0,1,2] = 1*1*1 = 1
    # T[0,1,3] = 1*1*1 = 1
    # T[5,1,0] = 1*1*1 = 1
    # T[5,1,1] = 1*1*1 = 1
    # T[5,1,2] = 1*1*1 = 1
    # T[5,1,3] = 1*1*1 = 1
    # 
    # Now, the matrix multiplication tensor requires:
    # T[i*m+j, j*p+k, k*n+i] = 1 for all i,j,k, and 0 otherwise.
    # For (i=0,j=0,k=0): T[0,0,0] should be 1. ✓
    # For (i=0,j=0,k=1): T[0,1,2] should be 1. ✓ (T[0,1,2]=1)
    # For (i=1,j=0,k=0): T[5,0,1] should be 1. ✓ (T[5,0,1]=1)
    # For (i=1,j=0,k=1): T[5,1,3] should be 1. ✓ (T[5,1,3]=1)
    # But also:
    # T[0,0,1] = 1, but this corresponds to (i*m+j=0, j*p+k=0, k*n+i=1) => (0,0,1)
    # Is there an (i,j,k) with i*m+j=0, j*p+k=0, k*n+i=1?
    # i*m+j=0 => i=0,j=0. j*p+k=0 => k=0. k*n+i=0*2+0=0≠1. So T[0,0,1] should be 0!
    # But our term gives T[0,0,1]=1. This is WRONG.
    #
    # So the naive construction doesn't work because it creates spurious non-zero entries.
    # The tensor must be EXACTLY the matrix multiplication tensor, which has T[a,b,c]=1 iff
    # there exist i,j,k with a=i*m+j, b=j*p+k, c=k*n+i.
    #
    # This means we can't just use arbitrary rank-1 terms; they must not create spurious entries.
    #
    # This is much harder. Let me just return the schoolbook decomposition.
    
    return schoolbook()
# EVOLVE-BLOCK-END
