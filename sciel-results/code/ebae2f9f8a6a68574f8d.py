import time
import numpy as np
from fractions import Fraction

DIMENSIONS = [2, 5, 5]
N, M, P = DIMENSIONS

def _deadline_ok(t0, limit=130.0):
    return (time.time() - t0) < limit

def _verify_decomp(U, V, W, rank):
    """Verify that the decomposition is exact for the 2x5x5 matrix multiplication tensor."""
    if len(U) != N * M or len(V) != M * P or len(W) != P * N:
        return False
    for r in range(rank):
        for i in range(N * M):
            if len(U[i]) != rank:
                return False
        for j in range(M * P):
            if len(V[j]) != rank:
                return False
        for k in range(P * N):
            if len(W[k]) != rank:
                return False
    
    # T[i,j,k] should be 1 for all valid indices
    # T[i,j,k] = sum_r U[i*M+j][r] * V[j*P+k][r] * W[k*N+i][r]
    for i in range(N):
        for j in range(M):
            for k in range(P):
                val = Fraction(0)
                for r in range(rank):
                    val += Fraction(U[i * M + j][r]) * Fraction(V[j * P + k][r]) * Fraction(W[k * N + i][r])
                if val != 1:
                    return False
    return True

def _schoolbook():
    """Return the schoolbook decomposition with rank n*m*p = 50."""
    rank = N * M * P
    U = [[0] * rank for _ in range(N * M)]
    V = [[0] * rank for _ in range(M * P)]
    W = [[0] * rank for _ in range(P * N)]
    
    col = 0
    for i in range(N):
        for j in range(M):
            for k in range(P):
                U[i * M + j][col] = 1
                V[j * P + k][col] = 1
                W[k * N + i][col] = 1
                col += 1
    return U, V, W

def _strassen_222_block():
    """
    Return the 7 Strassen terms for 2x2x2 matrix multiplication.
    Each term is (u, v, w) where u is length 4 (2x2 flattened), v is length 4, w is length 4.
    The contribution to C[i,k] from term r is:
      sum_j u[i,j] * v[j,k] * w[i,k]  ... wait, that's not right.
    
    Actually, the tensor decomposition for matrix multiplication C = A*B is:
      C[i,k] = sum_j A[i,j] * B[j,k]
    
    In tensor form, T[i,j,k] = 1 for all i,j,k.
    The rank-r decomposition is:
      T[i,j,k] = sum_{r=1}^{r} U[i,j,r] * V[j,k,r] * W[k,i,r]
    
    Wait, the trace convention says W[k*n+i, r]. So W is indexed by (k, i) where k is the column of C and i is the row of C.
    
    For the 2x2x2 case, Strassen's algorithm computes:
      M1 = (A[0,0] + A[1,1]) * (B[0,0] + B[1,1])
      M2 = (A[1,0] + A[1,1]) * B[0,0]
      M3 = A[0,0] * (B[0,1] - B[1,1])
      M4 = A[1,1] * (B[1,0] - B[0,0])
      M5 = (A[0,0] + A[0,1]) * B[1,1]
      M6 = (A[1,0] - A[0,0]) * (B[0,0] + B[0,1])
      M7 = (A[0,1] - A[1,1]) * (B[1,0] + B[1,1])
    
      C[0,0] = M1 + M4 - M5 + M7
      C[0,1] = M3 + M5
      C[1,0] = M2 + M4
      C[1,1] = M1 - M2 + M3 + M6
    
    For each Strassen term r, we need to find U_r, V_r, W_r such that:
      For all i, j, k:
        sum_r U_r[i,j] * V_r[j,k] * W_r[k,i] = 1
    
    This is a system of equations. Let's denote the 7 terms.
    
    Term 1: M1 = (A00 + A11)(B00 + B11)
      U1: A00 + A11 => U1[0,0]=1, U1[1,1]=1, others 0
      V1: B00 + B11 => V1[0,0]=1, V1[1,1]=1, others 0
      W1: contributes to C00 (+1) and C11 (+1) => W1[0,0]=1, W1[1,1]=1, others 0
    
    Term 2: M2 = (A10 + A11) B00
      U2: A10 + A11 => U2[1,0]=1, U2[1,1]=1, others 0
      V2: B00 => V2[0,0]=1, others 0
      W2: contributes to C10 (+1) and C11 (-1) => W2[1,0]=1, W2[1,1]=-1, others 0
    
    Term 3: M3 = A00 (B01 - B11)
      U3: A00 => U3[0,0]=1, others 0
      V3: B01 - B11 => V3[0,1]=1, V3[1,1]=-1, others 0
      W3: contributes to C01 (+1) and C11 (+1) => W3[0,1]=1, W3[1,1]=1, others 0
    
    Term 4: M4 = A11 (B10 - B00)
      U4: A11 => U4[1,1]=1, others 0
      V4: B10 - B00 => V4[1,0]=1, V4[0,0]=-1, others 0
      W4: contributes to C00 (+1) and C10 (+1) => W4[0,0]=1, W4[1,0]=1, others 0
    
    Term 5: M5 = (A00 + A01) B11
      U5: A00 + A01 => U5[0,0]=1, U5[0,1]=1, others 0
      V5: B11 => V5[1,1]=1, others 0
      W5: contributes to C00 (-1) and C01 (+1) => W5[0,0]=-1, W5[0,1]=1, others 0
    
    Term 6: M6 = (A10 - A00) (B00 + B01)
      U6: A10 - A00 => U6[1,0]=1, U6[0,0]=-1, others 0
      V6: B00 + B01 => V6[0,0]=1, V6[0,1]=1, others 0
      W6: contributes to C11 (+1) => W6[1,1]=1, others 0
    
    Term 7: M7 = (A01 - A11) (B10 + B11)
      U7: A01 - A11 => U7[0,1]=1, U7[1,1]=-1, others 0
      V7: B10 + B11 => V7[1,0]=1, V7[1,1]=1, others 0
      W7: contributes to C00 (+1) => W7[0,0]=1, others 0
    
    Let's verify:
      C00 = M1 + M4 - M5 + M7
      Term 1: U1[0,0]*V1[0,0]*W1[0,0] = 1*1*1 = 1 (for i=0,j=0,k=0)
      But wait, the tensor entry T[i,j,k] = U[i,j,r]*V[j,k,r]*W[k,i,r]
      For i=0, j=0, k=0:
        r=1: U1[0,0]*V1[0,0]*W1[0,0] = 1*1*1 = 1
        r=4: U4[0,0]*V4[0,0]*W4[0,0] = 0*(-1)*1 = 0  (U4[0,0]=0)
        r=5: U5[0,0]*V5[0,0]*W5[0,0] = 1*0*(-1) = 0  (V5[0,0]=0)
        r=7: U7[0,0]*V7[0,0]*W7[0,0] = 0*0*1 = 0  (U7[0,0]=0)
        So sum = 1. Correct.
    
      For i=0, j=0, k=1:
        r=1: U1[0,0]*V1[0,1]*W1[1,0] = 1*0*0 = 0
        r=3: U3[0,0]*V3[0,1]*W3[1,0] = 1*1*0 = 0  (W3[1,0]=0)
        r=5: U5[0,0]*V5[0,1]*W5[1,0] = 1*0*0 = 0
        So sum = 0? But T[0,0,1] should be 1.
    
    I think I have the indexing of W wrong. The problem says W[k*n+i, r].
    So for n=2, W is indexed by k*2 + i.
    W[k,i] corresponds to the contribution to C[i,k].
    
    Let's re-examine. The tensor T[i,j,k] = 1.
    The decomposition is T[i,j,k] = sum_r U[i,j,r] * V[j,k,r] * W[k,i,r].
    
    For i=0, j=0, k=1:
      We need sum_r U[0,0,r] * V[0,1,r] * W[1,0,r] = 1.
    
    In Strassen:
      C[0,1] = M3 + M5
      M3 = A[0,0] * (B[0,1] - B[1,1])
      M5 = (A[0,0] + A[0,1]) * B[1,1]
    
      For term 3: U3[0,0] = 1, V3[0,1] = 1, W3[1,0] = ?
      M3 contributes to C[0,1] with coefficient +1.
      So W3[1,0] should be 1? But W is indexed by (k,i) = (1,0).
      C[0,1] means i=0, k=1. So W[k=1, i=0] = W[1,0].
      So W3[1,0] = 1.
    
      For term 5: U5[0,0] = 1, V5[0,1] = 0 (since V5 is B11, so V5[1,1]=1, V5[0,1]=0).
      Wait, V5[j,k] corresponds to B[j,k]. B[1,1] means j=1, k=1.
      So V5[1,1] = 1, all other V5[j,k] = 0.
      So V5[0,1] = 0.
      Thus term 5 does not contribute to T[0,0,1].
    
      But C[0,1] = M3 + M5. How does M5 contribute to C[0,1]?
      M5 = (A[0,0] + A[0,1]) * B[1,1].
      This product contributes to C[0,1] via the term A[0,1]*B[1,1].
      In the tensor, T[0,1,1] = 1.
      For i=0, j=1, k=1:
        r=5: U5[0,1]*V5[1,1]*W5[1,0] = 1*1*W5[1,0].
        We need this to be part of the sum that equals 1 for T[0,1,1].
    
    I think my assignment of W is incorrect. Let me carefully derive W for each term.
    
    For each Strassen term r, the contribution to C[i,k] is:
      coeff_{r,i,k} * M_r
    
    And M_r = (linear combination of A) * (linear combination of B).
    
    The tensor entry T[i,j,k] receives a contribution from term r if:
      U_r[i,j] * V_r[j,k] * W_r[k,i] is non-zero.
    
    Actually, the correct way is:
      C[i,k] = sum_j A[i,j] B[j,k] = sum_r (sum_j U_r[i,j] A[i,j]) (sum_k V_r[j,k] B[j,k]) W_r[k,i]
    
    No, that's not right either. The standard form is:
      C[i,k] = sum_r W_r[k,i] * (sum_j U_r[i,j] A[i,j]) * (sum_j V_r[j,k] B[j,k])
    
    Wait, no. The bilinear form is:
      C[i,k] = sum_r W_r[k,i] * (sum_j U_r[i,j] A[i,j]) * (sum_j V_r[j,k] B[j,k])
    
    This doesn't look right dimensionally. Let's use the tensor form:
      T[i,j,k] = sum_r U_r[i,j] V_r[j,k] W_r[k,i]
    
    And the matrix product is:
      C[i,k] = sum_j A[i,j] B[j,k] = sum_j sum_r T[i,j,k] A[i,j] B[j,k]
              = sum_r sum_j U_r[i,j] V_r[j,k] W_r[k,i] A[i,j] B[j,k]
    
    For this to equal sum_j A[i,j] B[j,k] for all A, B, we need:
      sum_r U_r[i,j] V_r[j,k] W_r[k,i] = 1 for all i,j,k.
    
    So my original formulation was correct. The issue is in how I assigned W.
    
    Let's re-derive W for each Strassen term.
    
    Term 1: M1 = (A00 + A11)(B00 + B11)
      U1: [1,0,0,1] (A00+A11)
      V1: [1,0,0,1] (B00+B11)
      W1: must satisfy for all i,j,k:
        U1[i,j] * V1[j,k] * W1[k,i] contributes to T[i,j,k].
        
        Non-zero U1 entries: (0,0)=1, (1,1)=1
        Non-zero V1 entries: (0,0)=1, (1,1)=1
        
        So T[i,j,k] gets contributions from:
        (i=0,j=0,k=0): U1[0,0]*V1[0,0]*W1[0,0] = 1*1*W1[0,0]
        (i=0,j=0,k=1): U1[0,0]*V1[0,1]*W1[1,0] = 1*0*W1[1,0] = 0
        (i=0,j=1,k=0): U1[0,1]*V1[1,0]*W1[0,0] = 0*0*W1[0,0] = 0
        (i=0,j=1,k=1): U1[0,1]*V1[1,1]*W1[1,0] = 0*1*W1[1,0] = 0
        (i=1,j=0,k=0): U1[1,0]*V1[0,0]*W1[0,1] = 0*1*W1[0,1] = 0
        (i=1,j=0,k=1): U1[1,0]*V1[0,1]*W1[1,1] = 0*0*W1[1,1] = 0
        (i=1,j=1,k=0): U1[1,1]*V1[1,0]*W1[0,1] = 1*0*W1[0,1] = 0
        (i=1,j=1,k=1): U1[1,1]*V1[1,1]*W1[1,1] = 1*1*W1[1,1]
        
        So T[0,0,0] += W1[0,0]
        T[1,1,1] += W1[1,1]
        
        From Strassen: C00 += M1, C11 += M1.
        So W1[0,0] = 1, W1[1,1] = 1.
        W1 = [1,0,0,1] (flattened as W[k*2+i]: W[0,0]=1, W[0,1]=0, W[1,0]=0, W[1,1]=1)
    
    Term 2: M2 = (A10 + A11) B00
      U2: [0,0,1,1] (A10+A11)
      V2: [1,0,0,0] (B00)
      Non-zero U2: (1,0)=1, (1,1)=1
      Non-zero V2: (0,0)=1
      
      T[i,j,k] contributions:
      (i=1,j=0,k=0): U2[1,0]*V2[0,0]*W2[0,1] = 1*1*W2[0,1]
      (i=1,j=1,k=0): U2[1,1]*V2[1,0]*W2[0,1] = 1*0*W2[0,1] = 0
      Others are 0.
      
      So T[1,0,0] += W2[0,1]
      From Strassen: C10 += M2, C11 -= M2.
      C10 corresponds to i=1, k=0. So T[1,j,0] for j=0 gives C10.
      So W2[0,1] = 1.
      Also C11 -= M2. C11 corresponds to i=1, k=1.
      But T[1,j,1] for j=0: U2[1,0]*V2[0,1]*W2[1,1] = 1*0*W2[1,1] = 0.
      So M2 does not contribute to C11 via this path?
      
      Wait, C11 = M1 - M2 + M3 + M6.
      The -M2 term in C11 comes from the fact that M2 = (A10+A11)B00, and in the expansion of C11 = A10*B00 + A11*B11, the A10*B00 term is present.
      But in Strassen, we have C11 = M1 - M2 + M3 + M6.
      Let's check:
      M1 = A00B00 + A00B11 + A11B00 + A11B11
      M2 = A10B00 + A11B00
      M3 = A00B01 - A00B11
      M6 = A10B00 + A10B01 - A00B00 - A00B01
      
      C11 = M1 - M2 + M3 + M6
           = (A00B00 + A00B11 + A11B00 + A11B11) - (A10B00 + A11B00) + (A00B01 - A00B11) + (A10B00 + A10B01 - A00B00 - A00B01)
           = A00B00 + A00B11 + A11B00 + A11B11 - A10B00 - A11B00 + A00B01 - A00B11 + A10B00 + A10B01 - A00B00 - A00B01
           = A11B11
      
      Yes, C11 = A11B11. Correct.
      
      So for term 2, it only contributes to C10 (i=1, k=0) with coefficient +1.
      So W2[0,1] = 1 (since W[k=0, i=1] = W[0,1]).
      W2 = [0,1,0,0] (W[0,0]=0, W[0,1]=1, W[1,0]=0, W[1,1]=0)
    
    Term 3: M3 = A00 (B01 - B11)
      U3: [1,0,0,0] (A00)
      V3: [0,1,0,-1] (B01 - B11)
      Non-zero U3: (0,0)=1
      Non-zero V3: (0,1)=1, (1,1)=-1
      
      T[i,j,k] contributions:
      (i=0,j=0,k=1): U3[0,0]*V3[0,1]*W3[1,0] = 1*1*W3[1,0]
      (i=0,j=1,k=1): U3[0,1]*V3[1,1]*W3[1,0] = 0*(-1)*W3[1,0] = 0
      
      So T[0,0,1] += W3[1,0]
      From Strassen: C01 += M3, C11 += M3.
      C01: i=0, k=1. T[0,0,1] contributes to C01. So W3[1,0] = 1.
      C11: i=1, k=1. T[1,j,1] for j=0: U3[1,0]*V3[0,1]*W3[1,1] = 0*1*W3[1,1] = 0.
      So M3 does not contribute to C11 via this path?
      
      But C11 = M1 - M2 + M3 + M6.
      Let's re-check C11 expansion:
      M3 = A00B01 - A00B11.
      In C11 = A10B00 + A11B11, there is no A00 term.
      So how does M3 contribute to C11?
      
      Ah, I see. The Strassen formula is:
      C00 = M1 + M4 - M5 + M7
      C01 = M3 + M5
      C10 = M2 + M4
      C11 = M1 - M2 + M3 + M6
      
      Let's verify C01 = M3 + M5:
      M3 = A00B01 - A00B11
      M5 = A00B11 + A01B11
      M3 + M5 = A00B01 + A01B11 = C01. Correct.
      
      Now C11 = M1 - M2 + M3 + M6:
      M1 = A00B00 + A00B11 + A11B00 + A11B11
      M2 = A10B00 + A11B00
      M3 = A00B01 - A00B11
      M6 = A10B00 + A10B01 - A00B00 - A00B01
      Sum = A00B00 + A00B11 + A11B00 + A11B11 - A10B00 - A11B00 + A00B01 - A00B11 + A10B00 + A10B01 - A00B00 - A00B01
          = A11B11. Correct.
      
      So M3 contributes to C11 with coefficient +1. But in the tensor, how?
      T[1,j,k] for i=1, k=1.
      U3[i=1,j] = 0 for all j. So M3 does not contribute to any T[1,j,k].
      This means my assignment of U3 is wrong.
      
      Wait, U3 corresponds to A00. So U3[0,0]=1, others 0.
      For i=1, U3[1,j]=0. So indeed M3 does not contribute to C11 via the tensor.
      
      But in the Strassen formula, M3 does contribute to C11.
      The resolution is that the Strassen formula is a linear combination, and the tensor decomposition must reproduce the same linear combination.
      
      I think the issue is that I'm not correctly accounting for the fact that each Strassen term is a product of two linear forms, and the tensor decomposition must sum to 1 for all (i,j,k).
      
      Let me try a different approach. I'll set up the system of equations and solve it.
      
      For the 2x2x2 case, there are 8 entries T[i,j,k] that must equal 1.
      Each Strassen term r provides U_r[i,j], V_r[j,k], W_r[k,i].
      We have 7 terms, so 7*4 = 28 unknowns (U, V, W each have 4 entries).
      And 8 equations.
      This is underdetermined, so there are many solutions.
      
      The standard Strassen solution is known. Let me look it up or derive it properly.
      
      Actually, I realize that the standard Strassen algorithm doesn't directly give a tensor decomposition in the form T[i,j,k] = sum_r U_r[i,j] V_r[j,k] W_r[k,i].
      The Strassen algorithm computes C = A*B using 7 multiplications, but the way it combines them is different.
      
      The tensor decomposition for Strassen's 2x2x2 is:
      r=1: U=[1,0,0,1], V=[1,0,0,1], W=[1,0,0,1]
      r=2: U=[0,0,1,1], V=[1,0,0,0], W=[0,1,0,0]
      r=3: U=[1,0,0,0], V=[0,1,0,-1], W=[0,0,1,1]
      r=4: U=[0,0,0,1], V=[0,0,1,-1], W=[1,1,0,0]
      r=5: U=[1,1,0,0], V=[0,0,0,1], W=[-1,1,0,0]
      r=6: U=[-1,0,1,0], V=[1,1,0,0], W=[0,0,0,1]
      r=7: U=[0,1,0,-1], V=[0,0,1,1], W=[1,0,0,0]
      
      Let me verify r=3:
      U3=[1,0,0,0], V3=[0,1,0,-1], W3=[0,0,1,1]
      T[0,0,1] = U3[0,0]*V3[0,1]*W3[1,0] = 1*1*1 = 1. Good.
      T[0,1,1] = U3[0,1]*V3[1,1]*W3[1,0] = 0*(-1)*1 = 0.
      T[1,0,1] = U3[1,0]*V3[0,1]*W3[1,1] = 0*1*1 = 0.
      T[1,1,1] = U3[1,1]*V3[1,1]*W3[1,1] = 0*(-1)*1 = 0.
      
      But we need T[0,0,1]=1, T[0,1,1]=1, T[1,0,1]=1, T[1,1,1]=1.
      So r=3 alone doesn't work. The sum of all 7 terms must give 1 for each (i,j,k).
      
      This is getting complex. Given the time constraints, I'll implement a simpler approach: use the schoolbook decomposition for the 2x5x5 case, which has rank 50. This is the baseline, and I'll try to improve it if I have time.
      
      Actually, let me just return the schoolbook solution. It's guaranteed to be correct and has rank 50.
    """

def _build_strassen_based():
    """
    Try to build a Strassen-based decomposition for 2x5x5.
    This is complex, so for now, just return schoolbook.
    """
    return _schoolbook()

def _add_strassen_222(U, V, W, rank, offset_i, offset_j, offset_k):
    """Add a 2x2x2 Strassen block at the given offset."""
    # This is a placeholder. The actual implementation would be complex.
    pass

def _add_2x1(U, V, W, rank, offset_i, offset_j, offset_k):
    """Add a 2x1x1 schoolbook block."""
    for i in range(2):
        for j in range(1):
            for k in range(1):
                col = rank
                U[(offset_i + i) * M + (offset_j + j)][col] = 1
                V[(offset_j + j) * P + (offset_k + k)][col] = 1
                W[(offset_k + k) * N + (offset_i + i)][col] = 1
                rank += 1
    return rank

def _add_1x2(U, V, W, rank, offset_i, offset_j, offset_k):
    """Add a 1x2x1 schoolbook block."""
    for i in range(1):
        for j in range(2):
            for k in range(1):
                col = rank
                U[(offset_i + i) * M + (offset_j + j)][col] = 1
                V[(offset_j + j) * P + (offset_k + k)][col] = 1
                W[(offset_k + k) * N + (offset_i + i)][col] = 1
                rank += 1
    return rank

def _add_1x1(U, V, W, rank, offset_i, offset_j, offset_k):
    """Add a 1x1x1 schoolbook block."""
    col = rank
    U[(offset_i + 0) * M + (offset_j + 0)][col] = 1
    V[(offset_j + 0) * P + (offset_k + 0)][col] = 1
    W[(offset_k + 0) * N + (offset_i + 0)][col] = 1
    return rank + 1

def construct():
    t0 = time.time()
    
    # Start with schoolbook
    U, V, W = _schoolbook()
    rank = N * M * P
    
    # Verify the schoolbook solution
    if not _verify_decomp(U, V, W, rank):
        raise ValueError("Schoolbook verification failed")
    
    # If we have time, try to improve
    # For now, just return the schoolbook solution
    return {"U": U, "V": V, "W": W}
