"""Rank-44 decomposition for 2x3 * 3x8 matrix multiplication via Kronecker/Strassen structure."""
DIMENSIONS = [2, 3, 8]

# EVOLVE-BLOCK-START
import numpy as np
import time
_TIME_LIMIT = 110  # seconds, safety margin before 120s evaluator limit
_START_TIME = None

def _strassen_2x2():
    """Return the rank-7 Strassen decomposition for 2x2 * 2x2.
    Returns U (4x7), V (4x7), W (4x7) where:
    T[a,b,d] = sum_t U[a,t] * V[b,t] * W[d,t]
    Indexing: a = i*2+j (i row of A, j col of A), b = j*2+k (j row of B, k col of B), d = k*2+i
    """
    # Strassen's 7 products for C = A*B where A,B,C are 2x2
    # A = [[a11, a12], [a21, a22]], B = [[b11, b12], [b21, b22]]
    # m1 = (a11+a22)(b11+b22)
    # m2 = (a21+a22)b11
    # m3 = a11(b12-b22)
    # m4 = a22(b21-b11)
    # m5 = (a11+a12)b22
    # m6 = (a21-a11)(b11+b12)
    # m7 = (a12-a22)(b21+b22)
    # C11 = m1 + m4 - m5 + m7
    # C12 = m3 + m5
    # C21 = m2 + m4
    # C22 = m1 - m2 + m3 + m6
    
    # For matrix multiplication tensor M_{2,2,2}:
    # T[i,j,k, l,m,n] = 1 if j==l (the shared index)
    # In our 3-tensor form: T[a,b,d] where a=i*2+j, b=j*2+k, d=k*2+i
    # This is equivalent to: T[i,j,k; j,k,i] = 1 (using the shared j)
    
    # Actually let me think more carefully.
    # The tensor for 2x2 * 2x2 multiplication:
    # (A*B)[i,k] = sum_j A[i,j] * B[j,k]
    # So T[a,b,d] = 1 when a = i*2+j, b = j*2+k, d = k*2+i
    # i.e., T[i,j,k; j,k,i] = 1 in multi-index form
    
    # Strassen decomposition: we need U[a,t], V[b,t], W[d,t] for t=1..7
    # such that sum_t U[a,t]*V[b,t]*W[d,t] = 1 iff (a,b,d) corresponds to valid (i,j,k)
    
    # Let's use the standard Strassen scheme.
    # For each of the 7 Strassen products m_t, we have:
    # m_t = (linear combo of A entries) * (linear combo of B entries)
    # and C entries are linear combos of m_t.
    
    # U[a,t] = coefficient of A[i,j] in the first factor of m_t
    # V[b,t] = coefficient of B[j,k] in the second factor of m_t  
    # W[d,t] = coefficient of C[i,k] in the output from m_t
    
    # m1 = (A11+A22)(B11+B22): 
    #   A part: A11 + A22 -> U[0*2+0,0]=1, U[1*2+1,0]=1 (i=0,j=0 and i=1,j=1)
    #   B part: B11 + B22 -> V[0*2+0,0]=1, V[1*2+1,0]=1 (j=0,k=0 and j=1,k=1)
    #   C part: C11 + C22 - C12 - C21... wait no.
    #   C11 += m1, C22 += m1, so W[0*2+0,0]=1 (C11, k=0,i=0), W[1*2+1,0]=1 (C22, k=1,i=1)
    #   Wait: d = k*2+i. C11 means i=0,k=0 -> d=0. C22 means i=1,k=1 -> d=3.
    #   C12 means i=0,k=1 -> d=2. C21 means i=1,k=0 -> d=1.
    
    U = np.zeros((4, 7))
    V = np.zeros((4, 7))
    W = np.zeros((4, 7))
    
    # m1 = (a11+a22)(b11+b22) -> contributes to c11, c22
    U[0, 0] = 1  # a11 (i=0,j=0)
    U[3, 0] = 1  # a22 (i=1,j=1)
    V[0, 0] = 1  # b11 (j=0,k=0)
    V[3, 0] = 1  # b22 (j=1,k=1)
    W[0, 0] = 1  # c11 (k=0,i=0)
    W[3, 0] = 1  # c22 (k=1,i=1)
    
    # m2 = (a21+a22)b11 -> contributes to c21, c22
    U[2, 1] = 1  # a21 (i=1,j=0)
    U[3, 1] = 1  # a22 (i=1,j=1)
    V[0, 1] = 1  # b11 (j=0,k=0)
    W[1, 1] = 1  # c21 (k=0,i=1)
    W[3, 1] = -1 # c22 (k=1,i=1)  ... wait
    
    # Let me re-derive. C21 = m2 + m4, C22 = m1 - m2 + m3 + m6
    # So m2 contributes +1 to C21 and -1 to C22.
    W[1, 1] = 1   # c21
    W[3, 1] = -1  # c22
    
    # m3 = a11(b12-b22) -> contributes to c12, c22
    U[0, 2] = 1  # a11
    V[1, 2] = 1  # b12 (j=0,k=1)
    V[3, 2] = -1 # b22 (j=1,k=1)
    W[2, 2] = 1  # c12 (k=1,i=0)
    W[3, 2] = 1  # c22 (k=1,i=1)
    
    # m4 = a22(b21-b11) -> contributes to c11, c21
    U[3, 3] = 1  # a22
    V[2, 3] = 1  # b21 (j=1,k=0)
    V[0, 3] = -1 # b11 (j=0,k=0)
    W[0, 3] = 1  # c11
    W[1, 3] = 1  # c21
    
    # m5 = (a11+a12)b22 -> contributes to c11, c12
    U[0, 4] = 1  # a11
    U[1, 4] = 1  # a12 (i=0,j=1)
    V[3, 4] = 1  # b22
    W[0, 4] = -1 # c11
    W[2, 4] = 1  # c12
    
    # m6 = (a21-a11)(b11+b12) -> contributes to c22
    U[2, 5] = 1  # a21
    U[0, 5] = -1 # a11
    V[0, 5] = 1  # b11
    V[1, 5] = 1  # b12
    W[3, 5] = 1  # c22
    
    # m7 = (a12-a22)(b21+b22) -> contributes to c11
    U[1, 6] = 1  # a12
    U[3, 6] = -1 # a22
    V[2, 6] = 1  # b21
    V[3, 6] = 1  # b22
    W[0, 6] = 1  # c11
    
    return U, V, W

def _standard_2x2():
    """Return the rank-4 standard decomposition for 2x2 * 2x2."""
    U = np.zeros((4, 4))
    V = np.zeros((4, 4))
    W = np.zeros((4, 4))
    for i in range(2):
        for j in range(2):
            for k in range(2):
                t = i * 4 + j * 2 + k  # This won't work, we only have 4 slots
                # Standard: t = j (the shared index), but we need 4 terms for 2x2*2x2
                # Actually rank of 2x2*2x2 is 4 (the schoolbook rank, since Strassen gives 7 which is worse)
                # Wait no, Strassen gives rank 7 which is BETTER than 8 (schoolbook).
                # Standard rank-4: for each (i,k) pair, sum over j. But that's 4 terms total? No.
                # Standard: C[i,k] = sum_j A[i,j]*B[j,k]. That's 4 terms (i,k) in (0,0),(0,1),(1,0),(1,1)
                # but each involves a sum over j. As a tensor: T[i,j,k;j,k,i] = 1.
                # Rank-4 decomposition: for each j in {0,1}, we need to handle all (i,k).
                # Actually the standard rank is n*m*p = 2*2*2 = 8 for the full tensor.
                # But we can do better. Let me just use rank 4 by grouping.
                pass
    
    # Actually for 2x2*2x2, the minimum rank is 7 (Strassen). The "standard" is 8.
    # But we don't need a separate standard - we'll use Strassen for the inner blocks too.
    # Wait, the strategy says rank-4 standard for inner 2x2 blocks. Let me re-read.
    # "leveraging the rank-7 Strassen algorithm for the outer 2x4 structure and the rank-4 standard algorithm for the inner 2x2 blocks"
    # Hmm, but 2x2*2x2 has rank 7 (Strassen) or 8 (standard). There's no rank-4 for 2x2*2x2.
    # I think the strategy is: treat the 3x8 as [3x2, 3x2, 3x2, 3x2] (four blocks of width 2).
    # Then 2x3 * 3x2 has rank... let me think.
    # Actually, let me reconsider the whole approach.
    
    # The 2x3 * 3x8 multiplication:
    # A is 2x3, B is 3x8, C is 2x8.
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # 
    # Block B into 4 blocks of 3x2: B = [B_0, B_1, B_2, B_3] where B_t is 3x2.
    # Then C = [C_0, C_1, C_2, C_3] where C_t = A * B_t (2x2 result).
    #
    # Each C_t = A * B_t is a 2x3 * 3x2 = 2x2 multiplication.
    # The tensor for 2x3*3x2: T[a,b,d] where a=i*3+j, b=j*2+k, d=k*2+i.
    # This is M_{2,3,2}. Its rank is known to be 7 (same as 2x2*2x2 Strassen? No, different).
    # Actually M_{2,3,2} rank: the schoolbook rank is 2*3*2=12.
    # Using Strassen-like: we can do 2x3*3x2 by splitting A into [A_0, A_1, A_2] (three 2x1 vectors)
    # and B into [B_0, B_1] (two 3x1 vectors)... this gets complicated.
    
    # Let me just use the Kronecker product approach directly.
    # View A (2x3) as a 2x2 matrix of 1x2 blocks? No, 3 is not a power of 2.
    
    # Alternative: use the fact that M_{2,3,8} can be decomposed as M_{2,3,2} ⊗ M_{1,1,4}
    # i.e., 4 copies of M_{2,3,2} (one for each pair of columns of B).
    # Rank(M_{2,3,2}) * 4.
    # 
    # What is rank(M_{2,3,2})? 
    # M_{2,3,2}: A(2x3) * B(3x2) = C(2x2).
    # C[i,k] = sum_j A[i,j]*B[j,k], j=0..2, k=0..1, i=0..1.
    # Schoolbook: 2*3*2 = 12.
    # Can we do better? 
    # Split: C[:,0] = A*B[:,0] and C[:,1] = A*B[:,1].
    # Each is 2x3 * 3x1 = 2x1. Rank of M_{2,3,1} is 2*3*1=6 (each output is a dot product).
    # But we can share: C[:,0] and C[:,1] share the same A.
    # Actually M_{2,3,2} = M_{2,3,1} ⊗ M_{1,1,2} in some sense? No.
    
    # Let me just try to find a good decomposition for M_{2,3,2} first.
    # Using Strassen on the 2x2 output:
    # C = A * B where A is 2x3, B is 3x2.
    # Write A = [A1, A2, A3] (three column vectors of length 2)
    # B = [B1, B2] (two column vectors of length 3)
    # C[i,k] = sum_j A[i,j] * B[j,k]
    
    # Hmm, this is getting complex. Let me try a different approach.
    # 
    # Key insight from the strategy: "treating the 3x8 multiplication as a 2x4 matrix multiplication 
    # where each entry is a 2x2 block"
    # 
    # I think the idea is:
    # Reshape A (2x3) and B (3x8) appropriately.
    # Actually, 2x3 * 3x8 = 2x8.
    # If we view 8 = 4*2, then C (2x8) can be viewed as (2x4) of 2-element vectors.
    # And B (3x8) can be viewed as (3x4) of 2-element vectors.
    # So C[:, 2t:2t+2] = A * B[:, 2t:2t+2] for t=0..3.
    # Each block is 2x3 * 3x2 = 2x2.
    #
    # So total rank = 4 * rank(M_{2,3,2}).
    #
    # Now I need rank(M_{2,3,2}). Let me compute it.
    # M_{2,3,2}: the tensor T in R^{6} x R^{6} x R^{4} (since 2*3=6, 3*2=6, 2*2=4).
    # T[a,b,d] = 1 when a=i*3+j, b=j*2+k, d=k*2+i.
    # 
    # Lower bound: max(2*3, 3*2, 2*2) = max(6,6,4) = 6.
    # Upper bound: 12 (schoolbook).
    # 
    # Can we achieve rank 7? Let's try.
    # Actually, I recall that rank(M_{2,3,2}) = 7. Let me verify by construction.
    # 
    # One approach: use the 2x2 Strassen on one dimension.
    # C[i,k] = sum_j A[i,j]*B[j,k].
    # For fixed i, C[i,:] = A[i,:] * B (a 1x3 times 3x2 = 1x2).
    # C[0,:] = A[0,:] * B, C[1,:] = A[1,:] * B.
    # 
    # Let me try a rank-7 construction for M_{2,3,2}.
    # 
    # Actually, let me just use a computational approach to find a low-rank decomposition.
    # But since we need it to be exact and fast, let me try to construct one.
    
    # For now, let me use the schoolbook rank-12 for M_{2,3,2}, giving total rank 48.
    # Then I'll try to improve.
    
    # Actually, let me just directly construct a rank-28 decomposition for M_{2,3,8}.
    # 
    # Strategy: M_{2,3,8} = M_{2,3,2} ⊗ I_4 (4 blocks).
    # If rank(M_{2,3,2}) = 7, then total rank = 28.
    #
    # Let me find a rank-7 decomposition of M_{2,3,2}.
    #
    # M_{2,3,2}: A(2x3) * B(3x2) = C(2x2).
    # T[i,j,k; j,k,i] = 1 where i∈{0,1}, j∈{0,1,2}, k∈{0,1}.
    #
    # Let me try:
    # For the 2x2 output, use Strassen-like 7 products.
    # But A has 3 columns, not 2. So I need to adapt.
    #
    # C[0,0] = A[0,0]*B[0,0] + A[0,1]*B[1,0] + A[0,2]*B[2,0]
    # C[0,1] = A[0,0]*B[0,1] + A[0,1]*B[1,1] + A[0,2]*B[2,1]
    # C[1,0] = A[1,0]*B[0,0] + A[1,1]*B[1,0] + A[1,2]*B[2,0]
    # C[1,1] = A[1,0]*B[0,1] + A[1,1]*B[1,1] + A[1,2]*B[2,1]
    #
    # Let me define 7 products:
    # m1 = (A[0,0]+A[1,2]) * (B[0,0]+B[2,1])  -- cross terms
    # m2 = (A[1,0]+A[1,2]) * B[0,0]
    # m3 = A[0,0] * (B[0,1]-B[2,1])
    # m4 = A[1,2] * (B[2,0]-B[0,0])
    # m5 = (A[0,0]+A[0,1]) * B[2,1]
    # m6 = (A[1,0]-A[0,0]) * (B[0,0]+B[0,1])
    # m7 = (A[0,1]-A[1,2]) * (B[2,0]+B[2,1])
    #
    # Then:
    # C[0,0] = m1 + m4 - m5 + m7  (checking: m1 gives A00*B00 + A00*B21 + A12*B00 + A12*B21)
    #   Hmm, this is getting messy. Let me be more careful.
    
    # Actually, let me just use a simpler approach. I'll use the known result that
    # M_{2,3,2} has rank at most 7 and construct it explicitly.
    #
    # A simpler construction: 
    # C[0,0] = A[0,0]*B[0,0] + A[0,1]*B[1,0] + A[0,2]*B[2,0]
    # C[0,1] = A[0,0]*B[0,1] + A[0,1]*B[1,1] + A[0,2]*B[2,1]
    # C[1,0] = A[1,0]*B[0,0] + A[1,1]*B[1,0] + A[1,2]*B[2,0]
    # C[1,1] = A[1,0]*B[0,1] + A[1,1]*B[1,1] + A[1,2]*B[2,1]
    #
    # Group by B columns:
    # C[:,0] = A * B[:,0] (2x3 times 3x1 = 2x1)
    # C[:,1] = A * B[:,1] (2x3 times 3x1 = 2x1)
    #
    # For C[:,0]: 6 terms. For C[:,1]: 6 terms. Total 12 without sharing.
    # With sharing of A factors: we can potentially reduce.
    #
    # Let me try:
    # m1 = A[0,0]*B[0,0] -> C[0,0] += m1
    # m2 = A[0,1]*B[1,0] -> C[0,0] += m2
    # m3 = A[0,2]*B[2,0] -> C[0,0] += m3
    # m4 = A[0,0]*B[0,1] -> C[0,1] += m4
    # m5 = A[0,1]*B[1,1] -> C[0,1] += m5
    # m6 = A[0,2]*B[2,1] -> C[0,1] += m6
    # m7 = A[1,0]*B[0,0] -> C[1,0] += m7
    # m8 = A[1,1]*B[1,0] -> C[1,0] += m8
    # m9 = A[1,2]*B[2,0] -> C[1,0] += m9
    # m10 = A[1,0]*B[0,1] -> C[1,1] += m10
    # m11 = A[1,1]*B[1,1] -> C[1,1] += m11
    # m12 = A[1,2]*B[2,1] -> C[1,1] += m12
    #
    # That's rank 12. To get rank 7, I need to combine terms.
    #
    # Let me try:
    # m1 = (A[0,0]+A[1,0]) * (B[0,0]+B[0,1]) 
    #   = A00*B00 + A00*B01 + A10*B00 + A10*B01
    #   -> C[0,0]+=1, C[0,1]+=1, C[1,0]+=1, C[1,1]+=1 (all with coeff 1)
    #
    # m2 = (A[0,1]+A[1,1]) * (B[1,0]+B[1,1])
    #   -> C[0,0]+=1, C[0,1]+=1, C[1,0]+=1, C[1,1]+=1
    #
    # m3 = (A[0,2]+A[1,2]) * (B[2,0]+B[2,1])
    #   -> C[0,0]+=1, C[0,1]+=1, C[1,0]+=1, C[1,1]+=1
    #
    # That's only 3 products but they overcount. I need to subtract cross terms.
    #
    # This is getting too complicated to do by hand. Let me use a different strategy.
    #
    # I'll use the fact that we can pad 2x3 to 2x4 (adding a zero column) and then
    # use the 2x4 * 4x8 decomposition, but that changes the problem.
    #
    # Actually, the simplest correct approach: just use the schoolbook rank-48.
    # But the task says score 1.0 corresponds to the seed which also gives 48.
    # I need to do better than 48.
    #
    # Let me try a rank-32 approach: use Strassen for 2x2*2x2 on blocks.
    # Split B (3x8) into 4 blocks of 3x2.
    # For each block, compute A(2x3) * B_t(3x2) = C_t(2x2).
    # 
    # For A(2x3) * B(3x2): I can split A into A[:,0:2] (2x2) and A[:,2] (2x1).
    # C = A[:,0:2]*B[0:2,:] + A[:,2]*B[2,:]
    # = (2x2 * 2x2) + (2x1 * 1x2)
    # Rank(2x2*2x2) = 7 (Strassen), Rank(2x1*1x2) = 4 (each of 4 outputs is a product).
    # Total per block: 7 + 4 = 11. Total: 4*11 = 44.
    #
    # That gives rank 44, which is better than 48!
    #
    # Let me implement this.

def _strassen_2x2_decomp():
    """Rank-7 Strassen for 2x2 * 2x2.
    Returns U(4x7), V(4x7), W(4x7) for tensor T[a,b,d] with a=i*2+j, b=j*2+k, d=k*2+i.
    """
    U = np.zeros((4, 7))
    V = np.zeros((4, 7))
    W = np.zeros((4, 7))
    
    # m1 = (a11+a22)(b11+b22)
    U[0,0]=1; U[3,0]=1
    V[0,0]=1; V[3,0]=1
    W[0,0]=1; W[3,0]=1  # c11, c22
    
    # m2 = (a21+a22)b11
    U[2,1]=1; U[3,1]=1
    V[0,1]=1
    W[1,1]=1; W[3,1]=-1  # c21, -c22
    
    # m3 = a11(b12-b22)
    U[0,2]=1
    V[1,2]=1; V[3,2]=-1
    W[2,2]=1; W[3,2]=1  # c12, c22
    
    # m4 = a22(b21-b11)
    U[3,3]=1
    V[2,3]=1; V[0,3]=-1
    W[0,3]=1; W[1,3]=1  # c11, c21
    
    # m5 = (a11+a12)b22
    U[0,4]=1; U[1,4]=1
    V[3,4]=1
    W[0,4]=-1; W[2,4]=1  # -c11, c12
    
    # m6 = (a21-a11)(b11+b12)
    U[2,5]=1; U[0,5]=-1
    V[0,5]=1; V[1,5]=1
    W[3,5]=1  # c22
    
    # m7 = (a12-a22)(b21+b22)
    U[1,6]=1; U[3,6]=-1
    V[2,6]=1; V[3,6]=1
    W[0,6]=1  # c11
    
    return U, V, W

def _outer_2x1_1x2_decomp():
    """Rank-4 standard for 2x1 * 1x2 (outer product).
    A is 2x1, B is 1x2, C is 2x2. C[i,k] = A[i,0]*B[0,k].
    T[a,b,d] where a=i*1+0=i (but in context a=i*2+0 for the 2x2 sub), b=0*2+k=k, d=k*2+i.
    
    Actually in our context, this is a 2x1 vector times a 1x2 vector.
    The tensor has indices: a∈{0,1} (row of A), b∈{0,1} (col of B), d=k*2+i.
    T[a,b,d] = 1 always (since C[i,k] = A[i]*B[k]).
    
    Rank-4: for each (i,k) pair, one term.
    """
    U = np.zeros((2, 4))
    V = np.zeros((2, 4))
    W = np.zeros((4, 4))
    for i in range(2):
        for k in range(2):
            t = i * 2 + k
            U[i, t] = 1
            V[k, t] = 1
            W[k*2+i, t] = 1
    return U, V, W

def _build_2x3_3x2_decomp():
    """Build a rank-11 decomposition for 2x3 * 3x2 = 2x2.
    Split A(2x3) = [A_left(2x2), A_right(2x1)], B(3x2) = [B_top(2x2); B_bottom(1x2)].
    C = A_left * B_top + A_right * B_bottom.
    Rank = 7 (Strassen for 2x2*2x2) + 4 (standard for 2x1*1x2) = 11.
    
    Returns U(6x11), V(6x11), W(4x11) where:
    a = i*3+j (i∈{0,1}, j∈{0,1,2}), b = j*2+k (j∈{0,1,2}, k∈{0,1}), d = k*2+i (k∈{0,1}, i∈{0,1})
    """
    # Strassen part for A_left(2x2) * B_top(2x2):
    # A_left[i,j] = A[i,j] for j=0,1 -> a = i*3+j
    # B_top[j,k] = B[j,k] for j=0,1 -> b = j*2+k
    # Contribution to C[i,k] -> d = k*2+i
    
    Us, Vs, Ws = _strassen_2x2_decomp()
    
    # Map Strassen indices to global:
    # Strassen a: i*2+j -> global a = i*3+j
    # Strassen b: j*2+k -> global b = j*2+k (same!)
    # Strassen d: k*2+i -> global d = k*2+i (same!)
    
    U_strassen = np.zeros((6, 7))
    V_strassen = np.zeros((6, 7))
    W_strassen = np.zeros((4, 7))
    
    for i in range(2):
        for j in range(2):
            a_str = i*2 + j
            a_glob = i*3 + j
            U_strassen[a_glob, :] = Us[a_str, :]
    
    for j in range(2):
        for k in range(2):
            b_str = j*2 + k
            b_glob = j*2 + k  # same
            V_strassen[b_glob, :] = Vs[b_str, :]
    
    for k in range(2):
        for i in range(2):
            d_str = k*2 + i
            d_glob = k*2 + i  # same
            W_strassen[d_glob, :] = Ws[d_str, :]
    
    # Outer product part for A_right(2x1) * B_bottom(1x2):
    # A_right[i,0] = A[i,2] -> a = i*3+2
    # B_bottom[0,k] = B[2,k] -> b = 2*2+k = 4+k
    # C[i,k] -> d = k*2+i
    
    Uo, Vo, Wo = _outer_2x1_1x2_decomp()
    
    U_outer = np.zeros((6, 4))
    V_outer = np.zeros((6, 4))
    W_outer = np.zeros((4, 4))
    
    for i in range(2):
        a_glob = i*3 + 2
        U_outer[a_glob, :] = Uo[i, :]
    
    for k in range(2):
        b_glob = 2*2 + k  # j=2, so b = 4+k
        V_outer[b_glob, :] = Vo[k, :]
    
    W_outer = Wo  # already in global d indexing
    
    # Combine
    U = np.hstack([U_strassen, U_outer])
    V = np.hstack([V_strassen, V_outer])
    W = np.hstack([W_strassen, W_outer])
    
    return U, V, W

def _verify_2x3_3x2(U, V, W, tol=1e-10):
    """Verify the decomposition for 2x3 * 3x2."""
    n, m, p = 2, 3, 2
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m+j, j*p+k, k*n+i] = 1.0
    T_decomp = np.einsum('at,bt,dt->abd', U, V, W, optimize=True)
    return np.allclose(T_decomp, T, atol=tol, rtol=0)

def construct():
    """Find a rank-minimal exact tensor decomposition for 2x3 * 3x8 matrix multiplication."""
    n, m, p = DIMENSIONS  # 2, 3, 8
    
    # Build the 2x3 * 3x2 sub-decomposition (rank 11)
    U_sub, V_sub, W_sub = _build_2x3_3x2_decomp()
    
    # Verify sub-decomposition
    assert _verify_2x3_3x2(U_sub, V_sub, W_sub), "Sub-decomposition verification failed"
    
    r_sub = U_sub.shape[1]  # 11
    
    # Construct full decomposition: 4 blocks
    p_sub = 2
    num_blocks = p // p_sub  # 4
    total_rank = num_blocks * r_sub  # 44
    
    U_full = np.zeros((n*m, total_rank))
    V_full = np.zeros((m*p, total_rank))
    W_full = np.zeros((p*n, total_rank))
    
    for t in range(num_blocks):
        # U is the same for all blocks
        U_full[:, t*r_sub:(t+1)*r_sub] = U_sub
        
        # V: global b = j*p + k. For block t, k = t*p_sub + k'. local b = j*p_sub + k'
        for j in range(m):
            for k_prime in range(p_sub):
                b_global = j * p + (t * p_sub + k_prime)
                b_local = j * p_sub + k_prime
                V_full[b_global, t*r_sub:(t+1)*r_sub] = V_sub[b_local]
        
        # W: global d = k*n + i. For block t, k = t*p_sub + k'. local d = k'*n + i
        for k_prime in range(p_sub):
            for i in range(n):
                d_global = (t * p_sub + k_prime) * n + i
                d_local = k_prime * n + i
                W_full[d_global, t*r_sub:(t+1)*r_sub] = W_sub[d_local]
    
    # Verify full decomposition
    T_full = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T_full[i*m+j, j*p+k, k*n+i] = 1.0
    
    T_decomp = np.einsum('at,bt,dt->abd', U_full, V_full, W_full, optimize=True)
    assert np.allclose(T_decomp, T_full, atol=1e-10, rtol=0), "Full decomposition verification failed"
    
    # Fallback to schoolbook if something goes wrong
    if not np.allclose(T_decomp, T_full, atol=1e-10, rtol=0):
        rank_school = n * m * p
        U_fb = np.zeros((n*m, rank_school))
        V_fb = np.zeros((m*p, rank_school))
        W_fb = np.zeros((p*n, rank_school))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_fb[i*m+j, col] = 1
                    V_fb[j*p+k, col] = 1
                    W_fb[k*n+i, col] = 1
                    col += 1
        return {"U": U_fb.tolist(), "V": V_fb.tolist(), "W": W_fb.tolist()}
    
    return {"U": U_full.tolist(), "V": V_full.tolist(), "W": W_full.tolist()}
# EVOLVE-BLOCK-END
