"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 4]

# EVOLVE-BLOCK-START
import time
import random
import numpy as np
from itertools import product

# 2x2 Strassen rank-7 decomposition with trace convention W[k*n+i]
# For n=m=p=2, U is 4x7 (row i*m+j), V is 4x7 (row j*p+k), W is 4x7 (row k*n+i)
# Strassen's 7 products:
# M1 = (a11+a22)(b11+b22)
# M2 = (a21+a22)b11
# M3 = a11(b21-b22)
# M4 = a22(b12-b11)
# M5 = (a11+a12)b22
# M6 = (a21-a11)(b11+b12)
# M7 = (a12-a22)(b21+b22)
# C11 = M1 + M4 - M5 + M7
# C12 = M3 + M5
# C21 = M2 + M4
# C22 = M1 - M2 + M3 + M6

def _strassen_2x2():
    """Return U, V, W as lists of lists for 2x2x2 Strassen rank-7 decomposition."""
    # U: row = i*2+j, col = r (0..6)
    # V: row = j*2+k, col = r (0..6)
    # W: row = k*2+i, col = r (0..6)
    
    U = [[0.0]*7 for _ in range(4)]
    V = [[0.0]*7 for _ in range(4)]
    W = [[0.0]*7 for _ in range(4)]
    
    # M1 = (a11+a22)(b11+b22)
    # a11: i=0,j=0 -> row 0; a22: i=1,j=1 -> row 3
    U[0][0] = 1.0
    U[3][0] = 1.0
    # b11: j=0,k=0 -> row 0; b22: j=1,k=1 -> row 3
    V[0][0] = 1.0
    V[3][0] = 1.0
    # C11: i=0,k=0 -> W row 0; C22: i=1,k=1 -> W row 3
    W[0][0] = 1.0
    W[3][0] = 1.0
    
    # M2 = (a21+a22)b11
    # a21: i=1,j=0 -> row 2; a22: i=1,j=1 -> row 3
    U[2][1] = 1.0
    U[3][1] = 1.0
    # b11: j=0,k=0 -> row 0
    V[0][1] = 1.0
    # C21: i=1,k=0 -> W row 1
    W[1][1] = 1.0
    
    # M3 = a11(b21-b22)
    # a11: i=0,j=0 -> row 0
    U[0][2] = 1.0
    # b21: j=1,k=0 -> row 2; b22: j=1,k=1 -> row 3
    V[2][2] = 1.0
    V[3][2] = -1.0
    # C12: i=0,k=1 -> W row 2
    W[2][2] = 1.0
    
    # M4 = a22(b12-b11)
    # a22: i=1,j=1 -> row 3
    U[3][3] = 1.0
    # b12: j=0,k=1 -> row 1; b11: j=0,k=0 -> row 0
    V[1][3] = 1.0
    V[0][3] = -1.0
    # C11: i=0,k=0 -> W row 0; C21: i=1,k=0 -> W row 1
    W[0][3] = 1.0
    W[1][3] = 1.0
    
    # M5 = (a11+a12)b22
    # a11: i=0,j=0 -> row 0; a12: i=0,j=1 -> row 1
    U[0][4] = 1.0
    U[1][4] = 1.0
    # b22: j=1,k=1 -> row 3
    V[3][4] = 1.0
    # C11: i=0,k=0 -> W row 0; C12: i=0,k=1 -> W row 2
    W[0][4] = -1.0
    W[2][4] = 1.0
    
    # M6 = (a21-a11)(b11+b12)
    # a21: i=1,j=0 -> row 2; a11: i=0,j=0 -> row 0
    U[2][5] = 1.0
    U[0][5] = -1.0
    # b11: j=0,k=0 -> row 0; b12: j=0,k=1 -> row 1
    V[0][5] = 1.0
    V[1][5] = 1.0
    # C22: i=1,k=1 -> W row 3
    W[3][5] = 1.0
    
    # M7 = (a12-a22)(b21+b22)
    # a12: i=0,j=1 -> row 1; a22: i=1,j=1 -> row 3
    U[1][6] = 1.0
    U[3][6] = -1.0
    # b21: j=1,k=0 -> row 2; b22: j=1,k=1 -> row 3
    V[2][6] = 1.0
    V[3][6] = 1.0
    # C11: i=0,k=0 -> W row 0
    W[0][6] = 1.0
    
    return U, V, W


def _kronecker_product_4x4():
    """
    Build 4x4x4 tensor decomposition from 2x2x2 Strassen via Kronecker product.
    Rank = 7*7 = 49.
    
    For block decomposition:
    A = [[A11, A12], [A21, A22]] where each block is 2x2
    B = [[B11, B12], [B21, B22]] where each block is 2x2
    C = A*B where each block Cij is 2x2
    
    C11 = A11*B11 + A12*B21
    C12 = A11*B12 + A12*B22
    C21 = A21*B11 + A22*B21
    C22 = A21*B12 + A22*B22
    
    Each block multiplication (2x2 * 2x2) uses rank-7 Strassen.
    The Kronecker product structure gives rank 7*7 = 49.
    
    Indexing for 4x4:
    U row: i*4+j where i,j in 0..3
    V row: j*4+k where j,k in 0..3  
    W row: k*4+i where k,i in 0..3
    
    Block structure: i = 2*ib + ib2, j = 2*jb + jb2, k = 2*kb + kb2
    where ib,jb,kb are block indices (0 or 1) and ib2,jb2,kb2 are within-block (0 or 1).
    
    The 4x4 multiplication tensor T[i,j,k] = sum over r of U[i*4+j, r] * V[j*4+k, r] * W[k*4+i, r]
    
    Using Kronecker product of the 2x2 decomposition:
    For block-level Strassen, we have 7 products M_a (a=0..6).
    For each block product, we need to multiply two 2x2 matrices, which itself uses 7 Strassen products.
    
    So total rank = 7*7 = 49, indexed by (a, b) where a is block-level Strassen index and b is within-block Strassen index.
    
    Let's define:
    Block indices: ib, jb, kb in {0,1}
    Within-block indices: ib2, jb2, kb2 in {0,1}
    
    i = 2*ib + ib2
    j = 2*jb + jb2
    k = 2*kb + kb2
    
    The 4x4 multiplication:
    C[i,k] = sum_j A[i,j] * B[j,k]
    
    In block form:
    C[ib,ib2; kb,kb2] = sum_{jb} sum_{jb2} A[ib,ib2; jb,jb2] * B[jb,jb2; kb,kb2]
    
    = sum_{jb} (A_block[ib,jb] * B_block[jb,kb])[ib2, kb2]
    
    Each A_block[ib,jb] * B_block[jb,kb] is a 2x2 * 2x2 multiplication.
    
    Using Strassen for the block level:
    The 7 block-level products combine blocks of A and B.
    Then each combined block is multiplied using within-block Strassen.
    
    This is getting complex. Let me use a simpler approach: direct Kronecker product of tensors.
    
    Actually, the cleanest way: the 4x4x4 tensor is the Kronecker product of two 2x2x2 tensors.
    T_4[i,j,k] = T_2[ib, jb, kb] * T_2[ib2, jb2, kb2]
    where i = 2*ib+ib2, etc.
    
    If T_2 has rank-7 decomposition T_2[a,b,c] = sum_r U2[a,b,r] V2[b,c,r] W2[c,a,r],
    then T_4 has rank 49 decomposition with:
    U4[(ib,jb), (r,s)] = U2[ib,jb,r] * U2[jb,kb,s]... no, that's not right.
    
    Let me think again. The tensor T_4 is defined as:
    T_4[i,j,k] = 1 if this corresponds to a term in matrix multiplication.
    
    For 4x4: C[i,k] = sum_j A[i,j] B[j,k]
    So T_4[i,j,k] = 1 (the tensor that matrix multiplication computes).
    
    The Kronecker product structure:
    T_4[i,j,k] where i=2*ib+ib2, j=2*jb+jb2, k=2*kb+kb2
    
    C[2*ib+ib2, 2*kb+kb2] = sum_{j=0}^{3} A[2*ib+ib2, j] B[j, 2*kb+kb2]
    = sum_{jb=0}^{1} sum_{jb2=0}^{1} A[2*ib+ib2, 2*jb+jb2] B[2*jb+jb2, 2*kb+kb2]
    
    This is not simply a Kronecker product of T_2 tensors. The block multiplication is:
    C_block[ib,kb] = sum_jb A_block[ib,jb] * B_block[jb,kb]
    
    And each block multiplication (2x2 * 2x2) is a separate 2x2 multiplication.
    
    So the 4x4 multiplication decomposes into 4 block multiplications (for C11, C12, C21, C22),
    each of which is a 2x2 * 2x2 multiplication, and each of those uses 7 Strassen products.
    
    But the block multiplications share the same A and B blocks!
    
    Specifically:
    C11 = A11*B11 + A12*B21
    C12 = A11*B12 + A12*B22
    C21 = A21*B11 + A22*B21
    C22 = A21*B12 + A22*B22
    
    Each of these sums involves 2 products of 2x2 matrices, and each 2x2 product uses 7 Strassen products.
    Total: 8 * 7 = 56? No, that's not right either.
    
    Actually, let me just directly construct the Kronecker product properly.
    
    The matrix multiplication tensor for n*m*p where n=m=p=4:
    T[i,j,k] = 1 for all i,j,k (it's the standard matrix multiplication tensor)
    
    For n=m=p=2: T2[i,j,k] = 1 for all i,j,k in {0,1}
    
    The 4x4 tensor is NOT the Kronecker product of two 2x2 tensors in the standard sense.
    
    Let me just build it directly using the known rank-49 construction from literature, or use a randomized search.
    """
    pass


def construct():
    """
    Construct a rank-49 tensor decomposition for 4x4 matrix multiplication
    using the Kronecker product structure of Strassen's algorithm.
    """
    n, m, p = DIMENSIONS
    rank = 49  # 7 * 7
    
    # We'll build U, V, W as numpy arrays then convert to lists
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    # Get the 2x2 Strassen decomposition
    U2, V2, W2 = _strassen_2x2()
    U2 = np.array(U2)  # shape (4, 7)
    V2 = np.array(V2)  # shape (4, 7)
    W2 = np.array(W2)  # shape (4, 7)
    
    # The 4x4 matrix multiplication can be viewed as block multiplication.
    # Using the Kronecker product approach:
    # Index i in 4x4: i = 2*ib + ib2
    # Index j in 4x4: j = 2*jb + jb2
    # Index k in 4x4: k = 2*kb + kb2
    #
    # The rank-49 decomposition uses pairs (r, s) where r, s are in 0..6 (Strassen indices).
    # The tensor decomposition is:
    # T4[i,j,k] = sum_{r,s} U4[i*4+j, r*7+s] * V4[j*4+k, r*7+s] * W4[k*4+i, r*7+s]
    #
    # By the Kronecker product structure:
    # U4[(ib,jb), (r,s)] = U2[ib*2+jb, r] * U2[jb*2+?, s]... 
    
    # Actually, let me think about this differently.
    # The 4x4 multiplication tensor T4[i,j,k] for i,j,k in 0..3.
    # We want to express this as a sum of rank-1 tensors.
    # 
    # Using the block structure:
    # i = 2*ib + ib2, j = 2*jb + jb2, k = 2*kb + kb2
    #
    # C[i,k] = sum_j A[i,j] B[j,k]
    # C[2*ib+ib2, 2*kb+kb2] = sum_{jb,jb2} A[2*ib+ib2, 2*jb+jb2] B[2*jb+jb2, 2*kb+kb2]
    #
    # = sum_{jb} (sum_{jb2} A[2*ib+ib2, 2*jb+jb2] B[2*jb+jb2, 2*kb+kb2])
    #
    # The inner sum is (A_block[ib,jb] * B_block[jb,kb])[ib2, kb2]
    #
    # Each block multiplication A_block[ib,jb] * B_block[jb,kb] is a 2x2 * 2x2 product.
    # Using Strassen for this:
    # (A_block * B_block)[ib2, kb2] = sum_{r=0}^{6} U2[ib2,jb2,r] * V2[jb2,kb2,r] * W2[kb2,ib2,r]
    #
    # So C[2*ib+ib2, 2*kb+kb2] = sum_{jb} sum_{r} U2[ib2,jb2,r] V2[jb2,kb2,r] W2[kb2,ib2,r]
    #   * (contribution from A_block[ib,jb] and B_block[jb,kb])
    #
    # Wait, this isn't quite right. The Strassen decomposition is for the 2x2 multiplication tensor,
    # where the "tensor" T2[a,b,c] represents the (a,c)-th entry of the product of two 2x2 matrices,
    # with the summation over b.
    #
    # Let me be more explicit. For 2x2:
    # C2[i2,k2] = sum_{j2=0}^{1} A2[i2,j2] * B2[j2,k2]
    # Strassen: C2[i2,k2] = sum_{r=0}^{6} (linear_combination_of_A2_entries for r) * (linear_combination_of_B2_entries for r) * coeff[i2,k2,r]
    #
    # In the U,V,W representation:
    # T2[i2,j2,k2] = 1 (the 2x2 multiplication tensor)
    # T2[i2,j2,k2] = sum_r U2[i2*2+j2, r] * V2[j2*2+k2, r] * W2[k2*2+i2, r]
    #
    # For 4x4, we have:
    # T4[i,j,k] = 1
    # i = 2*ib+ib2, j = 2*jb+jb2, k = 2*kb+kb2
    #
    # T4[i,j,k] should equal sum over r,s of U4[i*4+j, r*7+s] * V4[j*4+k, r*7+s] * W4[k*4+i, r*7+s]
    #
    # By the Kronecker product structure of tensor decompositions:
    # If T4 = T2 ⊗ T2 (Kronecker product of tensors), then:
    # U4[(ib,jb), (r,s)] = U2[ib*2+jb, r] * U2[jb*2+?, s]
    #
    # This is getting confusing. Let me just use a known rank-49 construction.
    # 
    # Actually, the simplest correct approach: the 4x4 multiplication tensor 
    # T4[i,j,k] where i,j,k range over 0..3 is exactly the Kronecker product of T2 with itself
    # in the sense that:
    # T4[2*ib+ib2, 2*jb+jb2, 2*kb+kb2] = T2[ib,jb,kb] * T2[ib2,jb2,kb2]
    #
    # But T2[ib,jb,kb] = 1 for all ib,jb,kb (it's the 2x2 multiplication tensor, which is all ones).
    # And T2[ib2,jb2,kb2] = 1 for all ib2,jb2,kb2.
    # So T4 = 1 for all entries, which is correct.
    #
    # The Kronecker product of tensor decompositions:
    # If T2 has rank-7 decomposition with (U2, V2, W2), then T2⊗T2 has rank-49 decomposition with:
    # U4[(a1,a2), (r1,r2)] = U2[a1, r1] * U2[a2, r2]
    # V4[(b1,b2), (r1,r2)] = V2[b1, r1] * V2[b2, r2]
    # W4[(c1,c2), (r1,r2)] = W2[c1, r1] * W2[c2, r2]
    #
    # Where a1,a2 are indices in 0..3 (for the 4x4 case), etc.
    #
    # But wait, the indices don't match up this way. Let me reconsider.
    #
    # For the Kronecker product of two tensors T and S:
    # (T⊗S)[(i1,i2),(j1,j2),(k1,k2)] = T[i1,j1,k1] * S[i2,j2,k2]
    #
    # If T = T2 (2x2 mult tensor) and S = T2, then:
    # (T2⊗T2)[(i1,i2),(j1,j2),(k1,k2)] = T2[i1,j1,k1] * T2[i2,j2,k2] = 1
    #
    # This gives us a 4x4x4 tensor that is all ones, which is exactly T4.
    #
    # The rank-49 decomposition:
    # (T2⊗T2)[(i1,i2),(j1,j2),(k1,k2)] = sum_{r1,r2} (U2[i1,j1,r1]*U2[i2,j2,r2]) * (V2[j1,k1,r1]*V2[j2,k2,r2]) * (W2[k1,i1,r1]*W2[k2,i2,r2])
    #
    # So:
    # U4[(i1,i2), (r1,r2)] = U2[i1,j1,r1] * U2[i2,j2,r2]... 
    #
    # No, that's not right. The indices are coupled. Let me re-derive.
    #
    # T4[i,j,k] where i=2*i1+i2, j=2*j1+j2, k=2*k1+k2
    # T4[i,j,k] = sum_{r1=0}^{6} sum_{r2=0}^{6} U2[i1,j1,r1] * U2[i2,j2,r2] * V2[j1,k1,r1] * V2[j2,k2,r2] * W2[k1,i1,r1] * W2[k2,i2,r2]
    #
    # = sum_{r1,r2} [U2[i1,j1,r1]*V2[j1,k1,r1]*W2[k1,i1,r1]] * [U2[i2,j2,r2]*V2[j2,k2,r2]*W2[k2,i2,r2]]
    #
    # This is NOT a valid rank-1 decomposition because the sum is over r1 AND r2, and the terms are not separable in the U,V,W form.
    #
    # The correct Kronecker product of tensor decompositions:
    # If T[i,j,k] = sum_r U[i,j,r] V[j,k,r] W[k,i,r], then
    # (T⊗T)[(i1,i2),(j1,j2),(k1,k2)] = sum_{r1,r2} U[i1,j1,r1]U[i2,j2,r2] * V[j1,k1,r1]V[j2,k2,r2] * W[k1,i1,r1]W[k2,i2,r2]
    #
    # This IS a valid rank-(rank^2) decomposition with:
    # U_new[(i1,i2),(r1,r2)] = U[i1,j1,r1] * U[i2,j2,r2]
    # V_new[(j1,j2),(r1,r2)] = V[j1,k1,r1] * V[j2,k2,r2]
    # W_new[(k1,k2),(r1,r2)] = W[k1,i1,r1] * W[k2,i2,r2]
    #
    # But this doesn't work because U_new depends on j1,j2 which are not part of its row index!
    #
    # I think the issue is that the Kronecker product of tensor decompositions doesn't work this simply.
    #
    # Let me just use a different approach: I'll construct the rank-49 decomposition using the fact that
    # 4x4 multiplication can be done with 49 multiplications using the Kronecker product of Strassen's 2x2 algorithm.
    #
    # Here's a cleaner way to think about it:
    # The 4x4 matrix multiplication is equivalent to multiplying two 2x2 matrices of 2x2 blocks.
    # Each block multiplication (2x2 * 2x2) uses 7 Strassen multiplications.
    # There are 4 block multiplications (for C11, C12, C21, C22), but they share terms.
    #
    # Actually, the standard result is that rank(4x4 mult) <= 7^2 = 49, achieved by the Kronecker product construction.
    #
    # Let me just directly build it. The key insight is:
    # For the 4x4 case, we index by (block, within) pairs.
    # i = 2*ib + ib2, j = 2*jb + jb2, k = 2*kb + kb2
    #
    # The decomposition uses 49 terms indexed by (r1, r2) where r1 is for the "block-level" Strassen and r2 is for the "within-block" Strassen.
    #
    # But actually, I realize the Kronecker product works if we're careful about the tensor structure.
    #
    # Let me just verify: is T4 = T2 ⊗ T2 as tensors?
    # T2 is the 2x2 matrix multiplication tensor: T2[i,j,k] = 1 for i,j,k in {0,1}
    # T4 is the 4x4 matrix multiplication tensor: T4[i,j,k] = 1 for i,j,k in {0,1,2,3}
    #
    # T2 ⊗ T2 [(i1,i2),(j1,j2),(k1,k2)] = T2[i1,j1,k1] * T2[i2,j2,k2] = 1*1 = 1
    # T4[2*i1+i2, 2*j1+j2, 2*k1+k2] = 1
    #
    # Yes! T4 = T2 ⊗ T2.
    #
    # Now, for the rank decomposition of a Kronecker product:
    # If T has rank-r decomposition T[i,j,k] = sum_s U[i,j,s]V[j,k,s]W[k,i,s],
    # and S has rank-s decomposition S[i',j',k'] = sum_t U'[i',j',t]V'[j',k',t]W'[k',i',t],
    # then (T⊗S)[(i,i'),(j,j'),(k,k')] = sum_{s,t} U[i,j,s]U'[i',j',t] * V[j,k,s]V'[j',k',t] * W[k,i,s]W'[k',i',t]
    #
    # This is a valid rank-(r*s) decomposition with:
    # U_new[(i,i'),(s,t)] = U[i,j,s] * U'[i',j',t]
    #
    # But again, U_new depends on j which is not in its row index. This is the fundamental issue.
    #
    # The resolution: the Kronecker product of tensor decompositions does NOT generally give a valid tensor decomposition in the U,V,W form because the row indices of U, V, W in the product tensors don't align.
    #
    # So I need a different approach. Let me just use a randomized search or a known explicit construction.
    #
    # Actually, let me try yet another approach. I'll use the fact that the 4x4 multiplication tensor can be decomposed using 49 terms by exploiting the block structure more carefully.
    #
    # Here's the key: I'll define the 49 terms as follows.
    # For each pair of Strassen indices (r1, r2) in {0,...,6} x {0,...,6}:
    #   The term contributes to specific entries of U, V, W.
    #
    # Let me define:
    # i = 2*ib + ib2
    # j = 2*jb + jb2
    # k = 2*kb + kb2
    #
    # U[2*ib+ib2, 2*jb+jb2, (r1,r2)] = U2[ib, jb, r1] * U2[ib2, jb2, r2]
    # V[2*jb+jb2, 2*kb+kb2, (r1,r2)] = V2[jb, kb, r1] * V2[jb2, kb2, r2]
    # W[2*kb+kb2, 2*ib+ib2, (r1,r2)] = W2[kb, ib, r1] * W2[kb2, ib2, r2]
    #
    # Let's verify: sum_{r1,r2} U[2*ib+ib2, 2*jb+jb2, (r1,r2)] * V[2*jb+jb2, 2*kb+kb2, (r1,r2)] * W[2*kb+kb2, 2*ib+ib2, (r1,r2)]
    # = sum_{r1,r2} U2[ib,jb,r1]*U2[ib2,jb2,r2] * V2[jb,kb,r1]*V2[jb2,kb2,r2] * W2[kb,ib,r1]*W2[kb2,ib2,r2]
    # = (sum_{r1} U2[ib,jb,r1]*V2[jb,kb,r1]*W2[kb,ib,r1]) * (sum_{r2} U2[ib2,jb2,r2]*V2[jb2,kb2,r2]*W2[kb2,ib2,r2])
    # = T2[ib,jb,kb] * T2[ib2,jb2,kb2]
    # = 1 * 1 = 1
    #
    # YES! This works! The sums over r1 and r2 are independent, so they factorize.
    #
    # So the construction is:
    # U4[i*4+j, r1*7+r2] = U2[(i//2)*2+(j//2), r1] * U2[(i%2)*2+(j%2), r2]
    # V4[j*4+k, r1*7+r2] = V2[(j//2)*2+(k//2), r1] * V2[(j%2)*2+(k%2), r2]
    # W4[k*4+i, r1*7+r2] = W2[(k//2)*2+(i//2), r1] * W2[(k%2)*2+(i%2), r2]
    
    for r1 in range(7):
        for r2 in range(7):
            r = r1 * 7 + r2
            for i in range(4):
                for j in range(4):
                    ib = i // 2
                    ib2 = i % 2
                    jb = j // 2
                    jb2 = j % 2
                    U[i * 4 + j, r] = U2[ib * 2 + jb, r1] * U2[ib2 * 2 + jb2, r2]
            
            for j in range(4):
                for k in range(4):
                    jb = j // 2
                    jb2 = j % 2
                    kb = k // 2
                    kb2 = k % 2
                    V[j * 4 + k, r] = V2[jb * 2 + kb, r1] * V2[jb2 * 2 + kb2, r2]
            
            for k in range(4):
                for i in range(4):
                    kb = k // 2
                    kb2 = k % 2
                    ib = i // 2
                    ib2 = i % 2
                    W[k * 4 + i, r] = W2[kb * 2 + ib, r1] * W2[kb2 * 2 + ib2, r2]
    
    # Verify the decomposition
    # T4[i,j,k] should equal sum_r U[i*4+j,r] * V[j*4+k,r] * W[k*4+i,r]
    for i in range(4):
        for j in range(4):
            for k in range(4):
                total = 0.0
                for r in range(49):
                    total += U[i*4+j, r] * V[j*4+k, r] * W[k*4+i, r]
                if abs(total - 1.0) > 1e-10:
                    # If verification fails, fall back to schoolbook
                    rank = 64
                    U = np.zeros((16, 64))
                    V = np.zeros((16, 64))
                    W = np.zeros((16, 64))
                    col = 0
                    for ii in range(4):
                        for jj in range(4):
                            for kk in range(4):
                                U[ii*4+jj, col] = 1.0
                                V[jj*4+kk, col] = 1.0
                                W[kk*4+ii, col] = 1.0
                                col += 1
                    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
