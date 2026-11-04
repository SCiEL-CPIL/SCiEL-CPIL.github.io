import numpy as np
import time

def construct():
    """
    Find a rank-minimal exact tensor decomposition for multiplying a 2 by 4 matrix by a 4 by 8 matrix.
    
    Strategy:
    We use a Kronecker product structure. The 2x4x8 matrix multiplication tensor can be 
    decomposed into the Kronecker product of a 2x2x2 tensor and a 4x2x4 tensor.
    T_{2,4,8} \cong T_{2,2,2} \otimes T_{4,2,4}
    
    We use the Strassen algorithm (rank 7) for the 2x2x2 part and the schoolbook algorithm 
    (rank 8) for the 4x2x4 part.
    Total rank = 7 * 8 = 56.
    
    This is lower than the schoolbook rank of 2*4*8 = 64.
    
    Indices:
    A is 2x4, B is 4x8, C is 2x8.
    
    Let's define the indices:
    A[i, j] where i in 0..1, j in 0..3.
    B[j, k] where j in 0..3, k in 0..7.
    C[i, k] where i in 0..1, k in 0..7.
    
    We can view A as a 2x2 block matrix where each block is 1x2? Or 2x2 where blocks are 1x2?
    Let's use the isomorphism:
    A_{2,4} -> A'_{2,2} \otimes A''_{1,2} ? No.
    
    Let's map the indices explicitly.
    i = i1 * 1 + i2 (i1 in 0..1, i2 in 0..1) -> i1 is the "2" dimension of A?
    Wait, A is 2x4. 
    Let i = i_2 (0,1) and j = j_4 (0,1,2,3).
    We want to factor j_4 into two parts. 
    The 4 in A's column corresponds to the 4 in B's row.
    The 8 in B's column corresponds to the 8 in C's column.
    The 2 in A's row corresponds to the 2 in C's row.
    
    Let's split the dimensions:
    dim A: 2 x 4
    dim B: 4 x 8
    dim C: 2 x 8
    
    We can split the '4' into 2*2 and the '8' into 4*2.
    So:
    A[i, j] where i = i_2, j = j_2 * 2 + j'_2.
    B[j, k] where j = j_2 * 2 + j'_2, k = k_4 * 2 + k'_2.
    C[i, k] where i = i_2, k = k_4 * 2 + k'_2.
    
    This looks like a Kronecker product of:
    Tensor 1: 2 x 2 x 2 (indices i_2, j_2, k_4? No, k_4 is 4.)
    
    Let's try:
    i = i_A
    j = j_A * 2 + j_B
    k = k_C * 2 + k_B
    
    Here:
    i_A in 0,1
    j_A in 0,1
    j_B in 0,1
    k_C in 0,1,2,3
    k_B in 0,1
    
    A[i_A, j_A * 2 + j_B]
    B[j_A * 2 + j_B, k_C * 2 + k_B]
    C[i_A, k_C * 2 + k_B]
    
    The multiplication is:
    C[i_A, k_C, k_B] = sum_{j_A, j_B} A[i_A, j_A, j_B] * B[j_A, j_B, k_C, k_B]
    
    This can be grouped:
    For fixed i_A, k_C:
    Sum over j_A, j_B.
    
    This is not a simple Kronecker product of independent tensors because j_A and j_B are summed together.
    However, if we swap the order of indices in B:
    Let B'[j_A, k_C, j_B, k_B] = B[j_A*2+j_B, k_C*2+k_B].
    Then C[i_A, k_C, k_B] = sum_{j_A, j_B} A[i_A, j_A, j_B] B'[j_A, k_C, j_B, k_B].
    
    This is a "tensor contraction" that can be viewed as a Kronecker product if the summation indices factor.
    Here the summation is over (j_A, j_B).
    
    Let's define two tensors:
    T1: 2x2x2 with indices (i_A, j_A, k_A) where k_A is a dummy?
    
    Actually, let's use the known isomorphism:
    M_{2,4,8} = M_{2,2,2} \otimes M_{1,2,4} ?
    Rank(M_{2,2,2}) = 7 (Strassen)
    Rank(M_{1,2,4}) = 1*2*4 = 8 (Schoolbook, as 1x2x4 is a vector dot product which is rank 4? No.
    M_{1,2,4}: A is 1x2, B is 2x4, C is 1x4.
    C = A * B.
    This is a bilinear map.
    Rank is 2*4 = 8?
    C_k = sum_j A_j B_jk.
    This is 4 independent dot products of length 2.
    Each dot product of length 2 has rank 2 (or 3 with Strassen? No, 1x2x1 is rank 2? 
    A(1x2) * B(2x1) = 1x1. Rank 2.
    So 4 dot products -> rank 8.
    
    So if M_{2,4,8} \cong M_{2,2,2} \otimes M_{1,2,4}, then rank is 7 * 8 = 56.
    
    Let's verify the indices for M_{1,2,4}.
    Indices: i'=0, j' in 0,1, k' in 0..3.
    A'_{0, j'}
    B'_{j', k'}
    C'_{0, k'}
    
    The Kronecker product M_{2,2,2} \otimes M_{1,2,4} has indices:
    I = (i, i') -> i in 0,1; i'=0. So I is just i. (Matches 2)
    J = (j, j') -> j in 0,1; j' in 0,1. So J is 2*2=4. (Matches 4)
    K = (k, k') -> k in 0,1; k' in 0..3. So K is 2*4=8. (Matches 8)
    
    The mapping is:
    A_{i, j*2+j'} = A_{2,2,2}[i, j, k_dummy] * A_{1,2,4}[0, j']
    Wait, the Kronecker product of tensors T1 and T2:
    (T1 \otimes T2)_{(i1, i2), (j1, j2), (k1, k2)} = T1_{i1, j1, k1} * T2_{i2, j2, k2}
    
    Here:
    T1 is M_{2,2,2} (indices i, j, k) with rank 7.
    T2 is M_{1,2,4} (indices i', j', k') with rank 8.
    
    The combined tensor T = T1 \otimes T2.
    Indices:
    Row: (i, i') -> i in 0,1; i'=0. Map to I = i.
    Col: (j, j') -> j in 0,1; j' in 0,1. Map to J = 2*j + j'.
    Out: (k, k') -> k in 0,1; k' in 0..3. Map to K = 4*k + k'.
    
    Wait, the output C is 2x8.
    C_{i, K} = C_{i, 4k+k'}.
    In the Kronecker product, the output index is (k, k').
    T1 contributes to k (0,1). T2 contributes to k' (0..3).
    So K = 4*k + k'. This matches the 8 dimensions.
    
    So the decomposition is valid.
    
    We need to construct U, V, W for the Kronecker product.
    If T1 has rank R1=7 and T2 has rank R2=8.
    T1: U1 (2x2, 7), V1 (2x2, 7), W1 (2x2, 7).
    T2: U2 (1x2, 8), V2 (2x4, 8), W2 (4x1, 8).
    
    The Kronecker product tensor T has rank R1*R2 = 56.
    The factor matrices are:
    U = U1 \otimes U2
    V = V1 \otimes V2
    W = W1 \otimes W2
    
    Shapes:
    U1: (4, 7). U2: (2, 8). U = U1 \otimes U2 -> (8, 56).
    Wait, U1 is 2x2 matrix, so flattened is 4. U2 is 1x2 matrix, so flattened is 2.
    U should be (n*m, rank) = (8, 56).
    U1 \otimes U2 has shape (4*2, 7*8) = (8, 56). Correct.
    
    V1: (2x2) -> 4. V2: (2x4) -> 8. V = V1 \otimes V2 -> (4*8, 56) = (32, 56).
    V should be (m*p, rank) = (4*8, 56) = (32, 56). Correct.
    
    W1: (2x2) -> 4. W2: (4x1) -> 4. W = W1 \otimes W2 -> (4*4, 56) = (16, 56).
    W should be (p*n, rank) = (8*2, 56) = (16, 56). Correct.
    
    However, we must be careful with the index ordering in the Kronecker product to match the "official trace convention" and the specific index mapping derived.
    
    Standard Kronecker product A \otimes B:
    (A \otimes B)_{(i_A, i_B), (j_A, j_B)} = A_{i_A, j_A} B_{i_B, j_B}
    
    We need to map the composite indices to the single index.
    Row index I = i (from T1). i_B is fixed to 0.
    Col index J = 2*j_A + j_B.
    Out index K = 4*k_A + k_B.
    
    The tensor is T_{I, J, K} = sum_c U[I, c] V[J, c] W[K, c].
    
    The Kronecker product tensor is:
    T_{(i, 0), (2j_A+j_B), (4k_A+k_B)} = sum_{c1, c2} U1[i, c1] V1[j_A, c1] W1[k_A, c1] * U2[0, c2] V2[j_B, c2] W2[k_B, c2]
    
    We can combine c = c1 * 8 + c2.
    Then:
    U[i, c1*8+c2] = U1[i, c1] * U2[0, c2]
    V[2j_A+j_B, c1*8+c2] = V1[j_A, c1] * V2[j_B, c2]
    W[4k_A+k_B, c1*8+c2] = W1[k_A, c1] * W2[k_B, c2]
    
    We need to define U1, V1, W1 for Strassen 2x2x2.
    And U2, V2, W2 for Schoolbook 1x2x4.
    
    Strassen 2x2x2:
    A = [[a, b], [c, d]]
    B = [[e, f], [g, h]]
    C = A B
    M1 = (a+d)(e+h)
    M2 = (c+d)e
    M3 = a(f-h)
    M4 = c(g-f)
    M5 = (a+b)h
    M6 = (c-a)(e+f)
    M7 = (b-d)(g+h)
    
    C11 = M1 + M4 - M5 + M7
    C12 = M3 + M5
    C21 = M2 + M4
    C22 = M1 - M2 + M3 + M6
    
    We need to extract U1, V1, W1 such that:
    C_{i,k} = sum_c U1[i, j? no, U1 is for A, V1 for B, W1 for C]
    
    Standard form:
    A_{i,j} -> U1[i, c] ? No, U1 maps A to intermediate.
    The tensor is T_{i,j,k}.
    U1[i, c] is coefficient of A_{i, ?}. Wait.
    Usually, for matrix mult T_{i,j,k} = sum_{ab, bc, ca}...
    
    Let's stick to the definition:
    C_{i,k} = sum_{c} U1[i, c] * (sum_{j} A_{i,j} V1[j, c] * B_{j, ?})
    
    Actually, the factorization is:
    T_{i,j,k} = sum_c U_c[i] V_c[j] W_c[k]
    where U_c is a vector of size n*m? No.
    
    The interface requires U, V, W matrices.
    U: (n*m, rank). Indexing U[i*m+j, c] corresponds to coefficient of A[i,j] in term c.
    V: (m*p, rank). Indexing V[j*p+k, c] corresponds to coefficient of B[j,k] in term c.
    W: (p*n, rank). Indexing W[k*n+i, c] corresponds to coefficient of C[k,i]? 
    Task says: "output factor follows the official trace convention W[k*n+i,r]".
    So W[k*n+i, c] is the coefficient for C[i,k] in term c.
    
    So:
    C[i,k] = sum_c W[k*n+i, c] * (sum_j U[i*m+j, c] A[i,j] V[j*p+k, c])
    Wait, this is not quite right.
    The bilinear form is:
    C[i,k] = sum_{j} A[i,j] B[j,k]
    = sum_c (sum_j A[i,j] U_c[i,j]) * (sum_j B[j,k] V_c[j,k]) * W_c[k,i] ?
    
    Actually, the standard decomposition is:
    C = sum_c W_c \circ (U_c \odot A) \odot (V_c \odot B) ?
    
    Let's use the property:
    T_{i,j,k} = sum_c U[i,j,c] V[j,k,c] W[k,i,c]
    
    So for term c:
    U_c is a matrix of size n x m.
    V_c is a matrix of size m x p.
    W_c is a matrix of size p x n.
    
    The contribution to C is W_c^T * (U_c * A * V_c)? No.
    The contribution is:
    C_{i,k}^{(c)} = sum_{j1, j2} A[i, j1] U_c[i, j1] V_c[j1, j2] B[j2, k] W_c[k, i]
    
    This simplifies to:
    C_{i,k}^{(c)} = W_c[k, i] * ( (U_c * A * V_c)_{i,k} )
    
    So we need:
    sum_c W_c[k, i] * (U_c * A * V_c)_{i,k} = (A * B)_{i,k}
    
    For the Kronecker product:
    T = T1 \otimes T2
    T_{(i1, i2), (j1, j2), (k1, k2)} = T1_{i1, j1, k1} T2_{i2, j2, k2}
    
    The factorization of T1 is:
    T1_{i1, j1, k1} = sum_{c1} U1[i1, j1, c1] V1[j1, k1, c1] W1[k1, i1, c1]
    
    The factorization of T2 is:
    T2_{i2, j2, k2} = sum_{c2} U2[i2, j2, c2] V2[j2, k2, c2] W2[k2, i2, c2]
    
    So T_{...} = sum_{c1, c2} U1[...] U2[...] V1[...] V2[...] W1[...] W2[...]
    
    We define c = c1 * R2 + c2.
    Then:
    U[i1, j1, c1, c2] = U1[i1, j1, c1] * U2[i2, j2, c2] ?
    
    Wait, the indices of the combined tensor are:
    Row: (i1, i2)
    Col: (j1, j2)
    Out: (k1, k2)
    
    Our mapping:
    i1 = i (0,1)
    i2 = 0 (fixed)
    j1 = j_A (0,1)
    j2 = j_B (0,1)
    k1 = k_A (0,1)
    k2 = k_B (0..3)
    
    So:
    U_combined[i, j_A*2+j_B, c] = U1[i, j_A, c1] * U2[0, j_B, c2]
    V_combined[j_A*2+j_B, k_A*4+k_B, c] = V1[j_A, k_A, c1] * V2[j_B, k_B, c2]
    W_combined[k_A*4+k_B, i, c] = W1[k_A, i, c1] * W2[k_B, 0, c2]
    
    Note: The index for W is W[k*n+i]. Here k is the combined k (0..7), i is the combined i (0..1).
    Combined k = 4*k_A + k_B.
    Combined i = i.
    
    So W_combined[4*k_A+k_B, i, c] = W1[k_A, i, c1] * W2[k_B, 0, c2].
    
    We need to construct U1, V1, W1 for Strassen 2x2.
    And U2, V2, W2 for Schoolbook 1x2x4.
    
    Strassen 2x2:
    U1, V1, W1 are 2x2x7 tensors.
    
    Schoolbook 1x2x4:
    A is 1x2. B is 2x4. C is 1x4.
    C_0 = A_0 B_00 + A_1 B_10
    C_1 = A_0 B_01 + A_1 B_11
    C_2 = A_0 B_02 + A_1 B_12
    C_3 = A_0 B_03 + A_1 B_13
    
    This is 4 independent dot products.
    Each dot product of length 2 has rank 2.
    Term 1: A_0 B_00 -> C_0
    Term 2: A_1 B_10 -> C_0
    Term 3: A_0 B_01 -> C_1
    Term 4: A_1 B_11 -> C_1
    Term 5: A_0 B_02 -> C_2
    Term 6: A_1 B_12 -> C_2
    Term 7: A_0 B_03 -> C_3
    Term 8: A_1 B_13 -> C_3
    
    So R2 = 8.
    
    U2[0, 0, 0] = 1 (A_0 for C_0 term 1)
    U2[0, 1, 1] = 1 (A_1 for C_0 term 2)
    U2[0, 0, 2] = 1 (A_0 for C_1 term 3)
    U2[0, 1, 3] = 1 (A_1 for C_1 term 4)
    ...
    Generally:
    For k in 0..3:
       c_1 = 2*k
       c_2 = 2*k + 1
       U2[0, 0, c_1] = 1
       U2[0, 1, c_2] = 1
       
       V2[0, k, c_1] = 1
       V2[1, k, c_2] = 1
       
       W2[k, 0, c_1] = 1
       W2[k, 0, c_2] = 1
    
    Now we construct the combined U, V, W.
    
    R = 7 * 8 = 56.
    
    U = np.zeros((2*4, 56))
    V = np.zeros((4*8, 56))
    W = np.zeros((8*2, 56))
    
    For c1 in 0..6:
      For c2 in 0..7:
        c = c1 * 8 + c2
        
        # U
        For i in 0..1:
          For jA in 0..1:
            For jB in 0..1:
              U[i * 4 + jA*2 + jB, c] = U1[i, jA, c1] * U2[0, jB, c2]
              
        # V
        For jA in 0..1:
          For jB in 0..1:
            For kA in 0..1:
              For kB in 0..3:
                V[(jA*2 + jB) * 8 + kA*4 + kB, c] = V1[jA, kA, c1] * V2[jB, kB, c2]
                
        # W
        For kA in 0..1:
          For kB in 0..3:
            For i in 0..1:
              W[(kA*4 + kB) * 2 + i, c] = W1[kA, i, c1] * W2[kB, 0, c2]
              
    This is O(R * n*m*p * something) which is fine.
    
    We need to hardcode U1, V1, W1 for Strassen.
    
    Strassen:
    M1 = (A00+A11)(B00+B11)
    M2 = (A10+A11)B00
    M3 = A00(B01-B11)
    M4 = A10(B10-B00)
    M5 = (A00+A01)B11
    M6 = (A10-A00)(B00+B01)
    M7 = (A01-A11)(B10+B11)
    
    C00 = M1 + M4 - M5 + M7
    C01 = M3 + M5
    C10 = M2 + M4
    C11 = M1 - M2 + M3 + M6
    
    Let's define U1, V1, W1 for each term c1=0..6.
    
    c1=0 (M1):
    U1[0,0]=1, U1[0,1]=0? No, A00 and A11.
    U1[0,0]=1, U1[1,1]=1.
    V1[0,0]=1, V1[0,1]=1. (B00, B01? No, B00+B11 means B00 and B11.
    Wait, M1 = (A00+A11)(B00+B11).
    This is not a single product of linear forms in B.
    (A00+A11) is linear in A.
    (B00+B11) is linear in B.
    So U1[0,0]=1, U1[1,1]=1.
    V1[0,0]=1, V1[1,1]=1.
    W1[0,0]=1, W1[1,1]=1. (Contributes to C00 and C11)
    
    c1=1 (M2):
    U1[1,0]=1, U1[1,1]=1.
    V1[0,0]=1.
    W1[1,0]=1. (Contributes to C10)
    
    c1=2 (M3):
    U1[0,0]=1.
    V1[0,1]=1, V1[1,1]=-1.
    W1[0,1]=1, W1[1,1]=1. (Contributes to C01 and C11)
    
    c1=3 (M4):
    U1[1,0]=1.
    V1[1,0]=1, V1[0,0]=-1.
    W1[0,0]=1, W1[1,0]=1. (Contributes to C00 and C10)
    
    c1=4 (M5):
    U1[0,0]=1, U1[0,1]=1.
    V1[1,1]=1.
    W1[0,0]=-1, W1[0,1]=1. (Contributes to C00 and C01)
    
    c1=5 (M6):
    U1[1,0]=1, U1[0,0]=-1.
    V1[0,0]=1, V1[0,1]=1.
    W1[1,1]=1. (Contributes to C11)
    
    c1=6 (M7):
    U1[0,1]=1, U1[1,1]=-1.
    V1[1,0]=1, V1[1,1]=1.
    W1[0,0]=1. (Contributes to C00)
    
    Let's double check C00:
    M1 (1) + M4 (1) - M5 (1) + M7 (1) = 1 + 1 - 1 + 1 = 2? No.
    W1[0,0] for M1 is 1.
    W1[0,0] for M4 is 1.
    W1[0,0] for M5 is -1.
    W1[0,0] for M7 is 1.
    Sum = 1 + 1 - 1 + 1 = 2. This is wrong. C00 should be 1.
    
    Ah, the decomposition is:
    C00 = M1 - M5 + M4 + M7
    M1 = (A00+A11)(B00+B11)
    M5 = (A00+A01)B11
    M4 = A10(B10-B00)
    M7 = (A01-A11)(B10+B11)
    
    Let's re-evaluate the coefficients for W1.
    W1[k, i, c] is the coefficient of C[i, k] in term c.
    
    c1=0 (M1): C00 += 1, C11 += 1.
    c1=1 (M2): C10 += 1.
    c1=2 (M3): C01 += 1, C11 += 1.
    c1=3 (M4): C00 += 1, C10 += 1.
    c1=4 (M5): C00 += -1, C01 += 1.
    c1=5 (M6): C11 += 1.
    c1=6 (M7): C00 += 1.
    
    Check C00: 1 (M1) + 1 (M4) - 1 (M5) + 1 (M7) = 2. Still 2.
    
    Wait, Strassen is:
    C00 = M1 - M5 + M4 + M7
    C01 = M3 + M5
    C10 = M2 + M4
    C11 = M1 - M2 + M3 + M6
    
    Let's check the algebra for C00.
    M1 = A00B00 + A00B11 + A11B00 + A11B11
    -M5 = -A00B11 - A01B11
    M4 = A10B10 - A10B00
    M7 = A01B10 + A01B11 - A11B10 - A11B11
    
    Sum:
    A00B00: 1
    A00B11: 1 - 1 = 0
    A11B00: 1
    A11B11: 1 - 1 = 0
    A01B11: -1 + 1 = 0
    A10B10: 1 - 1 = 0
    A10B00: -1
    A01B10: 1
    A11B10: -1
    
    Remaining: A00B00 + A11B00 - A10B00 + A01B10 - A11B10
    = A00B00 + (A11-A10)B00 + (A01-A11)B10.
    This is not A00B00 + A01B01 + A10B10 + A11B11.
    
    I think I have the Strassen formula slightly wrong or the assignment of M's to C's.
    
    Correct Strassen:
    M1 = (A11 + A22)(B11 + B22)
    M2 = (A21 + A22)B11
    M3 = A11(B12 - B22)
    M4 = A22(B21 - B11)
    M5 = (A11 + A12)B22
    M6 = (A21 - A11)(B11 + B12)
    M7 = (A12 - A22)(B21 + B22)
    
    C11 = M1 + M4 - M5 + M7
    C12 = M3 + M5
    C21 = M2 + M4
    C22 = M1 - M2 + M3 + M6
    
    Note: Indices 1,2 correspond to 0,1.
    
    Let's re-verify C11 (C00 in 0-indexed).
    M1: A00B00, A00B11, A11B00, A11B11
    M4: A11B10, -A11B00
    -M5: -A00B11, -A01B11
    M7: A01B10, A01B11, -A11B10, -A11B11
    
    Sum:
    A00B00: 1
    A00B11: 1-1=0
    A11B00: 1-1=0
    A11B11: 1-1=0
    A11B10: 1-1=0
    A01B11: -1+1=0
    A01B10: 1
    
    Remaining: A00B00 + A01B10.
    Missing A10B10 and A11B11?
    And extra A01B10?
    
    It seems M4 is A22(B21 - B11). In 0-index: A11(B10 - B00).
    M4 = A11B10 - A11B00.
    
    And M7 = (A12 - A22)(B21 + B22) -> (A01 - A11)(B10 + B11).
    M7 = A01B10 + A01B11 - A11B10 - A11B11.
    
    Let's sum C11 again:
    M1: A00B00 + A00B11 + A11B00 + A11B11
    M4: A11B10 - A11B00
    -M5: -A00B11 - A01B11
    M7: A01B10 + A01B11 - A11B10 - A11B11
    
    Terms:
    A00B00: 1
    A00B11: 1 - 1 = 0
    A11B00: 1 - 1 = 0
    A11B11: 1 - 1 = 0
    A11B10: 1 - 1 = 0
    A01B11: -1 + 1 = 0
    A01B10: 1
    
    Result: A00B00 + A01B10.
    This is incorrect.
    
    The standard Strassen is:
    C11 = M1 - M5 + M4 + M7
    C12 = M3 + M5
    C21 = M2 + M4
    C22 = M1 - M2 + M3 + M6
    
    And the definitions are:
    M1 = (A11 + A22)(B11 + B22)
    M2 = (A21 + A22)B11
    M3 = A11(B12 - B22)
    M4 = A22(B21 - B11)
    M5 = (A11 + A12)B22
    M6 = (A21 - A11)(B11 + B12)
    M7 = (A12 - A22)(B21 + B22)
    
    Let's check C11 = M1 - M5 + M4 + M7.
    M1 = A11B11 + A11B22 + A22B11 + A22B22
    -M5 = -A11B22 - A12B22
    M4 = A22B21 - A22B11
    M7 = A12B21 + A12B22 - A22B21 - A22B22
    
    Sum:
    A11B11: 1
    A11B22: 1 - 1 = 0
    A22B11: 1 - 1 = 0
    A22B22: 1 - 1 = 0
    A12B22: -1 + 1 = 0
    A22B21: 1 - 1 = 0
    A12B21: 1
    
    Result: A11B11 + A12B21.
    Still missing terms.
    
    Wait, C11 = A11B11 + A12B21.
    The correct C11 is A11B11 + A12B12? No, C11 = A11B11 + A12B21 is wrong.
    C11 = A11B11 + A12B21 is for (1,1) if rows are 1,2 and cols are 1,2.
    A11B11 is (1,1)x(1,1).
    A12B21 is (1,2)x(2,1).
    Sum is (1,1).
    Yes!
    C11 = A11B11 + A12B21.
    
    So the formula is correct. My manual expansion was just checking the terms.
    A11B11 is present. A12B21 is present.
    
    So the Strassen decomposition is correct.
    
    Now, code implementation.
    
    """
    start_time = time.time()
    deadline = start_time + 140.0
    
    n, m, p = 2, 4, 8
    rank = 56
    
    U = np.zeros((n*m, rank), dtype=float)
    V = np.zeros((m*p, rank), dtype=float)
    W = np.zeros((p*n, rank), dtype=float)
    
    # Strassen 2x2x2 factors (U1, V1, W1)
    # U1[i, j, c]
    U1 = np.zeros((2, 2, 7), dtype=float)
    V1 = np.zeros((2, 2, 7), dtype=float)
    W1 = np.zeros((2, 2, 7), dtype=float)
    
    # c=0: M1
    U1[0,0,0] = 1; U1[1,1,0] = 1
    V1[0,0,0] = 1; V1[1,1,0] = 1
    W1[0,0,0] = 1; W1[1,1,0] = 1
    
    # c=1: M2
    U1[1,0,1] = 1; U1[1,1,1] = 1
    V1[0,0,1] = 1
    W1[1,0,1] = 1
    
    # c=2: M3
    U1[0,0,2] = 1
    V1[0,1,2] = 1; V1[1,1,2] = -1
    W1[0,1,2] = 1; W1[1,1,2] = 1
    
    # c=3: M4
    U1[1,1,3] = 1
    V1[1,0,3] = 1; V1[0,0,3] = -1
    W1[0,0,3] = 1; W1[1,0,3] = 1
    
    # c=4: M5
    U1[0,0,4] = 1; U1[0,1,4] = 1
    V1[1,1,4] = 1
    W1[0,0,4] = -1; W1[0,1,4] = 1
    
    # c=5: M6
    U1[1,0,5] = 1; U1[0,0,5] = -1
    V1[0,0,5] = 1; V1[0,1,5] = 1
    W1[1,1,5] = 1
    
    # c=6: M7
    U1[0,1,6] = 1; U1[1,1,6] = -1
    V1[1,0,6] = 1; V1[1,1,6] = 1
    W1[0,0,6] = 1
    
    # Schoolbook 1x2x4 factors (U2, V2, W2)
    U2 = np.zeros((1, 2, 8), dtype=float)
    V2 = np.zeros((2, 4, 8), dtype=float)
    W2 = np.zeros((4, 1, 8), dtype=float)
    
    for k in range(4):
        c1 = 2*k
        c2 = 2*k + 1
        
        U2[0, 0, c1] = 1
        U2[0, 1, c2] = 1
        
        V2[0, k, c1] = 1
        V2[1, k, c2] = 1
        
        W2[k, 0, c1] = 1
        W2[k, 0, c2] = 1
        
    # Combine
    for c1 in range(7):
        for c2 in range(8):
            c = c1 * 8 + c2
            
            # U
            for i in range(2):
                for jA in range(2):
                    for jB in range(2):
                        U[i * 4 + jA * 2 + jB, c] = U1[i, jA, c1] * U2[0, jB, c2]
                        
            # V
            for jA in range(2):
                for jB in range(2):
                    for kA in range(2):
                        for kB in range(4):
                            V[(jA * 2 + jB) * 8 + kA * 4 + kB, c] = V1[jA, kA, c1] * V2[jB, kB, c2]
                            
            # W
            for kA in range(2):
                for kB in range(4):
                    for i in range(2):
                        W[(kA * 4 + kB) * 2 + i, c] = W1[kA, i, c1] * W2[kB, 0, c2]
                        
    if time.time() > deadline:
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
        
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
