"""Matrix multiplication 4x4x4 via Kronecker product of rank-7 Strassen."""
import time
import numpy as np

DIMENSIONS = [4, 4, 4]

# EVOLVE-BLOCK-START

def _strassen_2x2():
    """
    Return U, V, W (each 4x7) for 2x2 matrix multiplication using rank-7 Strassen.
    Indexing convention:
      U indexed by (i,j) in row-major: i*2+j, where i is row of A, j is col of A
      V indexed by (j,k) in row-major: j*2+k, where j is row of B, k is col of B
      W indexed by (k,i) in row-major: k*2+i, where k is row of C, i is col of C
    The tensor identity: sum_r U[i,j,r] * V[j,k,r] * W[k,i,r] = 1 for all i,j,k
    which corresponds to C[k,i] = sum_{j} A[i,j] B[j,k].
    """
    # Strassen for 2x2:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B10-B11)
    # M4 = A11(B01-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B10)
    # M7 = (A01-A11)(B01+B11)
    #
    # C00 = M1 - M4 + M5 - M7
    # C01 = M3 + M7
    # C10 = M2 - M6
    # C11 = M1 - M2 + M3 + M6
    
    # U: rows indexed by (i,j), 4 rows. For each of 7 factors, which A elements appear.
    # A elements: A00=(0,0), A01=(0,1), A10=(1,0), A11=(1,1)
    # Row order: (0,0)=0, (0,1)=1, (1,0)=2, (1,1)=3
    
    U = np.zeros((4, 7), dtype=int)
    # M1: A00+A11
    U[0, 0] = 1; U[3, 0] = 1
    # M2: A10+A11
    U[2, 1] = 1; U[3, 1] = 1
    # M3: A00
    U[0, 2] = 1
    # M4: A11
    U[3, 3] = 1
    # M5: A00+A01
    U[0, 4] = 1; U[1, 4] = 1
    # M6: A10-A00
    U[2, 5] = 1; U[0, 5] = -1
    # M7: A01-A11
    U[1, 6] = 1; U[3, 6] = -1
    
    # V: rows indexed by (j,k), 4 rows. j is row of B, k is col of B.
    # B elements: B00=(0,0), B01=(0,1), B10=(1,0), B11=(1,1)
    # Row order: (0,0)=0, (0,1)=1, (1,0)=2, (1,1)=3
    
    V = np.zeros((4, 7), dtype=int)
    # M1: B00+B11
    V[0, 0] = 1; V[3, 0] = 1
    # M2: B00
    V[0, 1] = 1
    # M3: B10-B11
    V[2, 2] = 1; V[3, 2] = -1
    # M4: B01-B00
    V[1, 3] = 1; V[0, 3] = -1
    # M5: B11
    V[3, 4] = 1
    # M6: B00+B10
    V[0, 5] = 1; V[2, 5] = 1
    # M7: B01+B11
    V[1, 6] = 1; V[3, 6] = 1
    
    # W: rows indexed by (k,i), 4 rows. k is row of C, i is col of C.
    # C elements: C00=(0,0), C01=(0,1), C10=(1,0), C11=(1,1)
    # Row order: (0,0)=0, (0,1)=1, (1,0)=2, (1,1)=3
    # Note: W[k*n+i] so for 2x2, W[k*2+i]
    
    W = np.zeros((4, 7), dtype=int)
    # M1: C00, C11
    W[0, 0] = 1; W[3, 0] = 1
    # M2: C10 (negative)
    W[2, 1] = -1
    # M3: C01, C11
    W[1, 2] = 1; W[3, 2] = 1
    # M4: C00 (negative)
    W[0, 3] = -1
    # M5: C00
    W[0, 4] = 1
    # M6: C11
    W[3, 5] = 1
    # M7: C00 (negative), C01
    W[0, 6] = -1; W[1, 6] = 1
    
    return U, V, W


def _kron_product(U1, V1, W1, n1):
    """
    Compute the Kronecker product decomposition for n1 x n1 matrix multiplication.
    Given rank-r decomposition for 2x2 (or n1x1), produce one for (2*n1)x(2*n1).
    
    For n1=2, we get 4x4 with rank 7*7=49.
    
    The Kronecker product of matrix multiplication:
    If we partition A, B, C into 2x2 blocks of n1 x n1 blocks,
    A = [[A00, A01], [A10, A11]], etc.
    Then C = A*B can be computed via Strassen's formula on the blocks,
    where each block product is itself an n1 x n1 matrix product.
    
    For the tensor decomposition:
    U[(i1,i2),(j1,j2), (r1,r2)] = U1[i1,j1,r1] * U1[i2,j2,r2]  -- NO, this isn't right.
    
    Actually, the Kronecker product of tensor decompositions works as follows:
    If T1 has decomposition sum_r U1[a,b,r] V1[b,c,r] W1[c,a,r],
    and T2 has decomposition sum_s U2[a,b,s] V2[b,c,s] W2[c,a,s],
    then T1 x T2 (the tensor product) has decomposition:
    sum_{r,s} (U1 x U2)[(a1,a2),(b1,b2),(r,s)] * (V1 x V2)[(b1,b2),(c1,c2),(r,s)] * (W1 x W2)[(c1,c2),(a1,a2),(r,s)]
    where (U1 x U2)[(a1,a2),(b1,b2),(r,s)] = U1[a1,b1,r] * U2[a2,b2,s]
    
    But matrix multiplication is NOT simply the tensor product of two matrix multiplication tensors.
    Instead, for 4x4 = 2x2 blocks of 2x2, we use the Strassen formula at the block level,
    and each block multiplication uses the 2x2 decomposition.
    
    Let me think more carefully.
    
    For 4x4 matrix multiplication, partition into 2x2 blocks of 2x2 matrices.
    A = [[A00, A01], [A10, A11]] where each block is 2x2.
    B = [[B00, B01], [B10, B11]] where each block is 2x2.
    C = [[C00, C01], [C10, C11]] where each block is 2x2.
    
    Using Strassen:
    C00 = M1 - M4 + M5 - M7
    C01 = M3 + M7
    C10 = M2 - M6
    C11 = M1 - M2 + M3 + M6
    
    where M1 = (A00+A11)(B00+B11), etc.
    
    Each M is a 2x2 matrix product, which we can decompose using the rank-7 Strassen.
    
    So the total rank is 7 (block-level Strassen factors) * 7 (each block product) = 49.
    
    Now, for the index mapping:
    A[i,j] for i,j in 0..3.
    Block index: i_block = i // 2, i_in_block = i % 2.
    So A[i,j] is in block A[i_block, j_block] at position (i_in_block, j_in_block).
    
    For the U matrix of the 4x4 decomposition:
    U has shape (16, 49) where row index is i*4+j (i row of A, j col of A),
    and column index is (r_block, r_sub) where r_block in 0..6 (Strassen factor)
    and r_sub in 0..6 (sub-Strassen factor).
    
    U[i,j, (r_block, r_sub)] = U_block[i_block, j_block, r_block] * U_sub[i_in_block, j_in_block, r_sub]
    where U_block is the 2x2 U for the block-level Strassen, and U_sub is the 2x2 U for
    the sub-block level.
    
    Wait, but that's not quite right either. The block-level Strassen combines blocks,
    and each combination involves adding/subtracting blocks. So:
    
    For example, M1 = (A00+A11)(B00+B11).
    The A-part is A00 + A11, which is a 2x2 matrix that is the sum of two 2x2 matrices.
    The B-part is B00 + B11.
    Then M1 = (A00+A11) * (B00+B11).
    
    To decompose this product, we need to express (A00+A11) and (B00+B11) in terms of
    the original A and B entries.
    
    (A00+A11)[i,j] = A[2*i, 2*j] + A[2*i+2, 2*j+2] for i,j in 0,1.
    Wait, no. A00 is the top-left 2x2 block, so A00[i,j] = A[2*i, 2*j].
    A11 is the bottom-right 2x2 block, so A11[i,j] = A[2*i+2, 2*j+2].
    So (A00+A11)[i,j] = A[2*i, 2*j] + A[2*i+2, 2*j+2].
    
    Similarly for B.
    
    Now, (A00+A11) * (B00+B11) is a 2x2 matrix product.
    Let's call X = A00+A11 and Y = B00+B11.
    Then M1 = X * Y, where X and Y are 2x2.
    
    Using the rank-7 Strassen for X*Y:
    M1[k,l] = sum_{r=0}^{6} U_strassen[k,l,r] * V_strassen[l,m,r] * ... 
    
    Hmm, this is getting complex. Let me use a different approach.
    
    The Kronecker product of tensor decompositions:
    
    Actually, I think the correct way is:
    
    The 4x4 matrix multiplication tensor is related to the 2x2 tensor via:
    T_4x4[(i1,i2),(j1,j2),(k1,k2)] = T_2x2[i1,j1,k1] * T_2x2[i2,j2,k2]
    
    Wait no, that's the tensor product, which gives the multiplication of 4x4 matrices
    when viewed as 2x2 blocks of 2x2 matrices, but with each block element being a
    scalar. That's not quite right.
    
    Let me reconsider. The standard way to get a 4x4 decomposition from a 2x2 one:
    
    If we have a rank-r decomposition for 2x2 matrix multiplication:
    sum_{s=1}^{r} U_s[a,b] V_s[b,c] W_s[c,a] = delta (for all a,b,c in 0,1)
    
    Then for 4x4, we can write:
    sum_{s=1}^{r} sum_{t=1}^{r} (U_s x U_t)[(a1,a2),(b1,b2)] (V_s x V_t)[(b1,b2),(c1,c2)] (W_s x W_t)[(c1,c2),(a1,a2)]
    = sum_{s,t} U_s[a1,b1]U_t[a2,b2] V_s[b1,c1]V_t[b2,c2] W_s[c1,a1]W_t[c2,a2]
    = (sum_s U_s[a1,b1]V_s[b1,c1]W_s[c1,a1]) * (sum_t U_t[a2,b2]V_t[b2,c2]W_t[c2,a2])
    = delta[a1,b1,c1] * delta[a2,b2,c2]
    
    But this gives 1 only when (a1,b1,c1) is a valid 2x2 product AND (a2,b2,c2) is a valid 2x2 product.
    For 4x4 matrix multiplication, we need:
    sum_{j1,j2} A[a1,a2, j1,j2] * B[j1,j2, b1,b2] = C[a1,a2, b1,b2]
    where the sum is over j1 in 0,1 and j2 in 0,1.
    
    The tensor product approach gives:
    sum_{j1,j2} (sum_s U_s[a1,j1]V_s[j1,b1]W_s[b1,a1]) (sum_t U_t[a2,j2]V_t[j2,b2]W_t[b2,a2])
    which is not the same as the 4x4 product.
    
    So the simple Kronecker product of decompositions does NOT directly give the 4x4 multiplication.
    
    The correct approach is to use the Strassen formula at the block level, as I described earlier.
    Let me implement that directly.
    """
    # This function is not used as described; I'll implement the block-level Strassen directly.
    pass


def construct():
    """
    Construct a rank-49 exact tensor decomposition for 4x4 matrix multiplication
    using the Kronecker product (block-level) of the rank-7 Strassen decomposition.
    """
    start_time = time.time()
    deadline = start_time + 140  # Safety margin
    
    n = 4
    U2, V2, W2 = _strassen_2x2()
    
    # U2: shape (4, 7), indexed by (i,j) for i,j in 0,1 (row, col of 2x2 matrix)
    # V2: shape (4, 7), indexed by (j,k) for j,k in 0,1 (row, col of 2x2 matrix)  
    # W2: shape (4, 7), indexed by (k,i) for k,i in 0,1 (row, col of 2x2 output)
    
    # For 4x4, partition into 2x2 blocks of 2x2 matrices.
    # A[i,j] where i,j in 0..3.
    # i_block = i // 2, i_sub = i % 2
    # j_block = j // 2, j_sub = j % 2
    
    # Strassen block-level formula:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B10-B11)
    # M4 = A11(B01-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B10)
    # M7 = (A01-A11)(B01+B11)
    
    # Each M is a 2x2 product, decomposed with rank-7 Strassen.
    # Total rank = 7 * 7 = 49.
    
    # U4: shape (16, 49)
    # Row index: i*4 + j, where i is row of A (0..3), j is col of A (0..3)
    # Col index: m1*7 + m2, where m1 in 0..6 (block Strassen factor), m2 in 0..6 (sub Strassen factor)
    
    U4 = np.zeros((16, 49), dtype=int)
    V4 = np.zeros((16, 49), dtype=int)
    W4 = np.zeros((16, 49), dtype=int)
    
    # Define the block-level Strassen coefficients.
    # For each block factor f in 0..6:
    #   A-part: which blocks of A are involved, with what signs
    #   B-part: which blocks of B are involved, with what signs
    
    # A-blocks: A00, A01, A10, A11 (indexed by (ib, jb) where ib,jb in 0,1)
    # B-blocks: B00, B01, B10, B11
    
    # For factor f, A-part is a linear combination of A-blocks:
    #   A_f = sum_{ib,jb} a_coeff[f][ib][jb] * A_block[ib,jb]
    # Similarly B_f = sum_{jb,kb} b_coeff[f][jb][kb] * B_block[jb,kb]
    
    # Then M_f = A_f * B_f (2x2 matrix product)
    # And C_block[kb, ib] = sum_f c_coeff[f][kb][ib] * M_f
    
    # A coefficients (for each factor f, for each block (ib,jb)):
    a_coeff = [
        [[1, 0], [0, 1]],  # f=0: M1, A00 + A11
        [[0, 0], [1, 1]],  # f=1: M2, A10 + A11
        [[1, 0], [0, 0]],  # f=2: M3, A00
        [[0, 0], [0, 1]],  # f=3: M4, A11
        [[1, 1], [0, 0]],  # f=4: M5, A00 + A01
        [[0, 0], [1, 0]],  # f=5: M6, A10 - A00 => A10:1, A00:-1
        [[0, 1], [0, -1]], # f=6: M7, A01 - A11
    ]
    # Fix M6: A10 - A00
    a_coeff[5] = [[-1, 0], [1, 0]]
    
    # B coefficients (for each factor f, for each block (jb,kb)):
    b_coeff = [
        [[1, 0], [0, 1]],  # f=0: M1, B00 + B11
        [[1, 0], [0, 0]],  # f=1: M2, B00
        [[0, 0], [1, -1]], # f=2: M3, B10 - B11
        [[-1, 1], [0, 0]], # f=3: M4, B01 - B00
        [[0, 0], [0, 1]],  # f=4: M5, B11
        [[1, 1], [0, 0]],  # f=5: M6, B00 + B10
        [[0, 1], [0, 1]],  # f=6: M7, B01 + B11
    ]
    
    # C coefficients (for each factor f, for each block (kb,ib)):
    # C00 = M1 - M4 + M5 - M7
    # C01 = M3 + M7
    # C10 = M2 - M6
    # C11 = M1 - M2 + M3 + M6
    c_coeff = [
        [1, 0, 0, 0],  # f=0: M1
        [0, 0, -1, 0], # f=1: M2
        [0, 1, 0, 1],  # f=2: M3
        [-1, 0, 0, 1], # f=3: M4
        [1, 0, 0, 0],  # f=4: M5
        [0, 0, -1, 1], # f=5: M6
        [-1, 1, 0, 0], # f=6: M7
    ]
    # c_coeff[f][kb][ib] where kb is row block of C, ib is col block of C
    # Let me restructure: c_coeff[f][kb, ib]
    c_coeff = [
        [[1, 0], [0, 1]],  # f=0: M1 contributes to C00 and C11
        [[0, 0], [1, 0]],  # f=1: M2 contributes to C10
        [[0, 1], [0, 0]],  # f=2: M3 contributes to C01
        [[1, 0], [0, 0]],  # f=3: M4 contributes to C00
        [[1, 0], [0, 0]],  # f=4: M5 contributes to C00
        [[0, 0], [1, 0]],  # f=5: M6 contributes to C10
        [[1, 1], [0, 0]],  # f=6: M7 contributes to C00 and C01
    ]
    # Fix signs:
    # C00 = M1 - M4 + M5 - M7
    # C01 = M3 + M7
    # C10 = M2 - M6
    # C11 = M1 - M2 + M3 + M6
    
    c_coeff = [
        [[1, 0], [0, 1]],   # f=0 (M1): C00+=1, C11+=1
        [[0, 0], [1, 0]],   # f=1 (M2): C10+=1
        [[0, 1], [0, 0]],   # f=2 (M3): C01+=1
        [[1, 0], [0, 0]],   # f=3 (M4): C00+=1
        [[1, 0], [0, 0]],   # f=4 (M5): C00+=1
        [[0, 0], [1, 0]],   # f=5 (M6): C10+=1
        [[1, 1], [0, 0]],   # f=6 (M7): C00+=1, C01+=1
    ]
    # Apply signs:
    c_coeff[1][1][0] = 1   # M2 to C10: +1, but C10 = M2 - M6, so M2 is +1
    c_coeff[5][1][0] = -1  # M6 to C10: -1
    c_coeff[3][0][0] = -1  # M4 to C00: -1
    c_coeff[6][0][0] = -1  # M7 to C00: -1
    c_coeff[1][0][1] = -1  # M2 to C11: -1 (C11 = M1 - M2 + M3 + M6)
    c_coeff[3][1][1] = 0   # M4 to C11: 0
    c_coeff[6][1][1] = 0   # M7 to C11: 0
    
    # Let me just hardcode the correct c_coeff:
    # c_coeff[f][kb][ib] where kb in 0,1 is row block, ib in 0,1 is col block
    c_coeff = [
        [[1, 0], [0, 1]],   # M1: C00+=1, C11+=1
        [[0, 0], [1, 0]],   # M2: C10+=1
        [[0, 1], [0, 0]],   # M3: C01+=1
        [[1, 0], [0, 0]],   # M4: C00+=1
        [[1, 0], [0, 0]],   # M5: C00+=1
        [[0, 0], [1, 0]],   # M6: C10+=1
        [[1, 1], [0, 0]],   # M7: C00+=1, C01+=1
    ]
    # Now apply the correct signs from the Strassen formula:
    # C00 = M1 - M4 + M5 - M7
    # C01 = M3 + M7
    # C10 = M2 - M6
    # C11 = M1 - M2 + M3 + M6
    
    c_coeff[0][0][0] = 1; c_coeff[0][1][1] = 1   # M1: C00+=1, C11+=1
    c_coeff[1][1][0] = 1                        # M2: C10+=1
    c_coeff[2][0][1] = 1                        # M3: C01+=1
    c_coeff[3][0][0] = -1                       # M4: C00-=1
    c_coeff[4][0][0] = 1                        # M5: C00+=1
    c_coeff[5][1][0] = -1                       # M6: C10-=1
    c_coeff[6][0][0] = -1; c_coeff[6][0][1] = 1 # M7: C00-=1, C01+=1
    # M1 also contributes to C11:
    c_coeff[0][1][1] = 1
    # M2 also contributes to C11 (negative):
    c_coeff[1][1][1] = -1
    # M3 also contributes to C11:
    c_coeff[2][1][1] = 1
    # M6 also contributes to C11:
    c_coeff[5][1][1] = 1
    
    # Now build U4, V4, W4.
    # For each block factor f (0..6) and sub factor s (0..6):
    #   col = f*7 + s
    #   
    #   For A (U4):
    #   A_f is a 2x2 matrix: A_f[ib_sub, jb_sub] = sum_{ib,jb} a_coeff[f][ib][jb] * A_block[ib,jb][ib_sub, jb_sub]
    #   A_block[ib,jb][ib_sub, jb_sub] = A[(ib*2+ib_sub), (jb*2+jb_sub)]
    #   So A_f[ib_sub, jb_sub] = sum_{ib,jb} a_coeff[f][ib][jb] * A[(ib*2+ib_sub), (jb*2+jb_sub)]
    #   
    #   The U4 for the sub-product of A_f * B_f is:
    #   U4[(ib*2+ib_sub), (jb*2+jb_sub), f*7+s] = 
    #       sum over the Strassen sub-decomposition...
    #   
    #   Actually, A_f is defined as a linear combination of A-blocks. Each A-block is a 2x2 matrix.
    #   So A_f is a 2x2 matrix whose entries are linear combinations of A's entries.
    #   A_f[rs, tc] = sum_{ib,jb} a_coeff[f][ib][jb] * A[(ib*2+rs), (jb*2+tc)]
    #   
    #   Now, the product A_f * B_f is a 2x2 matrix product.
    #   Using the rank-7 Strassen for this product:
    #   (A_f * B_f)[rs, tc] = sum_s U2[rs, u, s] * V2[u, tc, s] * ... 
    #   Wait, U2 is indexed by (row, col) of the 2x2 matrix being multiplied.
    #   
    #   For the sub-product, the "A" matrix is A_f, which has entries:
    #   A_f[rs, tc] = sum_{ib,jb} a_coeff[f][ib][jb] * A[(ib*2+rs), (jb*2+tc)]
    #   
    #   In the sub-Strassen decomposition, U2[rs, tc, s] multiplies A_f[rs, tc].
    #   So the contribution to U4 is:
    #   U4[(ib*2+rs), (jb*2+tc), f*7+s] += a_coeff[f][ib][jb] * U2[rs, tc, s]
    #   
    #   Similarly for V4:
    #   B_f[tc, ud] = sum_{jb,kb} b_coeff[f][jb][kb] * B[(jb*2+tc), (kb*2+ud)]
    #   V4[(jb*2+tc), (kb*2+ud), f*7+s] += b_coeff[f][jb][kb] * V2[tc, ud, s]
    #   
    #   And for W4:
    #   (A_f * B_f)[rs, ud] contributes to C blocks via c_coeff.
    #   C_block[kb, ib][rs, ud] = sum_f c_coeff[f][kb][ib] * (A_f * B_f)[rs, ud]
    #   And (A_f * B_f)[rs, ud] = sum_s U2[rs, u, s] V2[u, ud, s] * ... 
    #   
    #   Wait, I need to be more careful. The Strassen decomposition says:
    #   (A_f * B_f)[rs, ud] = sum_s U2[rs, u, s] * V2[u, ud, s] * W2[ud, rs, s] ... no.
    #   
    #   The tensor identity is:
    #   sum_s U2[i, j, s] * V2[j, k, s] * W2[k, i, s] = 1 for all i,j,k
    #   which means: C[k, i] = sum_j A[i, j] B[j, k]
    #   
    #   So for the sub-product:
    #   (A_f * B_f)[ud, rs] = sum_u A_f[rs, u] * B_f[u, ud]
    #   = sum_u sum_{ib,jb} a_coeff[f][ib][jb] A[(ib*2+rs), (jb*2+u)] * sum_{jb2,kb} b_coeff[f][jb2][kb] B[(jb2*2+u), (kb*2+ud)]
    #   
    #   Hmm, the indices are getting confusing. Let me use different letters.
    #   
    #   Let the sub-product be X * Y = Z, where X = A_f, Y = B_f, Z is 2x2.
    #   Z[p, q] = sum_u X[p, u] Y[u, q]
    #   
    #   X[p, u] = sum_{ib,jb} a_coeff[f][ib][jb] * A[(ib*2+p), (jb*2+u)]
    #   Y[u, q] = sum_{jb,kb} b_coeff[f][jb][kb] * B[(jb*2+u), (kb*2+q)]
    #   
    #   The Strassen decomposition for X*Y:
    #   Z[p, q] = sum_s U2[p, u, s] V2[u, q, s] ... no, that's not right.
    #   
    #   The correct Strassen identity:
    #   sum_s U2[i, j, s] * V2[j, k, s] * W2[k, i, s] = delta (where delta=1 if the product is correct)
    #   
    #   For matrix multiplication C = A*B:
    #   C[k, i] = sum_j A[i, j] B[j, k]
    #   The tensor T[i, j, k] = 1 for all i,j,k (since each term A[i,j]*B[j,k] appears exactly once).
    #   So: sum_s U2[i, j, s] * V2[j, k, s] * W2[k, i, s] = 1
    #   
    #   For the sub-product Z = X*Y:
    #   Z[q, p] = sum_u X[p, u] Y[u, q]
    #   So the "i" in the Strassen formula is p (row of X), "j" is u (col of X / row of Y), "k" is q (col of Y).
    #   sum_s U2[p, u, s] * V2[u, q, s] * W2[q, p, s] = 1
    #   
    #   So Z[q, p] = sum_u X[p, u] Y[u, q]
    #   And using Strassen: this equals sum_s (sum_u U2[p,u,s] X[p,u]) * (sum_u V2[u,q,s] Y[u,q]) * W2[q,p,s]
    #   
    #   Wait, that's not how it works. The Strassen formula is:
    #   sum_s U2[p,u,s] V2[u,q,s] W2[q,p,s] = 1 for all p,u,q
    #   This means: X[p,u] * Y[u,q] = sum_s U2[p,u,s] V2[u,q,s] W2[q,p,s] * (X[p,u] * Y[u,q])
    #   No, that's circular.
    #   
    #   Let me start over with the correct formulation.
    #   
    #   The matrix multiplication tensor for 2x2 is:
    #   T[i, j, k] = 1 for all i, j, k in {0, 1}
    #   This means: C[k, i] = sum_j A[i, j] B[j, k]
    #   
    #   A rank-r decomposition of T is:
    #   T[i, j, k] = sum_{s=1}^r U[i, j, s] * V[j, k, s] * W[k, i, s]
    #   
    #   So for the sub-product Z = X * Y (where X, Y are 2x2):
    #   Z[q, p] = sum_u X[p, u] Y[u, q]
    #   = sum_u sum_s U2[p, u, s] * V2[u, q, s] * W2[q, p, s] * X[p, u] * Y[u, q] ... no.
    #   
    #   I think I'm overcomplicating this. Let me just directly compute the decomposition.
    
    # Direct approach:
    # For each f in 0..6 (block Strassen factor) and s in 0..6 (sub Strassen factor):
    #   col = f * 7 + s
    #   
    #   U4 row (i, j) where i,j in 0..3:
    #   i = ib*2 + p, j = jb*2 + u
    #   U4[i, j, col] = a_coeff[f][ib][jb] * U2[p, u, s]
    #   
    #   V4 row (j, k) where j,k in 0..3:
    #   j = jb*2 + u, k = kb*2 + q
    #   V4[j, k, col] = b_coeff[f][jb][kb] * V2[u, q, s]
    #   
    #   W4 row (k, i) where k,i in 0..3:
    #   k = kb*2 + q, i = ib*2 + p
    #   W4[k, i, col] = c_coeff[f][kb][ib] * W2[q, p, s]
    
    for f in range(7):
        for s in range(7):
            col = f * 7 + s
            for ib in range(2):
                for jb in range(2):
                    a_c = a_coeff[f][ib][jb]
                    if a_c == 0:
                        continue
                    for p in range(2):
                        for u in range(2):
                            i = ib * 2 + p
                            j = jb * 2 + u
                            U4[i * 4 + j, col] += a_c * U2[p * 2 + u, s]
            
            for jb in range(2):
                for kb in range(2):
                    b_c = b_coeff[f][jb][kb]
                    if b_c == 0:
                        continue
                    for u in range(2):
                        for q in range(2):
                            j = jb * 2 + u
                            k = kb * 2 + q
                            V4[j * 4 + k, col] += b_c * V2[u * 2 + q, s]
            
            for kb in range(2):
                for ib in range(2):
                    c_c = c_coeff[f][kb][ib]
                    if c_c == 0:
                        continue
                    for q in range(2):
                        for p in range(2):
                            k = kb * 2 + q
                            i = ib * 2 + p
                            W4[k * 4 + i, col] += c_c * W2[q * 2 + p, s]
    
    # Verify the decomposition
    if time.time() < deadline:
        # Check: sum_r U4[i,j,r] * V4[j,k,r] * W4[k,i,r] should be 1 for all i,j,k
        # This is expensive: 4*4*4*49 = 3136 operations, fine.
        error = 0
        for i in range(4):
            for j in range(4):
                for k in range(4):
                    total = 0
                    for r in range(49):
                        total += U4[i*4+j, r] * V4[j*4+k, r] * W4[k*4+i, r]
                    if total != 1:
                        error += 1
        if error > 0:
            # Fallback to schoolbook
            rank = 64
            U = [[0] * rank for _ in range(16)]
            V = [[0] * rank for _ in range(16)]
            W = [[0] * rank for _ in range(16)]
            col = 0
            for i in range(4):
                for j in range(4):
                    for k in range(4):
                        U[i*4+j][col] = 1
                        V[j*4+k][col] = 1
                        W[k*4+i][col] = 1
                        col += 1
            return {"U": U, "V": V, "W": W}
    
    # Convert to lists of lists for the interface
    U_list = U4.tolist()
    V_list = V4.tolist()
    W_list = W4.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}

# EVOLVE-BLOCK-END
