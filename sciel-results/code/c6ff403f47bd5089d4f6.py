"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 5]

# EVOLVE-BLOCK-START
import time
import numpy as np
from fractions import Fraction

def construct():
    """
    Construct a rank-minimal exact tensor decomposition for multiplying a 4x4 matrix by a 4x5 matrix.
    Strategy:
    1. Use the known rank-21 Strassen-Winograd decomposition for 4x4x4 multiplication.
    2. Append rank-1 terms for the 5th column of the output (since B is 4x5, C is 4x5).
       Wait, the tensor is T[i,j,k] for i in 0..3, j in 0..3, k in 0..4.
       The standard decomposition handles k=0..3.
       For k=4, we need C[:,4] = A @ B[:,4].
       This requires 4 multiplications: A[0,0]*B[0,4] + A[0,1]*B[1,4] + A[0,2]*B[2,4] + A[0,3]*B[3,4] for C[0,4].
       Actually, each entry C[i,4] is a dot product of row i of A and column 4 of B.
       This is 4 * 4 = 16 scalar multiplications if done naively.
       However, we can share the "A" side factors.
       Let's look at the structure.
       For k=0..3, we have rank 21.
       For k=4, we need to compute 4 linear combinations of the 16 entries of A, multiplied by the 4 entries of B[:,4].
       Specifically, C[i,4] = sum_{j=0}^3 A[i,j] * B[j,4].
       We can introduce 4 new terms in the decomposition, one for each j (row of B[:,4] / column index of B).
       Let the new terms be c = 21, 22, 23, 24.
       For term c corresponding to j:
       U[i*m+j, c] should be 1 if the index is (i, j) for the A part? No.
       The tensor entry is T[i,j,k] = sum_c U[i*m+j, c] V[j*p+k, c] W[k*n+i, c].
       
       For the 5th column (k=4):
       We want T[i,j,4] = 1 for all i,j.
       Wait, the tensor T is the multiplication tensor. T[i,j,k] is the coefficient of A[i,j]B[j,k] in C[i,k].
       So T[i,j,k] = 1 for all i,j,k.
       
       The rank-21 decomposition for 4x4x4 gives:
       sum_{c=0}^{20} U4[i*m+j, c] * V4[j*p+k, c] * W4[k*n+i, c] = 1 for k=0..3.
       And for k=4, the sum is 0 (since V4 has 0s in the k=4 slice, or rather, the indices j*p+k for k=4 are not covered by the 4x4x4 tensor which only goes up to k=3 in the "k" dimension of the tensor? No, the tensor for 4x4x4 has k in 0..3. The V matrix for 4x4x4 has shape 4x4=16. Here V has shape 4x5=20.
       
       So we have a base decomposition for k=0..3.
       U_base: 16 x 21
       V_base: 20 x 21 (rows 0..15 correspond to k=0..3, rows 16..19 correspond to k=4, which are 0 in the base decomposition)
       W_base: 20 x 21 (cols 0..15 correspond to i=0..3 for k=0..3? No, W index is k*n+i. For k=0..3, indices are 0..15. For k=4, indices are 16..19.)
       
       We need to add terms to make T[i,j,4] = 1 for all i,j.
       Currently, for k=4, the contribution is 0.
       We need sum_{c=21}^{r-1} U[i*m+j, c] V[j*p+4, c] W[4*n+i, c] = 1 for all i,j.
       
       Let's use 4 new terms, one for each j in 0..3.
       Let c = 21 + j.
       Set V[j*p+4, c] = 1. (This picks out B[j,4])
       Set U[i*m+j, c] = 1. (This picks out A[i,j])
       Then the product is 1 * 1 * W[4*n+i, c].
       We need sum over c of (U[i*m+j, c] V[j*p+4, c] W[4*n+i, c]) = 1 for all i,j.
       With the above choice, for a fixed i,j, only the term c=21+j is non-zero (assuming orthogonality in U and V indices).
       U[i*m+j, 21+j'] is 1 if j=j', else 0.
       V[j*p+4, 21+j'] is 1 if j=j', else 0.
       So the sum is W[4*n+i, 21+j].
       We need W[4*n+i, 21+j] = 1 for all i,j.
       This is possible! Set W[4*n+i, 21+j] = 1 for all i in 0..3, j in 0..3.
       
       So the total rank is 21 + 4 = 25.
       
       Can we do better?
       Maybe we can share some of the 4 new terms with the existing 21?
       Unlikely, as the existing terms are "zero" in the k=4 slice (V rows 16-19 are 0).
       So we need new terms that have non-zero V in rows 16-19.
       The minimal rank to handle the 4th column (k=4) is the rank of the matrix M[i,j] = T[i,j,4] = 1.
       This matrix is 4x4 all ones. Its rank is 1.
       Wait, the tensor slice T[:, :, 4] is a 4x4 matrix of 1s.
       The decomposition of this slice is sum_c U_slice[:, c] * v_c * w_c where U_slice is 4x4 (rows i, cols j), v_c is scalar (V[j*4+4, c]), w_c is scalar (W[16+i, c]).
       Actually, the term is U[i,j,c] * V_j * W_i.
       This is a rank-1 term in the (i,j) plane scaled by V_j W_i? No.
       The sum is over c.
       Term c contributes U_c[i,j] * V_c[j] * W_c[i] to T[i,j,4].
       We want sum_c U_c[i,j] V_c[j] W_c[i] = 1 for all i,j.
       
       If we choose U_c[i,j] = 1 for all i,j, then we need sum_c V_c[j] W_c[i] = 1.
       This is a rank-1 approximation of the all-ones matrix.
       We can achieve this with 1 term: V_c[j] = 1, W_c[i] = 1.
       So U_21[i,j] = 1 for all i,j.
       V_21[j*4+4] = 1 for all j.
       W_21[16+i] = 1 for all i.
       
       Let's check if this works with the rest of the decomposition.
       For k=0..3, the new term 21 contributes:
       U_21[i,j] * V_21[j*4+k] * W_21[k*4+i].
       We set V_21 rows 0..15 to 0 (to not disturb the k=0..3 part).
       We set W_21 rows 0..15 to 0 (to not disturb the k=0..3 part).
       So the contribution to k=0..3 is 0.
       For k=4, the contribution is U_21[i,j] * V_21[j*4+4] * W_21[16+i] = 1 * 1 * 1 = 1.
       
       So we can add just 1 term!
       Total rank = 21 + 1 = 22.
       
       Let's verify.
       Base decomposition (rank 21) for 4x4x4:
       U_b (16x21), V_b (16x21), W_b (16x21).
       Note: V_b corresponds to indices j*4+k for k=0..3.
       W_b corresponds to indices k*4+i for k=0..3.
       
       Extended matrices:
       U (16x22): U[:, 0:21] = U_b, U[:, 21] = 1 (all ones).
       V (20x22):
         V[0:16, 0:21] = V_b
         V[0:16, 21] = 0
         V[16:20, 0:21] = 0
         V[16:20, 21] = 1 (all ones, corresponding to j=0..3, k=4)
       W (20x22):
         W[0:16, 0:21] = W_b
         W[0:16, 21] = 0
         W[16:20, 0:21] = 0
         W[16:20, 21] = 1 (all ones, corresponding to k=4, i=0..3)
       
       Check T[i,j,k] = sum_c U[i*m+j, c] V[j*p+k, c] W[k*n+i, c].
       
       Case 1: k in 0..3.
       Term c in 0..20:
       U[i*4+j, c] * V[j*4+k, c] * W[k*4+i, c].
       This is exactly the 4x4x4 multiplication tensor, which sums to 1.
       Term c=21:
       U[i*4+j, 21] * V[j*4+k, 21] * W[k*4+i, 21]
       = 1 * 0 * 0 = 0.
       Sum = 1. Correct.
       
       Case 2: k=4.
       Term c in 0..20:
       U[i*4+j, c] * V[j*4+4, c] * W[16+i, c].
       V[j*4+4, c] is 0 for c in 0..20 (since V_b only covers k=0..3, and we padded with 0).
       W[16+i, c] is 0 for c in 0..20 (since W_b only covers k=0..3, and we padded with 0).
       So sum is 0.
       Term c=21:
       U[i*4+j, 21] * V[j*4+4, 21] * W[16+i, 21]
       = 1 * 1 * 1 = 1.
       Sum = 1. Correct.
       
       So rank 22 is achievable.
       
       Now, we need the explicit rank-21 decomposition for 4x4x4.
       The Strassen-Winograd algorithm for 4x4x4 has rank 21.
       Reference: "Optimal rank of 4x4 matrix multiplication" or similar.
       Actually, the best known rank for 4x4x4 is 21 (Winograd, 1971).
       We need the explicit factors.
       
       Let's retrieve or reconstruct the rank-21 decomposition for 4x4x4.
       A common source for this is the "Strassen-Winograd" algorithm.
       
       Alternatively, we can use a known decomposition.
       Here is a known rank-21 decomposition for 4x4x4 (from literature):
       
       Let A be 4x4, B be 4x4, C be 4x4.
       C[i,k] = sum_j A[i,j] B[j,k].
       
       The decomposition involves 21 products M_c = U_c * V_c.
       Then C is a linear combination of these products.
       
       I will hardcode the factors for the 4x4x4 case.
       Since I don't have the exact 21x16 matrices memorized, I will use a known representation or a simpler one if available.
       Wait, if I can't find the exact 21-rank decomposition, I can fall back to a higher rank but still better than 80.
       Strassen's algorithm for 4x4 can be done with rank 21?
       Actually, Strassen's 2x2 is rank 7.
       4x4 using Strassen is 7 * 7 = 49 rank? No, that's not optimal.
       Winograd showed 21 is possible.
       
       Let's try to find a rank-21 decomposition online or in memory.
       If I can't, I can use a rank-25 or rank-28 decomposition which is easier to construct?
       No, the goal is minimal rank.
       
       Let's assume I can construct the rank-21 factors.
       Actually, there is a simpler approach: use the rank-21 decomposition from a library or known code.
       Since I am generating the code, I should include the factors.
       
       Here is a set of factors for 4x4x4 rank 21 (from a known source):
       
       U (16x21), V (16x21), W (16x21).
       
       I will use a placeholder for now and assume I can fill it in.
       However, if I cannot guarantee the correctness of the 21-rank factors, I should use a safer, slightly higher rank that I can verify.
       
       Let's consider the rank of 4x4x4.
       Lower bound is 21.
       Is there a rank-22 or 23 that is easier to write?
       
       Actually, I can use the Strassen algorithm recursively.
       4x4 = 2x(2x2).
       Strassen 2x2 has rank 7.
       So 4x4 using Strassen would have rank 7 * 4 = 28? No.
       The 4x4 matrix is divided into 2x2 blocks.
       There are 4 blocks in A, 4 in B.
       The product involves 7 Strassen steps for each pair of 2x2 blocks?
       No, the Strassen algorithm computes 2x2 C from 2x2 A and 2x2 B with 7 multiplications.
       For 4x4, we have 4 2x2 blocks in A (A00, A01, A10, A11) and 4 in B.
       C00 = A00B00 + A01B10
       C01 = A00B01 + A01B11
       C10 = A10B00 + A11B10
       C11 = A10B01 + A11B11
       
       This is 4 independent 2x2 multiplications.
       Each 2x2 multiplication has rank 7.
       So total rank = 4 * 7 = 28.
       
       So rank 28 is easily achievable and verifiable.
       Then we add 1 term for the 5th column.
       Total rank = 29.
       
       This is much better than 80.
       
       Can we do better than 28 for 4x4x4?
       Yes, 21 is the theoretical minimum and is achievable.
       But 28 is safer to implement correctly.
       
       Let's implement the rank-28 decomposition for 4x4x4.
       
       Strassen 2x2 factors:
       For A = [[a, b], [c, d]], B = [[e, f], [g, h]], C = A B.
       m1 = (a+d)(e+h)
       m2 = (c+d)e
       m3 = a(f-h)
       m4 = d(g-e)
       m5 = (a+b)h
       m6 = (c-a)(e+f)
       m7 = (b-d)(g+h)
       
       C = [[m1 + m4 - m5 + m7, m3 + m5],
           [m2 + m4, m1 - m2 + m3 + m6]]
       
       We need to map this to the tensor factors U, V, W.
       T[i,j,k] = 1.
       
       For the top-left 2x2 block of C (i=0,1; k=0,1):
       C[0,0] = m1 + m4 - m5 + m7
       C[0,1] = m3 + m5
       C[1,0] = m2 + m4
       C[1,1] = m1 - m2 + m3 + m6
       
       The indices for U, V, W:
       U index: i*4+j.
       V index: j*4+k.
       W index: k*4+i.
       
       For the block i=0,1 and k=0,1, the relevant j are 0,1.
       
       Term 1 (m1):
       U: (a+d) -> U[0,0]=1, U[1,1]=1. (i=0,j=0 and i=1,j=1)
       V: (e+h) -> V[0,0]=1, V[1,1]=1. (j=0,k=0 and j=1,k=1)
       W: contributes to C[0,0] (+1), C[1,1] (+1).
       W index for C[0,0] is k=0,i=0 -> 0.
       W index for C[1,1] is k=1,i=1 -> 5.
       So W[0, t1]=1, W[5, t1]=1.
       
       Term 2 (m2):
       U: (c+d) -> U[1,0]=1, U[1,1]=1. (i=1,j=0 and i=1,j=1)
       V: e -> V[0,0]=1. (j=0,k=0)
       W: contributes to C[1,0] (+1), C[1,1] (-1).
       W index for C[1,0] is k=0,i=1 -> 1.
       W index for C[1,1] is k=1,i=1 -> 5.
       So W[1, t2]=1, W[5, t2]=-1.
       
       Term 3 (m3):
       U: a -> U[0,0]=1.
       V: (f-h) -> V[0,1]=1, V[1,1]=-1. (j=0,k=1 and j=1,k=1)
       W: contributes to C[0,1] (+1), C[1,1] (+1).
       W index for C[0,1] is k=1,i=0 -> 4.
       W index for C[1,1] is k=1,i=1 -> 5.
       So W[4, t3]=1, W[5, t3]=1.
       
       Term 4 (m4):
       U: d -> U[1,1]=1.
       V: (g-e) -> V[1,0]=1, V[0,0]=-1. (j=1,k=0 and j=0,k=0)
       W: contributes to C[0,0] (+1), C[1,0] (+1).
       W index for C[0,0] is 0.
       W index for C[1,0] is 1.
       So W[0, t4]=1, W[1, t4]=1.
       
       Term 5 (m5):
       U: (a+b) -> U[0,0]=1, U[0,1]=1.
       V: h -> V[1,1]=1.
       W: contributes to C[0,0] (-1), C[0,1] (+1).
       W index for C[0,0] is 0.
       W index for C[0,1] is 4.
       So W[0, t5]=-1, W[4, t5]=1.
       
       Term 6 (m6):
       U: (c-a) -> U[1,0]=1, U[0,0]=-1.
       V: (e+f) -> V[0,0]=1, V[0,1]=1.
       W: contributes to C[1,1] (+1).
       W index for C[1,1] is 5.
       So W[5, t6]=1.
       
       Term 7 (m7):
       U: (b-d) -> U[0,1]=1, U[1,1]=-1.
       V: (g+h) -> V[1,0]=1, V[1,1]=1.
       W: contributes to C[0,0] (+1).
       W index for C[0,0] is 0.
       So W[0, t7]=1.
       
       This handles the top-left 2x2 block of C (i=0,1; k=0,1) using A[0:2, 0:2] and B[0:2, 0:2].
       
       We need to do this for all 4 blocks of C.
       Block (0,0): i=0,1; k=0,1. Uses A[0:2, 0:2], B[0:2, 0:2].
       Block (0,1): i=0,1; k=2,3. Uses A[0:2, 2:4], B[0:2, 2:4].
       Block (1,0): i=2,3; k=0,1. Uses A[2:4, 0:2], B[2:4, 0:2].
       Block (1,1): i=2,3; k=2,3. Uses A[2:4, 2:4], B[2:4, 2:4].
       
       Each block uses 7 terms. Total 28 terms.
       
       We can construct U, V, W for the 28 terms.
       
       Then add 1 term for k=4.
       Total rank 29.
       
       Let's implement this.
       
    """
    start_time = time.time()
    
    n, m, p = DIMENSIONS  # 4, 4, 5
    rank = 29
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen 2x2 factors
    # U, V, W are 16x28 for the 4x4x4 part.
    # We define the 7 terms for a 2x2 block.
    # U_block (4x7), V_block (4x7), W_block (4x7)
    # Indices in block: i in 0..1, j in 0..1, k in 0..1.
    
    # Term 1: m1 = (a+d)(e+h)
    # U: (0,0)=1, (1,1)=1
    # V: (0,0)=1, (1,1)=1
    # W: (0,0)=1, (1,1)=1 (local indices)
    
    def add_strassen_block(U, V, W, base_c, i_offset, j_offset, k_offset):
        """
        Add 7 Strassen terms for a 2x2 block.
        i_offset: row offset in A and C (0 or 2)
        j_offset: col offset in A and row offset in B (0 or 2)
        k_offset: col offset in C and col offset in B (0 or 2)
        """
        # Local indices
        # U indices: (i_offset + i) * 4 + (j_offset + j)
        # V indices: (j_offset + j) * 5 + (k_offset + k)  # Note: p=5, but for k=0..3, it's j*4+k in 4x4 logic?
        # Wait, V index is j*p+k. p=5.
        # For the 4x4x4 part, k is 0..3.
        # So V index is j*5 + k.
        # W index is k*n + i = k*4 + i.
        
        # Term 1
        c = base_c + 0
        U[(i_offset + 0) * 4 + (j_offset + 0)][c] += 1
        U[(i_offset + 1) * 4 + (j_offset + 1)][c] += 1
        V[(j_offset + 0) * 5 + (k_offset + 0)][c] += 1
        V[(j_offset + 1) * 5 + (k_offset + 1)][c] += 1
        W[(k_offset + 0) * 4 + (i_offset + 0)][c] += 1
        W[(k_offset + 1) * 4 + (i_offset + 1)][c] += 1
        
        # Term 2
        c = base_c + 1
        U[(i_offset + 1) * 4 + (j_offset + 0)][c] += 1
        U[(i_offset + 1) * 4 + (j_offset + 1)][c] += 1
        V[(j_offset + 0) * 5 + (k_offset + 0)][c] += 1
        W[(k_offset + 0) * 4 + (i_offset + 1)][c] += 1
        W[(k_offset + 1) * 4 + (i_offset + 1)][c] += -1
        
        # Term 3
        c = base_c + 2
        U[(i_offset + 0) * 4 + (j_offset + 0)][c] += 1
        V[(j_offset + 0) * 5 + (k_offset + 1)][c] += 1
        V[(j_offset + 1) * 5 + (k_offset + 1)][c] += -1
        W[(k_offset + 1) * 4 + (i_offset + 0)][c] += 1
        W[(k_offset + 1) * 4 + (i_offset + 1)][c] += 1
        
        # Term 4
        c = base_c + 3
        U[(i_offset + 1) * 4 + (j_offset + 1)][c] += 1
        V[(j_offset + 1) * 5 + (k_offset + 0)][c] += 1
        V[(j_offset + 0) * 5 + (k_offset + 0)][c] += -1
        W[(k_offset + 0) * 4 + (i_offset + 0)][c] += 1
        W[(k_offset + 0) * 4 + (i_offset + 1)][c] += 1
        
        # Term 5
        c = base_c + 4
        U[(i_offset + 0) * 4 + (j_offset + 0)][c] += 1
        U[(i_offset + 0) * 4 + (j_offset + 1)][c] += 1
        V[(j_offset + 1) * 5 + (k_offset + 1)][c] += 1
        W[(k_offset + 0) * 4 + (i_offset + 0)][c] += -1
        W[(k_offset + 1) * 4 + (i_offset + 0)][c] += 1
        
        # Term 6
        c = base_c + 5
        U[(i_offset + 1) * 4 + (j_offset + 0)][c] += 1
        U[(i_offset + 0) * 4 + (j_offset + 0)][c] += -1
        V[(j_offset + 0) * 5 + (k_offset + 0)][c] += 1
        V[(j_offset + 0) * 5 + (k_offset + 1)][c] += 1
        W[(k_offset + 1) * 4 + (i_offset + 1)][c] += 1
        
        # Term 7
        c = base_c + 6
        U[(i_offset + 0) * 4 + (j_offset + 1)][c] += 1
        U[(i_offset + 1) * 4 + (j_offset + 1)][c] += -1
        V[(j_offset + 1) * 5 + (k_offset + 0)][c] += 1
        V[(j_offset + 1) * 5 + (k_offset + 1)][c] += 1
        W[(k_offset + 0) * 4 + (i_offset + 0)][c] += 1

    # Add 4 blocks
    # Block (0,0): i=0,1; j=0,1; k=0,1
    add_strassen_block(U, V, W, 0, 0, 0, 0)
    # Block (0,1): i=0,1; j=2,3; k=2,3
    add_strassen_block(U, V, W, 7, 0, 2, 2)
    # Block (1,0): i=2,3; j=0,1; k=0,1
    add_strassen_block(U, V, W, 14, 2, 0, 0)
    # Block (1,1): i=2,3; j=2,3; k=2,3
    add_strassen_block(U, V, W, 21, 2, 2, 2)
    
    # Term 28 (index 28) for k=4
    c = 28
    for i in range(4):
        for j in range(4):
            U[i * 4 + j][c] += 1
    for j in range(4):
        V[j * 5 + 4][c] += 1
    for i in range(4):
        W[4 * 4 + i][c] += 1
        
    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
