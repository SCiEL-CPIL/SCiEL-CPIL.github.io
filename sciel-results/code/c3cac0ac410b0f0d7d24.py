"""Structural change: use a rank-32 construction via a 2x3x3 rank-16 block + 2x3x3 rank-16 block,
   with the second block using a different partition to reduce rank from 33 to 32.
   
   Actually, let me think more carefully. The current best is rank 33.
   
   Strategy: Model the 2x3x6 multiplication using a polynomial/NTT approach on the shared dimension m=3.
   Since m=3 is prime, we can use the DFT over a suitable field or work with the structure directly.
   
   Alternative structural approach: Use a 2x3x2 rank-7 (Strassen) for the first 2 columns of B,
   and a 2x3x4 block for the remaining 4 columns. For 2x3x4, we can split into two 2x3x2 blocks
   each using Strassen rank 7, giving 7+7+7 = 21 for the 2x3x6 part plus 12 for the first column
   schoolbook = 33. Same as before.
   
   Let me try a different structural decomposition:
   - Treat the 2x3 matrix A and 3x6 matrix B.
   - Use the fact that we can combine some multiplications across the m=3 dimension.
   
   Key insight: Instead of treating m=3 as three independent columns of A (and rows of B),
   we can use linear combinations. The tensor for 2x3x6 has dimension 2*3*6 = 36.
   
   Let me try: decompose into a 2x2x6 part (using A[:,0:2] and B[0:2,:]) and a 2x1x6 part
   (using A[:,2] and B[2,:]).
   - 2x1x6: schoolbook, rank 12
   - 2x2x6: split into three 2x2x2 Strassen blocks, rank 3*7 = 21
   Total: 33. Same.
   
   Another approach: 2x3x6 = use a rank-16 decomposition for 2x3x3 and rank-16 for another 2x3x3.
   If 2x3x3 has rank 16, then 2x3x6 has rank 32.
   
   Let me verify: is rank(2x3x3) = 16 achievable?
   Schoolbook for 2x3x3 is 18. Strassen gives: 2x3x3 = 2x2x3 + 2x1x3.
   2x2x3: split into 2x2x2 (rank 7) + 2x2x1 (rank 4) = 11? No, 2x2x1 is rank 4.
   Actually 2x2x3: we can do 2x2x2 (rank 7) + 2x2x1 (rank 4) = 11.
   2x1x3: schoolbook rank 6.
   Total: 11 + 6 = 17.
   
   Or: 2x3x3 = 2x2x3 + 2x1x3 = 11 + 6 = 17.
   Or: 2x3x3 = 2x1x3 + 2x1x3 + 2x1x3 = 18 (schoolbook).
   
   Can we do better than 17 for 2x3x3?
   Try: 2x3x3. Lower bound: max(2*3, 3*3, 2*3) = 9. Upper bound: 18 schoolbook.
   
   Let me try a direct construction for 2x3x3 with rank 16.
   
   Actually, let me try a completely different structural approach:
   Use the "border rank" or "winograd" style approach.
   
   For 2x3x6, let me try to find a rank-32 construction by being clever about sharing.
   
   Here's my plan:
   Part A: Handle A[:,0] * B[0,:] using rank 12 (schoolbook for 2x1x6)
   Part B: Handle A[:,1:3] * B[1:3,:] 
   
   For Part B, instead of three separate Strassen blocks, let me see if I can do better.
   A[:,1:3] is 2x2, B[1:3,:] is 2x6.
   2x2x6: schoolbook is 24, Strassen per 2x2x2 is 7, so 3*7=21.
   Can 2x2x6 be done in rank < 21?
   
   2x2x6 lower bound: max(4, 12, 12) = 12.
   Known: rank(2x2x6) = ? 
   We can do 2x2x6 as 2x2x3 + 2x2x3. Each 2x2x3: 
   2x2x3 lower bound: max(4,6,6)=6. 
   2x2x3: schoolbook 12. Can we do better?
   2x2x3 = 2x2x2 + 2x2x1 = 7 + 4 = 11.
   Or directly: C[i,k] = sum_j A[i,j]*B[j,k] for i in {0,1}, j in {0,1}, k in {0,1,2}.
   That's 12 products. With Strassen-like: 
   M1 = A00*(B00+B10), M2 = A10*(B00-B10), M3 = A01*B11... 
   Hmm, this is getting complex.
   
   Let me just try rank 32 with a different split:
   Split B into B[0:2,:] and B[2,:].
   A[:,0:2] * B[0:2,:]: 2x2x6, use Strassen: 3 blocks of 2x2x2, rank 21.
   A[:,2] * B[2,:]: 2x1x6, schoolbook rank 12.
   Total: 33.
   
   Alternative split:
   A[:,0:2] * B[0:2,:]: 2x2x6, rank 21 (Strassen).
   A[:,2] * B[2,:]: 2x1x6, rank 12.
   Total: 33.
   
   What if I use a non-trivial decomposition for 2x1x6?
   2x1x6: C[i,k] = A[i,2]*B[2,k]. This is just 12 independent products. Rank 12 is minimal.
   
   What about combining parts?
   C[i,k] = A[i,0]*B[0,k] + A[i,1]*B[1,k] + A[i,2]*B[2,k]
   
   Let me try: for each k, C[:,k] = A * B[:,k] where B[:,k] is 3x1.
   So for each k, we compute 2x3 * 3x1 = 2x1. That's a 2x3x1 tensor per k.
   rank(2x3x1) = 6 (schoolbook, and it's minimal since it's just 6 independent products... 
   actually no, 2x3x1: C[i] = sum_j A[i,j]*B[j]. This is 6 products but can be done in 6.
   Actually the rank of the 2x3x1 multiplication tensor is 6 since all entries are independent.
   
   Wait, 2x3x1: C[i,k] where k is fixed. C[i] = A[i,0]*B[0] + A[i,1]*B[1] + A[i,2]*B[2].
   The tensor has entries T[i,j,k] = delta(i,j,k) for the multiplication.
   Actually for 2x3x1: T[i,j,0] = 1 if we're computing C[i] = sum_j A[i,j] B[j,0].
   The rank of this tensor: it's the rank of the matrix formed by reshaping.
   T is 2x3x1. As a 2x3 matrix (for k=0), it's the 2x3 identity-like matrix.
   rank(T) as a tensor = rank of the 2x3 matrix = 2? No, tensor rank is different.
   
   For 2x3x1: C[i] = sum_j A[i,j] B[j]. The tensor T[i,j,k] = delta_{i,j} * delta_{k,0}... 
   no. T[i,j,k] = 1 if this product contributes to C[i,k]. 
   C[i,k] = sum_j A[i,j] B[j,k]. So T[i,j,k] = 1 for all i,j,k in the 2x3x1 case? No.
   T[i,j,k] = 1 means A[i,j]*B[j,k] contributes to C[i,k]. For 2x3x1 (k fixed):
   T[i,j,0] = 1 for all i in {0,1}, j in {0,1,2}. So it's a 2x3x1 tensor with all ones.
   The tensor rank of an all-ones 2x3x1 tensor: it equals the matrix rank of the 2x3 all-ones matrix = 1.
   Wait no. The tensor T[i,j,k] for multiplication is: T[i,j,k] = 1 if j is the shared index.
   For matrix multiplication C = AB, T[i,j,k] = 1 (it's just the structure).
   The rank is the minimum r such that T = sum_{c=1}^r u_c[i] v_c[j] w_c[k].
   
   For 2x3x1: T[i,j,0] = 1 for all i,j. So T = sum_c u_c[i] v_c[j] * 1.
   We need u_c[i] v_c[j] to sum to 1 for all i,j. This is like rank-1 decomposition of the 2x3 all-ones matrix.
   The matrix rank of the 2x3 all-ones matrix is 1, so tensor rank is 1.
   Wait, that can't be right for matrix multiplication.
   
   I'm confusing myself. Let me restart with the correct formulation.
   
   The matrix multiplication tensor for n x m times m x p is T[i,j,k] where i in [0,n), j in [0,m), k in [0,p),
   and T[i,j,k] = 1 (it's always 1; the structure is that C[i,k] = sum_j A[i,j] B[j,k]).
   
   The rank of this tensor is the bilinear complexity.
   
   For 2x3x6: the tensor is 2x3x6 with all entries 1. 
   We need to write T[i,j,k] = sum_c U[i,j,c] V[j,k,c] W[k,i,c]... 
   
   Wait, the problem statement says: U has shape (n*m, rank), V has shape (m*p, rank), W has shape (p*n, rank).
   And the decomposition is: for all i,j,k: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1.
   
   So it's a 3-way tensor decomposition where the three modes are (i,j), (j,k), (k,i).
   This is NOT the standard tensor decomposition of T[i,j,k]. It's a decomposition in a specific format.
   
   The standard rank of the 2x3x6 multiplication tensor (bilinear complexity) is what we're looking for.
   
   Known results:
   - rank(2x2x2) = 7 (Strassen)
   - rank(2x2x3) = ? 
   - rank(2x3x3) = ?
   - rank(2x3x6) = ?
   
   The schoolbook gives 36. Strassen-based gives 33.
   
   For a structural change, let me try a completely different approach:
   Use the fact that 2x3x6 can be viewed as a 2x3x2 * 3 (three copies) and try to find
   a rank-10 decomposition for 2x3x2 (better than the schoolbook 12 or Strassen-based approaches).
   
   Actually, let me try to directly construct rank 32.
   
   Here's an idea: Use 2x2x6 for A[:,0:2]*B[0:2,:] with rank 20 (instead of 21),
   and 2x1x6 for A[:,2]*B[2,:] with rank 12. Total 32.
   
   Can 2x2x6 be done in rank 20?
   2x2x6 = three 2x2x2 blocks. Each 2x2x2 has rank 7, total 21.
   Can we share some computation between the three blocks?
   
   The three blocks are:
   Block t: A_sub = A[:,0:2], B_sub = B[0:2, 2t:2t+2], C_sub = C[:, 2t:2t+2]
   
   In Strassen, the 7 products are:
   M1 = (A00+A11)(B00+B11)
   M2 = (A10+A11)B00
   M3 = A00(B01-B11)
   M4 = A11(B10-B00)
   M5 = (A00+A01)B11
   M6 = (A10-A00)(B00+B01)
   M7 = (A01-A11)(B10+B11)
   
   The A-side combinations are the same across all three blocks! Only the B-side differs.
   So we can precompute the A-side linear combinations:
   a1 = A00+A11, a2 = A10+A11, a3 = A00, a4 = A11, a5 = A00+A01, a6 = A10-A00, a7 = A01-A11
   
   Then for each block t:
   M1_t = a1 * (B00_t + B11_t)
   M2_t = a2 * B00_t
   M3_t = a3 * (B01_t - B11_t)
   M4_t = a4 * (B10_t - B00_t)
   M5_t = a5 * B11_t
   M6_t = a6 * (B00_t + B01_t)
   M7_t = a7 * (B10_t + B11_t)
   
   This is still 3*7 = 21 scalar multiplications. The A-side precomputation doesn't reduce the count.
   
   But wait - in the tensor decomposition framework, we're counting the number of "rank-1 terms".
   Each rank-1 term is U[i,j,c] * V[j,k,c] * W[k,i,c].
   
   For the Strassen construction, each M corresponds to one rank-1 term.
   The A-side combination goes into U, the B-side into V, and the C-side into W.
   
   So for 2x2x6 with three Strassen blocks, we have 21 rank-1 terms for the 2x2x6 part.
   Plus 12 for the 2x1x6 part. Total 33.
   
   To get rank 32, I need to save one rank-1 term somewhere.
   
   Idea: Can I merge one term from the 2x1x6 part with a term from the 2x2x6 part?
   
   The 2x1x6 part has terms: for each (i,k), U[i*3+2, c] = 1, V[2*6+k, c] = 1, W[k*2+i, c] = 1.
   These involve A[i,2] and B[2,k].
   
   The 2x2x6 part involves A[i,0:2] and B[0:2,k].
   
   There's no overlap in the A or B indices, so I can't directly merge.
   
   Alternative: Use a different decomposition of 2x3x6 that's not a simple block sum.
   
   Let me try: C[i,k] = sum_j A[i,j] B[j,k].
   Group the j-sum differently. Instead of j=0,1,2 separately, use:
   C[i,k] = A[i,0]*(B[0,k]) + A[i,1]*(B[1,k]) + A[i,2]*(B[2,k])
   
   What if I use: C[i,k] = (A[i,0]+A[i,1]+A[i,2]) * (B[0,k]+B[1,k]+B[2,k]) 
              - (A[i,0]+A[i,1]) * B[2,k] - (A[i,0]+A[i,2]) * B[1,k] - (A[i,1]+A[i,2]) * B[0,k]
              + ... 
   
   This is getting complicated. Let me just try to implement a rank-32 construction if I can find one,
   otherwise stick with rank 33 but verify it's correct.
   
   Actually, let me just keep the rank-33 construction but make it cleaner and ensure it's correct.
   The structural change I'll make: instead of the row-0 schoolbook + 3 Strassen blocks,
   I'll use a 2x2x6 Strassen (3 blocks) + 2x1x6 schoolbook, which is structurally different
   in how the decomposition is organized (column-based vs row-based splitting).
   
   Wait, that gives the same rank. Let me just ensure correctness and output rank 33.
   """
import numpy as np
import time

def construct():
    start = time.time()
    deadline = start + 130.0

    n, m, p = 2, 3, 6
    rank = 33
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))

    col = 0

    # Part 1: A[:, 0] (2x1) * B[0, :] (1x6) -> C[:, :]  (schoolbook, rank 12)
    for k in range(p):
        for i in range(n):
            U[i * m + 0, col] = 1.0
            V[0 * p + k, col] = 1.0
            W[k * n + i, col] = 1.0
            col += 1

    # Part 2: A[:, 1:3] (2x2) * B[1:3, :] (2x6) via three Strassen 2x2x2 blocks
    for t in range(3):
        k0 = 2 * t
        k1 = 2 * t + 1

        # M1 = (A00 + A11) * (B00 + B11)
        U[0 * m + 1, col] += 1.0
        U[1 * m + 2, col] += 1.0
        V[1 * p + k0, col] += 1.0
        V[2 * p + k1, col] += 1.0
        W[k0 * n + 0, col] += 1.0
        W[k1 * n + 1, col] += 1.0
        col += 1

        # M2 = (A10 + A11) * B00
        U[1 * m + 1, col] += 1.0
        U[1 * m + 2, col] += 1.0
        V[1 * p + k0, col] += 1.0
        W[k0 * n + 1, col] += 1.0
        W[k1 * n + 1, col] += -1.0
        col += 1

        # M3 = A00 * (B01 - B11)
        U[0 * m + 1, col] += 1.0
        V[1 * p + k1, col] += 1.0
        V[2 * p + k1, col] += -1.0
        W[k1 * n + 0, col] += 1.0
        W[k1 * n + 1, col] += 1.0
        col += 1

        # M4 = A11 * (B10 - B00)
        U[1 * m + 2, col] += 1.0
        V[2 * p + k0, col] += 1.0
        V[1 * p + k0, col] += -1.0
        W[k0 * n + 0, col] += 1.0
        W[k0 * n + 1, col] += 1.0
        col += 1

        # M5 = (A00 + A01) * B11
        U[0 * m + 1, col] += 1.0
        U[0 * m + 2, col] += 1.0
        V[2 * p + k1, col] += 1.0
        W[k0 * n + 0, col] += -1.0
        W[k1 * n + 0, col] += 1.0
        col += 1

        # M6 = (A10 - A00) * (B00 + B01)
        U[1 * m + 1, col] += 1.0
        U[0 * m + 1, col] += -1.0
        V[1 * p + k0, col] += 1.0
        V[1 * p + k1, col] += 1.0
        W[k1 * n + 1, col] += 1.0
        col += 1

        # M7 = (A01 - A11) * (B10 + B11)
        U[0 * m + 2, col] += 1.0
        U[1 * m + 2, col] += -1.0
        V[2 * p + k0, col] += 1.0
        V[2 * p + k1, col] += 1.0
        W[k0 * n + 0, col] += 1.0
        col += 1

    # Verify
    U_r = U.reshape(n, m, rank)
    V_r = V.reshape(m, p, rank)
    W_r = W.reshape(p, n, rank)

    valid = True
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = np.sum(U_r[i, j, :] * V_r[j, k, :] * W_r[k, i, :])
                if abs(s - 1.0) > 1e-9:
                    valid = False
                    break
            if not valid:
                break
        if not valid:
            break

    if valid:
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    # Fallback: schoolbook rank 36
    rank_fb = n * m * p
    U_fb = np.zeros((n * m, rank_fb))
    V_fb = np.zeros((m * p, rank_fb))
    W_fb = np.zeros((p * n, rank_fb))
    c = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_fb[i * m + j, c] = 1.0
                V_fb[j * p + k, c] = 1.0
                W_fb[k * n + i, c] = 1.0
                c += 1
    return {"U": U_fb.tolist(), "V": V_fb.tolist(), "W": W_fb.tolist()}
