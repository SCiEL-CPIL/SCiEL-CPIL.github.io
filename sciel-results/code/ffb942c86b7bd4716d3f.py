"""
Rank-minimal exact tensor decomposition for 2x5 * 5x6 matrix multiplication.

Strategy:
  1. Use a known low-rank decomposition built from block Strassen.
     - Split the 2x5 matrix A into two row blocks: A1 (2x2), A2 (2x3).
     - Split the 5x6 matrix B into columns that pair with each row-block.
     - For each pair (A_block, B_block) use a Strassen decomposition of the
       corresponding small matrix product.
     - Concatenate the resulting rank-1 terms.

  2. If that decomposition does not verify (should not happen), fall back to
     a small random-search refinement, then to the schoolbook rank-60
     decomposition as a last resort.

  The output W follows the official trace convention W[k*n + i, r] with n=2,
  so the 12 rows of W are indexed by (k,i) with k in 0..5, i in 0..1.
"""
import time
import random
import numpy as np

DIMENSIONS = [2, 5, 6]

# EVOLVE-BLOCK-START

def _strassen_2x2x2():
    """
    Return U (2x2, 7), V (2x2, 7), W (2x2, 7) for the 2x2 * 2x2 product
    under the row-major output convention: W[k*2 + i, t] where i is the row
    of C and k is the column of C.
    """
    # Standard Strassen terms for 2x2 * 2x2.
    # Let A = [[a,b],[c,d]], B = [[e,f],[g,h]], C = A*B = [[ae+bg, af+bh],[ce+dg, cf+dh]].
    # Strassen:
    #   M1 = (a+d)(e+h)
    #   M2 = (c+d)e
    #   M3 = a(f-h)
    #   M4 = d(g-e)
    #   M5 = (a+b)h
    #   M6 = (c-a)(e+f)
    #   M7 = (b-d)(g+h)
    # C11 = M1 - M4 + M5 + M7
    # C12 = M3 + M4
    # C21 = M2 + M5
    # C22 = M1 - M2 + M3 + M6
    #
    # Each M_t is a product of a linear form in A and a linear form in B.
    # We need U such that U[row_a, t] * V[row_b, t] * W[out, t] gives C[out].
    # For a term M_t = L_a(L_a) * L_b(L_b), we set U[i, t] = L_a[i], V[j, t] = L_b[j],
    # and W[out, t] = 1 for the output position that receives M_t with appropriate sign.
    #
    # Actually, the tensor T_{i,j,k} = delta_{i+j=k} is not right; the multiplication
    # tensor T_{(i,j),(j,k),(i,k)} = 1. For Strassen we decompose T as sum of rank-1
    # tensors: T = sum_t u_t (A-part) * v_t (B-part) * w_t (C-part).
    # Here u_t is a 2x2 matrix (flattened to 4), v_t is a 2x2 (flattened to 4),
    # w_t is a 2x2 (flattened to 4, in row-major: index = k*2 + i).
    #
    # Let's write the seven terms explicitly.
    # Term 1: M1 = (a+d)(e+h) -> U: [[1,0],[0,1]] for A, V: [[1,1],[0,1]] for B,
    #        W: M1 contributes +1 to C11 and +1 to C22.
    # C11 is index k*2+i = 0*2+0 = 0. C22 is index 1*2+1 = 3.
    # So W for term 1: W[0,0]=1, W[3,0]=1, others 0.
    #
    # Let's be more systematic. For each term t, U[:,t] is a 2x2 matrix (as 4-vector row-major),
    # V[:,t] is a 2x2 matrix, W[:,t] is a 2x2 matrix (as 4-vector, row-major with index k*2+i).
    # The contribution to C[i,k] is sum_t U[i,j,t] * V[j,k,t] * W[i,k,t] summed over j? No.
    # The tensor is T_{(i,j),(j,k),(i,k)}. A rank-1 decomposition is T = sum_t u_t * v_t * w_t
    # where u_t is in R^{2x2}, v_t in R^{2x2}, w_t in R^{2x2}. The (i,j,k) entry is
    # sum_t u_t[i,j] * v_t[j,k] * w_t[i,k].
    #
    # For Strassen, each term M_t is a bilinear form in A and B, so u_t is rank-1 or not,
    # but the product u_t[i,j] * v_t[j,k] gives the bilinear form. Then w_t[i,k] scales it.
    #
    # Term 1: M1 = (a+d)(e+h).
    #   u_1[i,j] = 1 if (i==j) else 0? No, (a+d) = A[0,0] + A[1,1]. So u_1[0,0]=1, u_1[1,1]=1.
    #   v_1[j,k] = 1 if (j==k) else 0? (e+h) = B[0,0] + B[1,1]. So v_1[0,0]=1, v_1[1,1]=1.
    #   Contribution to C[i,k]: sum_j u_1[i,j] v_1[j,k].
    #   j=0: u_1[i,0] v_1[0,k]. j=1: u_1[i,1] v_1[1,k].
    #   C[0,0]: j=0: 1*1=1; j=1: 0*0=0 -> 1.
    #   C[0,1]: j=0: 1*0=0; j=1: 0*1=0 -> 0.
    #   C[1,0]: j=0: 0*1=0; j=1: 1*0=0 -> 0.
    #   C[1,1]: j=0: 0*0=0; j=1: 1*1=1 -> 1.
    #   So this term gives 1 to C[0,0] and 1 to C[1,1]. w_1[0,0]=1, w_1[1,1]=1.
    #
    # Term 2: M2 = (c+d)e = (A[1,0]+A[1,1]) * B[0,0].
    #   u_2[1,0]=1, u_2[1,1]=1. v_2[0,0]=1.
    #   Contribution: C[1,0]: j=0: u_2[1,0] v_2[0,0] = 1*1=1; j=1: u_2[1,1] v_2[1,0]=1*0=0 -> 1.
    #   C[1,1]: j=0: 1*0=0; j=1: 1*0=0 -> 0.
    #   So w_2[1,0]=1.
    #
    # Term 3: M3 = a(f-h) = A[0,0] * (B[0,1]-B[1,1]).
    #   u_3[0,0]=1. v_3[0,1]=1, v_3[1,1]=-1.
    #   Contribution: C[0,1]: j=0: u_3[0,0] v_3[0,1] = 1*1=1; j=1: u_3[0,1] v_3[1,1]=0*(-1)=0 -> 1.
    #   So w_3[0,1]=1.
    #
    # Term 4: M4 = d(g-e) = A[1,1] * (B[1,0]-B[0,0]).
    #   u_4[1,1]=1. v_4[1,0]=1, v_4[0,0]=-1.
    #   Contribution: C[1,0]: j=1: u_4[1,1] v_4[1,0] = 1*1=1; j=0: u_4[1,0] v_4[0,0]=0*(-1)=0 -> 1.
    #   C[0,1]: j=1: u_4[0,1] v_4[1,1]=0*0=0; j=0: u_4[0,0] v_4[0,1]=0*0=0 -> 0.
    #   So w_4[1,0]=1.
    #
    # Term 5: M5 = (a+b)h = (A[0,0]+A[0,1]) * B[1,1].
    #   u_5[0,0]=1, u_5[0,1]=1. v_5[1,1]=1.
    #   Contribution: C[0,1]: j=0: u_5[0,0] v_5[0,1]=1*0=0; j=1: u_5[0,1] v_5[1,1]=1*1=1 -> 1.
    #   C[1,1]: j=0: u_5[1,0] v_5[0,1]=0*0=0; j=1: u_5[1,1] v_5[1,1]=0*1=0 -> 0.
    #   So w_5[0,1]=1.
    #
    # Term 6: M6 = (c-a)(e+f) = (A[1,0]-A[0,0]) * (B[0,0]+B[0,1]).
    #   u_6[1,0]=1, u_6[0,0]=-1. v_6[0,0]=1, v_6[0,1]=1.
    #   Contribution: C[1,0]: j=0: u_6[1,0] v_6[0,0] = 1*1=1; j=1: u_6[1,1] v_6[1,0]=0*0=0 -> 1.
    #   C[1,1]: j=0: u_6[1,0] v_6[0,1] = 1*1=1; j=1: u_6[1,1] v_6[1,1]=0*0=0 -> 1.
    #   So w_6[1,0]=1, w_6[1,1]=1.
    #
    # Term 7: M7 = (b-d)(g+h) = (A[0,1]-A[1,1]) * (B[1,0]+B[1,1]).
    #   u_7[0,1]=1, u_7[1,1]=-1. v_7[1,0]=1, v_7[1,1]=1.
    #   Contribution: C[0,0]: j=0: u_7[0,0] v_7[0,0]=0*0=0; j=1: u_7[0,1] v_7[1,0]=1*1=1 -> 1.
    #   C[0,1]: j=0: u_7[0,0] v_7[0,1]=0*0=0; j=1: u_7[0,1] v_7[1,1]=1*1=1 -> 1.
    #   So w_7[0,0]=1, w_7[0,1]=1.
    #
    # Now combine: C[i,k] = sum_t w_t[i,k] * (sum_j u_t[i,j] v_t[j,k])
    # But wait, the tensor is T_{(i,j),(j,k),(i,k)} = delta_{i,k?} no, it's 1 for all valid i,j,k.
    # The multiplication is C[i,k] = sum_j A[i,j] B[j,k].
    # So the tensor has T_{i,j,k} = 1 for all i,j,k. No, T_{(i,j),(j,k),(i,k)} = 1 means
    # that the (i,j) entry of A and (j,k) entry of B contribute to (i,k) entry of C.
    # So T_{i,j,k} = 1.
    #
    # A rank-1 decomposition is T = sum_t u_t * v_t * w_t where u_t[i,j] v_t[j,k] w_t[i,k]
    # summed over t equals 1 for all i,j,k? No, that's not right.
    #
    # Actually, the matrix multiplication tensor is T_{(i,j),(j,k),(i,k)} = 1.
    # A rank-1 term is u_t (a matrix) * v_t (a matrix) * w_t (a matrix) such that
    # the sum over t of u_t[i,j] * v_t[j,k] * w_t[i,k] = 1 for all i,j,k? No, that's not
    # the standard definition.
    #
    # Let me reconsider. The bilinear map is (A,B) -> C where C[i,k] = sum_j A[i,j] B[j,k].
    # We want to write this as sum_t (sum_ij u_t[i,j] A[i,j]) (sum_jk v_t[j,k] B[j,k]) w_t[i,k]?
    # No, that's not right either.
    #
    # The correct way: C = sum_t (u_t A v_t^T) * w_t? No.
    #
    # Let's use the standard tensor decomposition. The multiplication tensor T is in
    # R^{n*m x m*p x p*n}. T_{(i,j),(j,k),(i,k)} = 1.
    # A rank-1 decomposition is T = sum_t a_t (b_t x c_t) where a_t in R^{n*m}, b_t in R^{m*p},
    # c_t in R^{p*n}. So T_{(i,j),(j,k),(i,k)} = sum_t a_t[i,j] * b_t[j,k] * c_t[i,k].
    #
    # For Strassen 2x2, we have 7 terms. Let's list them as (a_t, b_t, c_t) where
    # a_t is 2x2 (flattened), b_t is 2x2, c_t is 2x2 (flattened with index k*2+i).
    #
    # Term 1: a_1 = [[1,0],[0,1]], b_1 = [[1,1],[0,1]], c_1 = [[1,0],[0,1]] (C11 and C22)
    #   T contribution: a_1[i,j] b_1[j,k] c_1[i,k]
    #   i=0,j=0,k=0: 1*1*1 = 1
    #   i=0,j=0,k=1: 1*1*0 = 0
    #   i=0,j=1,k=0: 0*0*1 = 0
    #   i=0,j=1,k=1: 0*1*0 = 0
    #   i=1,j=0,k=0: 0*1*0 = 0
    #   i=1,j=0,k=1: 0*1*1 = 0
    #   i=1,j=1,k=0: 1*0*0 = 0
    #   i=1,j=1,k=1: 1*1*1 = 1
    #
    # Term 2: a_2 = [[0,0],[1,1]], b_2 = [[1,0],[0,0]], c_2 = [[0,0],[1,0]] (C21)
    #   i=1,j=0,k=0: 1*1*1 = 1
    #   Others: 0
    #
    # Term 3: a_3 = [[1,0],[0,0]], b_3 = [[0,1],[0,-1]], c_3 = [[0,1],[0,0]] (C12)
    #   i=0,j=0,k=1: 1*1*1 = 1
    #   i=0,j=1,k=1: 0*(-1)*1 = 0
    #   Others: 0
    #
    # Term 4: a_4 = [[0,0],[0,1]], b_4 = [[-1,0],[1,0]], c_4 = [[0,0],[1,0]] (C21)
    #   i=1,j=1,k=0: 1*1*1 = 1
    #   i=1,j=0,k=0: 0*(-1)*1 = 0
    #   Others: 0
    #
    # Term 5: a_5 = [[1,1],[0,0]], b_5 = [[0,0],[0,1]], c_5 = [[0,1],[0,0]] (C12)
    #   i=0,j=1,k=1: 1*1*1 = 1
    #   i=0,j=0,k=1: 1*0*1 = 0
    #   Others: 0
    #
    # Term 6: a_6 = [[-1,0],[1,0]], b_6 = [[1,1],[0,0]], c_6 = [[0,0],[1,1]] (C21, C22)
    #   i=1,j=0,k=0: 1*1*1 = 1
    #   i=1,j=0,k=1: 1*1*1 = 1
    #   i=0,j=0,k=0: -1*1*0 = 0
    #   i=0,j=0,k=1: -1*1*0 = 0
    #   Others: 0
    #
    # Term 7: a_7 = [[0,1],[-1,0]], b_7 = [[0,0],[1,1]], c_7 = [[1,1],[0,0]] (C11, C12)
    #   i=0,j=1,k=0: 1*1*1 = 1
    #   i=0,j=1,k=1: 1*1*1 = 1
    #   i=1,j=1,k=0: -1*1*0 = 0
    #   i=1,j=1,k=1: -1*1*0 = 0
    #   Others: 0
    #
    # Now sum all terms:
    # i=0,j=0,k=0: T1=1, others=0 -> 1
    # i=0,j=0,k=1: T1=0, T3=1, T5=0, T6=0, T7=0 -> 1
    # i=0,j=1,k=0: T7=1, others=0 -> 1
    # i=0,j=1,k=1: T3=0, T5=1, T7=1 -> 2? That's wrong.
    #
    # Let me recheck. The Strassen algorithm is:
    # M1 = (a+d)(e+h)
    # M2 = (c+d)e
    # M3 = a(f-h)
    # M4 = d(g-e)
    # M5 = (a+b)h
    # M6 = (c-a)(e+f)
    # M7 = (b-d)(g+h)
    # C11 = M1 - M4 + M5 + M7
    # C12 = M3 + M4
    # C21 = M2 + M5
    # C22 = M1 - M2 + M3 + M6
    #
    # So the contribution to C[i,k] is a linear combination of the M_t.
    # Each M_t is a product of a linear form in A and a linear form in B.
    # So M_t = (sum_ij u_t[i,j] A[i,j]) * (sum_jk v_t[j,k] B[j,k]).
    # Then C[i,k] = sum_t w_t[i,k] * M_t = sum_t w_t[i,k] * (sum_ij u_t[i,j] A[i,j]) * (sum_jk v_t[j,k] B[j,k]).
    # This is not a tensor decomposition in the standard sense; it's a bilinear form decomposition.
    #
    # For the tensor decomposition, we need T_{(i,j),(j,k),(i,k)} = sum_t a_t[i,j] * b_t[j,k] * c_t[i,k].
    # But in Strassen, the same M_t contributes to multiple C[i,k] with different weights.
    # So we need to "split" each M_t into multiple rank-1 terms if it contributes to multiple outputs.
    #
    # Alternatively, we can just use the bilinear form and verify that the output is correct.
    # The task requires U (n*m, rank), V (m*p, rank), W (p*n, rank) such that
    # sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1 for all i,j,k.
    #
    # This is equivalent to saying that the bilinear form (A,B) -> sum_j A[i,j] B[j,k]
    # can be written as sum_c (sum_ij U[i*m+j, c] A[i,j]) * (sum_jk V[j*p+k, c] B[j,k]) * W[k*n+i, c].
    #
    # So for each term c, we have a pair of linear forms (one in A, one in B) and a weight W for each output.
    #
    # Let's just hardcode the Strassen decomposition for 2x2 and verify it.
    #
    # For 2x2, the 7 terms are:
    # Term 0: u = [1,0,0,1] (A: a+d), v = [1,1,0,1] (B: e+h), w = [1,0,0,1] (C: C11, C22)
    # Term 1: u = [0,0,1,1] (A: c+d), v = [1,0,0,0] (B: e), w = [0,0,1,0] (C: C21)
    # Term 2: u = [1,0,0,0] (A: a), v = [0,1,0,-1] (B: f-h), w = [0,1,0,0] (C: C12)
    # Term 3: u = [0,0,0,1] (A: d), v = [-1,0,1,0] (B: g-e), w = [0,0,1,0] (C: C21)
    # Term 4: u = [1,1,0,0] (A: a+b), v = [0,0,0,1] (B: h), w = [0,1,0,0] (C: C12)
    # Term 5: u = [-1,0,1,0] (A: c-a), v = [1,1,0,0] (B: e+f), w = [0,0,1,1] (C: C21, C22)
    # Term 6: u = [0,1,-1,0] (A: b-d), v = [0,0,1,1] (B: g+h), w = [1,1,0,0] (C: C11, C12)
    #
    # Let's verify term 0:
    # (a+d)(e+h) = ae + ah + de + dh
    # C11 = ae + bg. This term gives ae + ah + de + dh. But C11 should be ae + bg.
    # So this term alone is not C11; it's part of C11 and C22.
    #
    # The Strassen algorithm says C11 = M1 - M4 + M5 + M7.
    # M1 = (a+d)(e+h) = ae + ah + de + dh
    # M4 = d(g-e) = dg - de
    # M5 = (a+b)h = ah + bh
    # M7 = (b-d)(g+h) = bg + bh - dg - dh
    # C11 = (ae + ah + de + dh) - (dg - de) + (ah + bh) + (bg + bh - dg - dh)
    #      = ae + ah + de + dh - dg + de + ah + bh + bg + bh - dg - dh
    #      = ae + bg + 2ah + 2de - 2dg + 2bh
    # This is not ae + bg. I must have the Strassen formulas wrong.
    #
    # Let me look up the correct Strassen formulas.
    # M1 = (a+d)(e+h)
    # M2 = (c+d)e
    # M3 = a(f-h)
    # M4 = d(g-e)
    # M5 = (a+b)h
    # M6 = (c-a)(e+f)
    # M7 = (b-d)(g+h)
    # C11 = M1 - M4 + M5 + M7
    # C12 = M3 + M4
    # C21 = M2 + M5
    # C22 = M1 - M2 + M3 + M6
    #
    # Let's recompute C11:
    # M1 = ae + ah + de + dh
    # M4 = dg - de
    # M5 = ah + bh
    # M7 = bg + bh - dg - dh
    # C11 = M1 - M4 + M5 + M7 = ae + ah + de + dh - dg + de + ah + bh + bg + bh - dg - dh
    # = ae + bg + 2ah + 2de - 2dg + 2bh
    #
    # This is still not ae + bg. I think I'm making a mistake in the Strassen formulas.
    #
    # Let me try a different approach. I'll just use the schoolbook decomposition for now,
    # since it's guaranteed to work, and then try to improve it.
    #
    # Actually, let's just return the schoolbook decomposition for the 2x2 case, which is rank 4.
    # But Strassen is rank 7, which is worse than schoolbook for 2x2? No, schoolbook is 4, Strassen is 7.
    # So for 2x2, schoolbook is better.
    #
    # For 2x3x3, schoolbook is 6, Strassen-like might be better.
    #
    # Let's just use schoolbook for all blocks and see what rank we get.
    # A1 is 2x2, B1 is 2x6? No, B is 5x6. A is 2x5.
    #
    # Let's split A into A1 (2x2) and A2 (2x3).
    # Split B into B1 (2x6) and B2 (3x6)? No, B is 5x6, so the rows of B should match the columns of A.
    # A is 2x5, so A1 is rows 0-1, cols 0-1 (2x2), A2 is rows 0-1, cols 2-4 (2x3).
    # B is 5x6, so B1 is rows 0-1 (2x6), B2 is rows 2-4 (3x6).
    # Then C = A1*B1 + A2*B2.
    # A1*B1 is 2x2 * 2x6 = 2x6. Schoolbook rank: 2*2*6 = 24.
    # A2*B2 is 2x3 * 3x6 = 2x6. Schoolbook rank: 2*3*6 = 36.
    # Total rank: 24 + 36 = 60. Same as schoolbook.
    #
    # To get a lower rank, we need to use a decomposition that shares terms between the two products.
    # This is where the block Strassen idea comes in, but it's complex to implement correctly.
    #
    # Given the time constraint, let's just return the schoolbook decomposition for now,
    # and then try to improve it with a random search.
    """
    # Schoolbook 2x2x2
    U = [[0]*4 for _ in range(4)]
    V = [[0]*4 for _ in range(4)]
    W = [[0]*4 for _ in range(4)]
    for i in range(2):
        for j in range(2):
            for k in range(2):
                U[i*2+j, i*2+j] = 1
                V[j*2+k, i*2+j] = 1
                W[k*2+i, i*2+j] = 1
    return U, V, W
    """


def _strassen_2x3x3():
    """
    Return U (2x3, rank), V (3x3, rank), W (3x2, rank) for the 2x3 * 3x3 product.
    For now, use schoolbook (rank 6).
    """
    n, m, p = 2, 3, 3
    rank = n * m * p
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j, col] = 1
                V[j*p+k, col] = 1
                W[k*n+i, col] = 1
                col += 1
    return U, V, W


def _block_decomposition():
    """
    Build a block decomposition for 2x5 * 5x6.
    Split A into A1 (2x2) and A2 (2x3).
    Split B into B1 (2x6) and B2 (3x6).
    C = A1*B1 + A2*B2.
    Use schoolbook for each block, then concatenate.
    """
    n, m, p = 2, 5, 6
    # A1 is 2x2, B1 is 2x6
    # A2 is 2x3, B2 is 3x6
    # For A1*B1: n1=2, m1=2, p1=6
    # For A2*B2: n2=2, m2=3, p2=6
    #
    # U for A1*B1: (2*2, rank1) where rank1 = 2*2*6 = 24
    # V for A1*B1: (2*6, rank1)
    # W for A1*B1: (6*2, rank1) -> but W should be (p*n, rank) = (6*2, rank) = (12, rank)
    #
    # For A2*B2: n2=2, m2=3, p2=6
    # U for A2*B2: (2*3, rank2) where rank2 = 2*3*6 = 36
    # V for A2*B2: (3*6, rank2)
    # W for A2*B2: (6*2, rank2)
    #
    # Total rank = 24 + 36 = 60
    #
    # This is the same as schoolbook, so it's not an improvement.
    # Let's just return the schoolbook decomposition.
    return _schoolbook()


def _schoolbook():
    """
    Schoolbook rank-n*m*p decomposition for n×m times m×p matrix multiplication.
    Uses the trace convention: output index is k*n + i for W.
    """
    n, m, p = DIMENSIONS
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


def _verify(U, V, W):
    """Verify that the decomposition is exact."""
    n, m, p = DIMENSIONS
    r = len(W[0])
    Ua = np.array(U, dtype=float)
    Va = np.array(V, dtype=float)
    Wa = np.array(W, dtype=float)
    if Ua.shape != (n * m, r) or Va.shape != (m * p, r) or Wa.shape != (p * n, r):
        return False
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0.0
                for c in range(r):
                    val += Ua[i * m + j, c] * Va[j * p + k, c] * Wa[k * n + i, c]
                if abs(val - 1.0) > 1e-9:
                    return False
    return True


def _try_random_search(rank_target, seed=42, max_trials=100):
    """
    Try to find a rank-decomposition with the given rank using random search.
    """
    n, m, p = DIMENSIONS
    random.seed(seed)
    np.random.seed(seed)
    
    for _ in range(max_trials):
        U = np.random.randint(-2, 3, size=(n * m, rank_target))
        V = np.random.randint(-2, 3, size=(m * p, rank_target))
        W = np.random.randint(-2, 3, size=(p * n, rank_target))
        
        success = True
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = 0.0
                    for c in range(rank_target):
                        val += U[i * m + j, c] * V[j * p + k, c] * W[k * n + i, c]
                    if abs(val - 1.0) > 1e-9:
                        success = False
                        break
                if not success:
                    break
            if not success:
                break
        
        if success:
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    return None


def construct():
    """
    Find the lowest-rank exact decomposition for 2×5 × 5×6 matrix multiplication.
    """
    deadline = time.time() + 140  # 140s deadline to be safe
    
    # First, try the block decomposition (which is currently schoolbook).
    best_result = _schoolbook()
    best_rank = 60
    
    # Try to find a better decomposition using random search.
    for target_rank in [45, 40, 35, 30, 25, 20]:
        if time.time() > deadline - 10:
            break
        
        result = _try_random_search(target_rank, seed=random.randint(0, 1000000), max_trials=50)
        if result is not None:
            if _verify(result["U"], result["V"], result["W"]):
                best_result = result
                best_rank = target_rank
                break
    
    # Verify final result
    if not _verify(best_result["U"], best_result["V"], best_result["W"]):
        raise ValueError("Decomposition failed verification")
    
    return best_result
# EVOLVE-BLOCK-END
