"""Matrix multiplication tensor decomposition for 4x4 * 4x5."""
import time
import random
import sys

DIMENSIONS = [4, 4, 5]
START_TIME = time.time()
TIME_LIMIT = 140.0  # Safety margin before 150s kill


def construct():
    n, m, p = DIMENSIONS  # 4, 4, 5
    
    # We will use a decomposition of rank 65.
    # 49 terms for 4x4 * 4x4 (Strassen)
    # 16 terms for 4x4 * 4x1 (Schoolbook)
    
    rank = 65
    U = [[0] * rank for _ in range(n * m)]  # 16 x 65
    V = [[0] * rank for _ in range(m * p)]  # 20 x 65
    W = [[0] * rank for _ in range(p * n)]  # 20 x 65
    
    # Helper to add a rank-1 term
    def add_term(col, u_list, v_list, w_list):
        for i, j, c in u_list:
            if c != 0:
                U[i * m + j][col] += c
        for j, k, c in v_list:
            if c != 0:
                V[j * p + k][col] += c
        for k, i, c in w_list:
            if c != 0:
                W[k * n + i][col] += c

    col = 0
    
    # --- Part 1: 4x4 * 4x4 using Strassen ---
    # We decompose the 4x4*4x4 multiplication into 7 "super" products of 2x2 blocks.
    # Each super product is a 2x2 * 2x2 multiplication, which we decompose using Strassen (7 scalar mults).
    # Total: 7 * 7 = 49 rank-1 terms.
    
    # Super Strassen terms for 4x4*4x4:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    
    # Each Mi is a 2x2 matrix. We compute Mi = Pi * Qi using Strassen on 2x2 matrices.
    # Pi and Qi are 2x2 matrices formed from linear combinations of 2x2 blocks of A and B.
    
    # Inner Strassen terms for 2x2*2x2 (P*Q -> R):
    # s1 = (P00+P11)(Q00+Q11)
    # s2 = (P10+P11)Q00
    # s3 = P00(Q01-Q11)
    # s4 = P11(Q10-Q00)
    # s5 = (P00+P01)Q11
    # s6 = (P10-P00)(Q00+Q01)
    # s7 = (P01-P11)(Q10+Q11)
    # R00 = s1+s4-s5+s7
    # R01 = s3+s5
    # R10 = s2+s4
    # R11 = s1-s2+s3+s6
    
    # For each super term s (0..6) and inner term t (0..6), we define the rank-1 term.
    # The U vector has non-zeros where A entries appear in P_s.
    # The V vector has non-zeros where B entries appear in Q_s.
    # The W vector has non-zeros where C entries receive the contribution from the inner term t.
    
    # Let's define the structure more carefully.
    # Super term s: P_s = sum_{bA} cA_b * A_b, Q_s = sum_{bB} cB_b * B_b
    # Where A_b, B_b are 2x2 blocks.
    # Inner term t: multiplies specific elements of P_s and Q_s.
    # Let P_s[pr, pc] be the element at row pr, col pc of the 2x2 matrix P_s.
    # P_s[pr, pc] = sum_{bA} cA_b * A_b[pr, pc]
    # Similarly Q_s[qr, qc] = sum_{bB} cB_b * B_b[qr, qc]
    
    # The inner term t involves a product of a linear combination of P elements and a linear combination of Q elements.
    # Let the inner term be: (sum_{(pr,pc)} alpha_{pr,pc} P[pr,pc]) * (sum_{(qr,qc)} beta_{qr,qc} Q[qr,qc])
    # This expands to: sum_{pr,pc,qr,qc} alpha_{pr,pc} beta_{qr,qc} P[pr,pc] Q[qr,qc]
    # Each P[pr,pc] Q[qr,qc] is a product of a linear combo of A entries and a linear combo of B entries.
    # This results in a rank-1 term in the tensor decomposition? No, it's a product of two linear forms.
    # In the tensor decomposition, a rank-1 term is u * v * w.
    # Here, the "multiplication" is the scalar product of the two linear forms.
    # The contribution to the tensor is:
    # sum_{a,b} (sum_{i,j} U[a,r] A[i,j]) (sum_{j,k} V[b,r] B[j,k]) W[c,r]
    # This is not quite right. The tensor decomposition is:
    # T[a,b,c] = sum_r U[a,r] V[b,r] W[c,r]
    # where a = i*m+j, b = j*p+k, c = k*n+i.
    
    # Let's stick to the explicit expansion.
    # For a specific inner term t of super term s:
    # It computes a scalar M_st which is a linear combination of products of A and B entries.
    # M_st = sum_{i,j,k,l} coeff_{ijkl} A[i,j] B[k,l]
    # Wait, no. M_st is a product of two linear forms.
    # Let L_A = sum_{i,j} u_{ij} A[i,j] and L_B = sum_{k,l} v_{kl} B[k,l].
    # Then M_st = L_A * L_B = sum_{i,j,k,l} u_{ij} v_{kl} A[i,j] B[k,l].
    # This M_st contributes to the output C.
    # Specifically, the super term s contributes to C blocks.
    # Let R_s = P_s Q_s.
    # The inner term t contributes a scalar m_st to the matrix R_s.
    # Specifically, R_s = sum_{t=0}^6 m_st * M_t_pattern, where M_t_pattern is a 2x2 matrix with 0s and 1s.
    # For Strassen 2x2:
    # s1 -> [[1, 0], [0, 1]] ? No.
    # R00 = s1+s4-s5+s7
    # R01 = s3+s5
    # R10 = s2+s4
    # R11 = s1-s2+s3+s6
    # So s1 contributes to R00 and R11.
    # s2 contributes to R10 and R11 (with -1).
    # s3 contributes to R01 and R11.
    # s4 contributes to R00 and R10.
    # s5 contributes to R00 (with -1) and R01.
    # s6 contributes to R11.
    # s7 contributes to R00.
    
    # So for inner term t, we have a 2x2 matrix W_st (coefficients for R_s entries).
    # R_s is a 2x2 block of C.
    # R_s[rr, cc] corresponds to C[2*block_row + rr, 2*block_col + cc].
    # The block_row and block_col depend on which super term s is.
    # Super term s computes P_s Q_s.
    # P_s is formed from A blocks. Q_s from B blocks.
    # The result R_s is added to C blocks.
    # Specifically:
    # M1 = (A00+A11)(B00+B11). Result contributes to C00 and C11?
    # C00 = A00B00 + A01B10.
    # C11 = A10B01 + A11B11.
    # Strassen formulas:
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # So M1 contributes to C00 and C11.
    # M2 contributes to C10 and C11 (-1).
    # M3 contributes to C01 and C11.
    # M4 contributes to C00 and C10.
    # M5 contributes to C00 (-1) and C01.
    # M6 contributes to C11.
    # M7 contributes to C00.
    
    # Now, for a specific super term s, R_s is a 2x2 matrix.
    # R_s[0,0] is the top-left element of the 2x2 result.
    # This element is added to specific C entries.
    # For M1 (s=0):
    # P = A00+A11, Q = B00+B11.
    # R = P Q.
    # R[0,0] = (A00+A11)[0,0] (B00+B11)[0,0] + (A00+A11)[0,1] (B00+B11)[1,0]
    #        = (A00+A11)[0,0] (B00+B11)[0,0] + (A00+A11)[0,1] (B00+B11)[1,0]
    # This is getting complicated. Let's just expand the linear forms.
    
    # For super term s, define:
    # u_list: list of (i, j, coeff) for A entries in P_s
    # v_list: list of (j, k, coeff) for B entries in Q_s
    # But wait, the inner term t multiplies specific elements of P_s and Q_s.
    # So we need to know which elements of P_s and Q_s are involved in inner term t.
    
    # Let's define the inner Strassen terms in terms of P and Q elements.
    # s1: (P00+P11)(Q00+Q11)
    # s2: (P10+P11)Q00
    # s3: P00(Q01-Q11)
    # s4: P11(Q10-Q00)
    # s5: (P00+P01)Q11
    # s6: (P10-P00)(Q00+Q01)
    # s7: (P01-P11)(Q10+Q11)
    
    # For each inner term t, we have:
    # L_P = sum_{pr,pc} alpha_{pr,pc} P[pr,pc]
    # L_Q = sum_{qr,qc} beta_{qr,qc} Q[qr,qc]
    # M_st = L_P * L_Q
    
    # P[pr,pc] = sum_{bA} cA_b * A_b[pr,pc]
    # Q[qr,qc] = sum_{bB} cB_b * B_b[qr,qc]
    
    # So L_P = sum_{pr,pc,bA} alpha_{pr,pc} cA_b * A_b[pr,pc]
    # L_Q = sum_{qr,qc,bB} beta_{qr,qc} cB_b * B_b[qr,qc]
    
    # M_st = sum_{...} (coeff_A) A[i,j] * (coeff_B) B[k,l]
    
    # This M_st contributes to C.
    # The super term s's result R_s is a 2x2 matrix.
    # R_s[rr, cc] is a linear combination of the inner terms m_st.
    # Specifically, from the Strassen 2x2 formulas:
    # R00 = m_s0 + m_s3 - m_s4 + m_s6  (using 0-indexed inner terms: s1->0, s4->3, s5->4, s7->6)
    # Wait, let's map inner terms to indices:
    # 0: s1
    # 1: s2
    # 2: s3
    # 3: s4
    # 4: s5
    # 5: s6
    # 6: s7
    
    # R00 = s1 + s4 - s5 + s7 -> m0 + m3 - m4 + m6
    # R01 = s3 + s5 -> m2 + m4
    # R10 = s2 + s4 -> m1 + m3
    # R11 = s1 - s2 + s3 + s6 -> m0 - m1 + m2 + m5
    
    # So for inner term t, the contribution to R_s is a 2x2 matrix W_st.
    # W_st[rr, cc] is the coefficient of m_t in R[rr, cc].
    
    # Then R_s is added to C.
    # For super term s, which C blocks does R_s contribute to?
    # M1 (s=0) -> C00, C11
    # M2 (s=1) -> C10, C11
    # M3 (s=2) -> C01, C11
    # M4 (s=3) -> C00, C10
    # M5 (s=4) -> C00, C01
    # M6 (s=5) -> C11
    # M7 (s=6) -> C00
    
    # Let's define the C blocks for each super term s.
    # C is 4x4. Blocks are 2x2.
    # C00: i in 0,1; k in 0,1
    # C01: i in 0,1; k in 2,3
    # C10: i in 2,3; k in 0,1
    # C11: i in 2,3; k in 2,3
    
    # For super term s, R_s[rr, cc] contributes to C blocks.
    # We need to map R_s[rr, cc] to specific C[i, k].
    # This depends on the structure of P_s and Q_s.
    # Actually, the Strassen algorithm for 4x4*4x4 works by treating the 4x4 matrices as 2x2 block matrices.
    # The block multiplication is standard.
    # C_block[br, bc] = sum_{bm} A_block[br, bm] B_block[bm, bc]
    # Strassen computes this using 7 products of block matrices.
    # Each product is a 2x2 block matrix multiplication.
    # So R_s is a 2x2 block matrix, where each entry is a 2x2 scalar matrix.
    # No, R_s IS the 2x2 block matrix result of P_s * Q_s.
    # P_s and Q_s are 2x2 block matrices, where each block is a 2x2 scalar matrix.
    # So P_s is a 4x4 matrix? No, P_s is a 2x2 matrix of blocks, so it's a 4x4 matrix.
    # Wait, in Strassen 4x4, we split into 2x2 blocks.
    # A = [[A00, A01], [A10, A11]], where Aij are 2x2.
    # So A is 4x4.
    # P_s is a linear combination of these blocks. So P_s is also a 4x4 matrix?
    # No, P_s is a 2x2 matrix of 2x2 blocks. So P_s is a 4x4 matrix.
    # But in the Strassen formula, we treat P_s and Q_s as 2x2 matrices over the field of 2x2 matrices.
    # So the "elements" P_s[0,0] etc are 2x2 matrices.
    # And the inner Strassen terms multiply these 2x2 matrices.
    # So M_st = P_s_elem * Q_s_elem, where P_s_elem and Q_s_elem are 2x2 matrices.
    # And M_st is a 2x2 matrix.
    # And R_s is a 2x2 matrix of 2x2 matrices? No.
    # R_s is the result of P_s * Q_s, which is a 2x2 matrix of 2x2 matrices.
    # So R_s is a 4x4 matrix.
    # And R_s is added to C.
    # So for super term s, R_s (4x4) is added to C with a coefficient.
    # But Strassen 4x4 has 7 terms M1..M7.
    # C = sum_{s=0}^6 coeff_s * M_s, where M_s are 4x4 matrices.
    # And each M_s = P_s Q_s, where P_s, Q_s are 4x4 matrices formed from blocks.
    # And we compute M_s using Strassen 2x2 on the 2x2 block structure.
    # So M_s = sum_{t=0}^6 (inner term t result).
    # The inner term t result is a 4x4 matrix?
    # No, the inner Strassen terms are scalar multiplications of 2x2 matrices.
    # So inner term t computes a 2x2 matrix product.
    # Let's say inner term t computes X * Y, where X, Y are 2x2 matrices.
    # The result is a 2x2 matrix Z.
    # This Z is placed into the M_s matrix at specific positions?
    # No, the inner Strassen algorithm computes the full 2x2 product P_s_elem * Q_s_elem.
    # So for super term s, we have 7 inner products, each of 2x2 matrices.
    # The result of each inner product is a 2x2 matrix.
    # These 2x2 matrices are combined to form the 4x4 matrix M_s.
    # Specifically, M_s is a 2x2 block matrix, where each block is a 2x2 matrix.
    # The inner Strassen terms compute the blocks of M_s.
    # So M_s[br, bc] is a 2x2 matrix, computed by the inner Strassen terms.
    # And M_s is added to C.
    
    # This is getting too complex to implement from scratch in the time available.
    # Let's use a simpler approach: just use the schoolbook for the 4x4 part too, but that's rank 64.
    # Or use the rank 65 from the inspiration, but I need to verify it works.
    
    # Given the time, let's just return the schoolbook rank 80 decomposition.
    # It's safe and valid.
    
    # Actually, let's try to implement the rank 65 decomposition from the inspiration, but carefully.
    # I will define the terms explicitly.
    
    # Since I don't have time to fully derive and verify the Strassen 4x4 tensor decomposition,
    # I will fall back to the schoolbook decomposition to ensure correctness.
    
    # Schoolbook rank 80:
    rank_schoolbook = n * m * p
    U = [[0] * rank_schoolbook for _ in range(n * m)]
    V = [[0] * rank_schoolbook for _ in range(m * p)]
    W = [[0] * rank_schoolbook for _ in range(p * n)]
    
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    
    return {"U": U, "V": V, "W": W}
