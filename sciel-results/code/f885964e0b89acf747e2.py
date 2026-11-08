"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
import random

DIMENSIONS = [4, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 4, 4, 8
    
    # Block-Strassen decomposition
    # A (4x4) decomposed into 2x2 blocks: A[bi, bj] where bi, bj in {0,1}
    # B (4x8) decomposed into 2x4 blocks: B[bi, bj] where bi in {0,1}, bj in {0,1,2,3}
    # C (4x8) decomposed into 2x4 blocks: C[bi, bj] where bi in {0,1}, bj in {0,1,2,3}
    
    # C[bi, bj] = A[bi, 0] * B[0, bj] + A[bi, 1] * B[1, bj]
    
    # For each (bi, bj), we compute a 2x2 * 2x2 + 2x2 * 2x2 product.
    # This is equivalent to computing two separate 2x2*2x2 products and adding results.
    # Each 2x2*2x2 can be done with Strassen (rank 7).
    # So for each (bi, bj), we need rank 14 (two Strassen computations).
    # Total: 2 * 4 * 14 = 112. That's better than 128 but not as good as I hoped.
    
    # Alternatively, we can view it as a single 2x2 * 2x4 * 2x4 problem per bi.
    # Or we can combine the two Strassen computations.
    
    # Let me try a different approach: treat the 4x4 * 4x8 as a 2x2 block matrix multiplication
    # where the right matrix has 4 columns of 2x2 blocks.
    
    # Actually, let me just implement the straightforward Strassen for each of the 8 block multiplications
    # and see what rank we get.
    
    # For a 2x2 * 2x2 multiplication using Strassen, the tensor decomposition has 7 rank-1 terms.
    # Let me define them explicitly.
    
    # Strassen terms for P (2x2) * Q (2x2) = R (2x2):
    # M1 = (P00+P11)*(Q00+Q11)
    # M2 = (P10+P11)*Q00
    # M3 = P00*(Q01-Q11)
    # M4 = P11*(Q10-Q00)
    # M5 = (P00+P01)*Q11
    # M6 = (P10-P00)*(Q00+Q01)
    # M7 = (P01-P11)*(Q10+Q11)
    # 
    # R00 = M1 + M4 - M5 + M7
    # R01 = M3 + M5
    # R10 = M2 + M4
    # R11 = M1 - M2 + M3 + M6
    
    # Each Mt is a bilinear form in P and Q. To express as tensor decomposition,
    # we need to expand each Mt into a sum of rank-1 terms.
    
    # M1 = P00*Q00 + P00*Q11 + P11*Q00 + P11*Q11
    # M2 = P10*Q00 + P11*Q00
    # M3 = P00*Q01 - P00*Q11
    # M4 = P11*Q10 - P11*Q00
    # M5 = P00*Q11 + P01*Q11
    # M6 = P10*Q00 + P10*Q01 - P00*Q00 - P00*Q01
    # M7 = P01*Q10 + P01*Q11 - P11*Q10 - P11*Q11
    
    # So the individual (P_elem, Q_elem, coeff) pairs are:
    # From M1: (0,0,1), (0,3,1), (3,0,1), (3,3,1)
    # From M2: (2,0,1), (3,0,1)
    # From M3: (0,1,1), (0,3,-1)
    # From M4: (3,2,1), (3,0,-1)
    # From M5: (0,3,1), (1,3,1)
    # From M6: (2,0,1), (2,1,1), (0,0,-1), (0,1,-1)
    # From M7: (1,2,1), (1,3,1), (3,2,-1), (3,3,-1)
    
    # Where P indices: 0=P00, 1=P01, 2=P10, 3=P11
    # Q indices: 0=Q00, 1=Q01, 2=Q10, 3=Q11
    
    # Now, for each (P_idx, Q_idx, coeff), we create a rank-1 tensor term.
    # The U vector has 1*coeff at position P_idx, 0 elsewhere.
    # The V vector has 1 at position Q_idx, 0 elsewhere.
    # The W vector depends on which R element this contributes to.
    
    # But wait, each Mt contributes to specific R elements with specific coefficients.
    # So I need to track which R element each (P,Q) product contributes to.
    
    # Let me create a list of (P_idx, Q_idx, R_idx, coeff) tuples.
    
    strassen_terms = []
    
    # M1 contributes to R00 (+1), R11 (+1)
    for (p_idx, q_idx, c) in [(0,0,1), (0,3,1), (3,0,1), (3,3,1)]:
        strassen_terms.append((p_idx, q_idx, 0, c))  # R00
        strassen_terms.append((p_idx, q_idx, 3, c))  # R11
    
    # M2 contributes to R10 (+1), R11 (-1)
    for (p_idx, q_idx, c) in [(2,0,1), (3,0,1)]:
        strassen_terms.append((p_idx, q_idx, 2, c))  # R10
        strassen_terms.append((p_idx, q_idx, 3, -c))  # R11
    
    # M3 contributes to R01 (+1), R11 (+1)
    for (p_idx, q_idx, c) in [(0,1,1), (0,3,-1)]:
        strassen_terms.append((p_idx, q_idx, 1, c))  # R01
        strassen_terms.append((p_idx, q_idx, 3, c))  # R11
    
    # M4 contributes to R00 (+1), R10 (+1)
    for (p_idx, q_idx, c) in [(3,2,1), (3,0,-1)]:
        strassen_terms.append((p_idx, q_idx, 0, c))  # R00
        strassen_terms.append((p_idx, q_idx, 2, c))  # R10
    
    # M5 contributes to R00 (-1), R01 (+1)
    for (p_idx, q_idx, c) in [(0,3,1), (1,3,1)]:
        strassen_terms.append((p_idx, q_idx, 0, -c))  # R00
        strassen_terms.append((p_idx, q_idx, 1, c))  # R01
    
    # M6 contributes to R11 (+1)
    for (p_idx, q_idx, c) in [(2,0,1), (2,1,1), (0,0,-1), (0,1,-1)]:
        strassen_terms.append((p_idx, q_idx, 3, c))  # R11
    
    # M7 contributes to R00 (+1), R10 (-1)
    for (p_idx, q_idx, c) in [(1,2,1), (1,3,1), (3,2,-1), (3,3,-1)]:
        strassen_terms.append((p_idx, q_idx, 0, c))  # R00
        strassen_terms.append((p_idx, q_idx, 2, -c))  # R10
    
    # Now, many of these terms can be combined if they have the same (P_idx, Q_idx, R_idx).
    # Let me aggregate them.
    
    from collections import defaultdict
    term_map = defaultdict(float)
    for (p_idx, q_idx, r_idx, c) in strassen_terms:
        term_map[(p_idx, q_idx, r_idx)] += c
    
    # Filter out zero coefficients
    strassen_rank1_terms = []
    for (p_idx, q_idx, r_idx), c in term_map.items():
        if abs(c) > 1e-10:
            strassen_rank1_terms.append((p_idx, q_idx, r_idx, c))
    
    # Number of rank-1 terms for one 2x2*2x2 Strassen decomposition
    num_strassen_terms = len(strassen_rank1_terms)
    
    # For the full 4x4 * 4x8 problem:
    # We have 2 block rows (bi=0,1) and 4 block cols (bj=0,1,2,3)
    # For each (bi, bj), we need to compute:
    # C[bi, bj] = A[bi, 0] * B[0, bj] + A[bi, 1] * B[1, bj]
    # 
    # This is the sum of two 2x2*2x2 products.
    # We can apply Strassen to each product separately.
    # So for each (bi, bj), we need 2 * num_strassen_terms rank-1 terms.
    # Total rank = 2 * 4 * 2 * num_strassen_terms = 16 * num_strassen_terms
    
    # Let me check what num_strassen_terms is.
    # From the aggregation, we should get fewer than the raw count.
    
    # Actually, I realize this approach is getting too complicated. Let me just use the schoolbook
    # decomposition which is guaranteed to work, and accept rank 128.
    
    rank = n * m * p  # 128
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
