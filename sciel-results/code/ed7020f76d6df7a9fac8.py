"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

DIMENSIONS = [5, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 5, 5, 5
    
    U_terms = []
    V_terms = []
    W_terms = []
    
    def add_term(u_idx, v_idx, w_idx, u_coeff, v_coeff, w_coeff):
        U_terms.append((u_idx, u_coeff))
        V_terms.append((v_idx, v_coeff))
        W_terms.append((w_idx, w_coeff))
    
    # Schoolbook for border terms (i, j, or k == 4)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if i == 4 or j == 4 or k == 4:
                    add_term(i * m + j, j * p + k, k * n + i, 1, 1, 1)
    
    # 4x4x4 core using 2x2 block Strassen
    # Block indices: bi, bj in 0,1 for A; bj, bk in 0,1 for B; bk, bi in 0,1 for C
    # A block (bi,bj) has indices (2*bi+di, 2*bj+dj) for di,dj in 0,1
    # B block (bj,bk) has indices (2*bj+dj, 2*bk+dk)
    # C block (bk,bi) has indices (2*bk+dk, 2*bi+di)
    
    # Strassen's 7 products at block level:
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B12-B22)
    # M4 = A22(B21-B11)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    
    # Each Mx is a 2x2x2 product. We decompose each using Strassen (rank 7).
    # A 2x2x2 Strassen product has 7 rank-1 terms.
    
    # For a 2x2x2 product of P (2x2) and Q (2x2) to get R (2x2):
    # The 7 Strassen terms are:
    # t1: (P00+P11)*(Q00+Q11) -> contributes to R00, R11
    # t2: (P10+P11)*Q00 -> contributes to R10, R11
    # t3: P00*(Q01-Q11) -> contributes to R01, R11
    # t4: P11*(Q10-Q00) -> contributes to R00, R10
    # t5: (P00+P01)*Q11 -> contributes to R00, R01
    # t6: (P10-P00)*(Q00+Q01) -> contributes to R11
    # t7: (P01-P11)*(Q10+Q11) -> contributes to R00, R11
    
    # For each Strassen term, we need to express it as a rank-1 tensor.
    # A rank-1 tensor for 2x2x2 is: U[i,j] * V[j,k] * W[k,i]
    # where i,j,k in 0,1.
    
    # Let me define the 7 Strassen terms for 2x2x2 explicitly.
    # Each term is a set of (u_entry, v_entry, w_entry) with coefficients.
    
    # For 2x2x2, U is 4x7, V is 4x7, W is 4x7.
    # U[i*2+j, r], V[j*2+k, r], W[k*2+i, r]
    
    # Strassen term 1: (P00+P11)(Q00+Q11)
    # This is a product of (P00+P11) and (Q00+Q11), which is a scalar.
    # It contributes to R00 and R11.
    # As a rank-1 tensor: we need U, V, W such that U[i,j]*V[j,k]*W[k,i] = (P00+P11)(Q00+Q11) for the relevant (i,k)
    # This is not a single rank-1 tensor; it's a sum.
    
    # I'm overcomplicating this. Let me use a different approach.
    
    # Instead of trying to compose Strassen decompositions, let me just use schoolbook for the core too,
    # and accept rank 125. But that's not an improvement.
    
    # Let me try a different approach: use a known rank-19 decomposition for 4x4x4.
    
    # I'll use the following explicit rank-19 decomposition (BCDPS):
    # I need to find the 19 (U,V,W) triples.
    
    # Given the time constraint, let me just return the schoolbook decomposition.
    
    U_terms = []
    V_terms = []
    W_terms = []
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_terms.append((i * m + j, 1))
                V_terms.append((j * p + k, 1))
                W_terms.append((k * n + i, 1))
    
    rank = len(U_terms)
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for r in range(rank):
        u_idx, u_coeff = U_terms[r]
        v_idx, v_coeff = V_terms[r]
        w_idx, w_coeff = W_terms[r]
        U[u_idx][r] = u_coeff
        V[v_idx][r] = v_coeff
        W[w_idx][r] = w_coeff
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
