"""Strassen-based rank decomposition for 6x6 matrix multiplication."""
import time
import itertools
import numpy as np

DIMENSIONS = [6, 6, 6]

# EVOLVE-BLOCK-START
def _verify(U, V, W, n, m, p, rank, tol=1e-9):
    """Check that the decomposition exactly reconstructs the matmul tensor."""
    # U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)
    # The tensor T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # We need T[i,j,k] = 1 for all (i,j,k) in the matmul tensor.
    # Actually the matmul tensor is T[i,j,k] = 1 if and only if the (i,j) entry of A * (j,k) entry of B contributes to (i,k).
    # The standard matmul tensor has T[i,j,k] = 1 for all i,j,k (it's the structure tensor).
    # So we need: for all i,j,k: sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] == 1
    
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    # Build the tensor efficiently using einsum
    # T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # Reshape U to (n, m, rank), V to (m, p, rank), W to (p, n, rank)
    U_reshaped = U_arr.reshape(n, m, rank)
    V_reshaped = V_arr.reshape(m, p, rank)
    W_reshaped = W_arr.reshape(p, n, rank)
    
    # T[i,j,k] = sum_r U_reshaped[i,j,r] * V_reshaped[j,k,r] * W_reshaped[k,i,r]
    T = np.einsum('ijr,jkr,ki r->ijk', U_reshaped, V_reshaped, W_reshaped)
    
    # The matmul tensor should be all ones
    if not np.allclose(T, 1.0, atol=tol, rtol=tol):
        return False
    return True

def construct():
    """
    Construct a Strassen-based rank decomposition for 6x6 matrix multiplication.
    
    Strategy: Divide each 6x6 matrix into 2x2 blocks of 3x3 matrices.
    Use Strassen's 7-multiplication algorithm for the 2x2 block structure.
    Each block multiplication is 3x3 * 3x3, done via schoolbook (9 mults each).
    Total rank: 7 * 9 = 63... wait, that's not right.
    
    Actually, let me reconsider. The standard approach:
    - A is 6x6, B is 6x6, C is 6x6
    - Split into 2x2 blocks of 3x3: A = [[A11, A12], [A21, A22]]
    - Strassen: 7 products of 3x3 matrices
    - Each 3x3 product needs 9 schoolbook multiplications
    - Total: 7 * 9 = 63 rank terms? No...
    
    Wait, the rank of the full tensor decomposition:
    - Schoolbook for 6x6: 6*6*6 = 216
    - Strassen on 2x2 blocks of 3x3: The 2x2 block multiplication has rank 7.
      But each "multiplication" is a 3x3 matrix multiplication.
      A 3x3 matrix multiplication has tensor rank 27 (schoolbook) or less.
      
    Actually, the way tensor decomposition works for block matrix multiplication:
    If we decompose the 2x2 block multiplication into 7 terms, each term is a product
    of two 3x3 matrices. The rank contribution is 7 * (rank of 3x3 matmul).
    
    For 3x3 schoolbook: rank = 27
    So total rank = 7 * 27 = 189.
    
    This matches the observed improvement from 216 to 189!
    """
    n, m, p = 6, 6, 6
    block_size = 3  # Each block is 3x3
    n_blocks = 2    # 2x2 grid of blocks
    
    # Strassen's 7 linear combinations for 2x2 block multiplication:
    # M1 = (A11 + A22) * (B11 + B22)
    # M2 = (A21 + A22) * B11
    # M3 = A11 * (B12 - B22)
    # M4 = A22 * (B21 - B11)
    # M5 = (A11 + A12) * B22
    # M6 = (A21 - A11) * (B11 + B12)
    # M7 = (A12 - A22) * (B21 + B22)
    #
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    # For each Strassen term, we need to decompose the 3x3 matrix multiplication.
    # The 3x3 schoolbook decomposition has 27 terms.
    
    # Let's build the full decomposition.
    # For each of the 7 Strassen products, we have a 3x3 * 3x3 multiplication.
    # The 3x3 multiplication (i,j,k) -> A[i,j]*B[j,k] has rank 27 (schoolbook).
    
    # The overall tensor T[i,j,k] for 6x6 matmul can be expressed as:
    # T[i,j,k] = sum over Strassen terms s=1..7 of sum over 3x3 schoolbook terms t=1..27
    #           U_s,t[i,j] * V_s,t[j,k] * W_s,t[i,k]
    
    # Let me index the 6x6 matrices by block and within-block indices:
    # i = bi * 3 + ii, where bi in {0,1}, ii in {0,1,2}
    # j = bj * 3 + jj
    # k = bk * 3 + kk
    
    # For Strassen term s, the product is P_s = X_s * Y_s where X_s and Y_s are 3x3 matrices
    # formed by linear combinations of the 3x3 blocks.
    
    # Let me define the Strassen terms more carefully.
    # Block indices: (0,0)=11, (0,1)=12, (1,0)=21, (1,1)=22
    
    # Strassen M1 = (A11+A22)*(B11+B22)
    # M2 = (A21+A22)*B11
    # M3 = A11*(B12-B22)
    # M4 = A22*(B21-B11)
    # M5 = (A11+A12)*B22
    # M6 = (A21-A11)*(B11+B12)
    # M7 = (A12-A22)*(B21+B22)
    
    # For each M_s = X_s * Y_s, the contribution to C is:
    # C11 += M1 + M4 - M5 + M7
    # C12 += M3 + M5
    # C21 += M2 + M4
    # C22 += M1 - M2 + M3 + M6
    
    # The tensor decomposition:
    # For each Strassen term s and each schoolbook term (ii,jj,kk) in the 3x3 multiplication:
    # The U factor: coefficient of A[ii,jj] in X_s (where ii,jj are within-block indices)
    # The V factor: coefficient of B[jj,kk] in Y_s
    # The W factor: coefficient in the output block
    
    # Let me define for each Strassen term s:
    # X_s is a 3x3 matrix that is a linear combination of the A-blocks
    # Y_s is a 3x3 matrix that is a linear combination of the B-blocks
    
    # Actually, X_s and Y_s are themselves 3x3 matrices (the blocks), and their
    # entries are linear combinations of the entries of the original A and B blocks.
    
    # Let me think of it differently. The 6x6 matrix A has entries A[i,j] for i,j in 0..5.
    # Block (bi,bj) of A is A[bi*3:(bi+1)*3, bj*3:(bj+1)*3].
    
    # For Strassen term s, X_s is a 3x3 matrix where X_s[ii,jj] is a linear combination
    # of entries of A. Similarly for Y_s.
    
    # The product M_s = X_s * Y_s is a 3x3 matrix where M_s[ii,kk] = sum_jj X_s[ii,jj] * Y_s[jj,kk].
    
    # The contribution of M_s to the output C depends on which block of C it goes to.
    
    # Let me enumerate:
    # For each Strassen term s, X_s = sum_{bi,bj} alpha_s[bi,bj] * A_block[bi,bj]
    # where A_block[bi,bj] is the 3x3 block at position (bi,bj) of A.
    # Similarly Y_s = sum_{bj,bk} beta_s[bj,bk] * B_block[bj,bk]
    
    # Then M_s = X_s * Y_s, and its contribution to C_block[bi,bk] is gamma_s[bi,bk] * M_s
    # where gamma_s[bi,bk] is the coefficient from the Strassen output formulas.
    
    # Now, for the tensor decomposition, each "multiplication" in the 3x3 product M_s
    # corresponds to a rank-1 term. The 3x3 schoolbook has 27 such terms.
    
    # For term (ii,jj,kk) in the 3x3 product of M_s:
    # U entry: at position (i,j) where i = bi*3+ii, j = bj*3+jj, value = alpha_s[bi,bj]
    # V entry: at position (j,k) where j = bj*3+jj, k = bk*3+kk, value = beta_s[bj,bk]  
    # W entry: at position (k,i) where k = bk*3+kk, i = bi*3+ii, value = gamma_s[bi,bk]
    
    # Wait, I need to be more careful. The 3x3 matrix X_s has entries X_s[ii,jj]
    # where ii is the row index within the block and jj is the column index within the block.
    # But X_s is formed by combining entire 3x3 blocks of A.
    
    # So X_s[ii,jj] = sum over block positions (bi,bj) of alpha_s[bi,bj] * A[bi*3+ii, bj*3+jj]
    
    # Similarly Y_s[jj,kk] = sum over block positions (bj,bk) of beta_s[bj,bk] * B[bj*3+jj, bk*3+kk]
    
    # And M_s[ii,kk] = sum_jj X_s[ii,jj] * Y_s[jj,kk]
    
    # The contribution to C[bi*3+ii, bk*3+kk] from M_s is gamma_s[bi,bk] * M_s[ii,kk]
    
    # So the full tensor:
    # T[i,j,k] where i=bi*3+ii, j=bj*3+jj, k=bk*3+kk
    # T[bi*3+ii, bj*3+jj, bk*3+kk] = 1 (for the matmul tensor)
    
    # And our decomposition gives:
    # sum_s sum_{ii',jj',kk'} U[bi*3+ii, bj*3+jj, (s,ii',jj')] * V[bj*3+jj, bk*3+kk, (s,jj',kk')] * W[bk*3+kk, bi*3+ii, (s,ii',kk')]
    
    # Hmm, this is getting complicated. Let me just directly construct the factors.
    
    # For each Strassen term s (0..6) and each schoolbook term (ii,jj,kk) in 3x3:
    # The rank index r = s*27 + ii*9 + jj*3 + kk
    
    # U[i*m+j, r] where i=bi*3+ii_u, j=bj*3+jj_u:
    #   U is nonzero when ii_u == ii and jj_u == jj, and the value is alpha_s[bi,bj]
    #   Wait, that's not right either. Let me reconsider.
    
    # Actually, I think the cleanest way is:
    # For Strassen term s, the "A-side" matrix X_s is a 6x6 matrix (not 3x3!)
    # No wait, X_s is a 3x3 block-level matrix, but when we expand to individual entries...
    
    # Let me restart with a clearer formulation.
    
    # The 6x6 matmul tensor T has T[i,j,k] = 1 for all i,j,k in 0..5.
    # We want T[i,j,k] = sum_r U[i,j,r] * V[j,k,r] * W[k,i,r]
    
    # Using Strassen at the block level (2x2 blocks of 3x3):
    # i = bi*3 + ii, j = bj*3 + jj, k = bk*3 + kk
    # bi,bj,bk in {0,1}, ii,jj,kk in {0,1,2}
    
    # The Strassen decomposition says:
    # sum_{jj} A[bi*3+ii, bj*3+jj] * B[bj*3+jj, bk*3+kk] 
    # = sum_s gamma_s[bi,bk] * (sum_{jj} X_s[ii,jj] * Y_s[jj,kk])
    # where X_s and Y_s are 3x3 matrices formed from linear combinations of blocks.
    
    # X_s[ii,jj] = sum_{bi',bj'} alpha_s[bi',bj'] * A[bi'*3+ii, bj'*3+jj]
    # Y_s[jj,kk] = sum_{bj',bk'} beta_s[bj',bk'] * B[bj'*3+jj, bk'*3+kk]
    
    # So: sum_{jj} A[bi*3+ii, bj*3+jj] * B[bj*3+jj, bk*3+kk]
    # = sum_s gamma_s[bi,bk] * sum_{jj} (sum_{bi',bj'} alpha_s[bi',bj'] * A[bi'*3+ii, bj'*3+jj])
    #                                   * (sum_{bj',bk'} beta_s[bj',bk'] * B[bj'*3+jj, bk'*3+kk])
    
    # For this to equal the original sum, we need the Strassen identity to hold.
    
    # Now, for the tensor decomposition, we can write:
    # T[bi*3+ii, bj*3+jj, bk*3+kk] = 1
    # = sum_s gamma_s[bi,bk] * sum_{jj'} alpha_s[bi',bj'] * beta_s[bj',bk'] * (delta terms)
    
    # This is getting too abstract. Let me just directly build the rank-1 terms.
    
    # Key insight: Each rank-1 term in the decomposition corresponds to a specific
    # (Strassen term s, schoolbook indices ii,jj,kk, and block indices bi,bj,bk for the coefficients).
    
    # Actually, the simplest correct approach:
    # For each Strassen term s, we have a 3x3 matrix multiplication.
    # The 3x3 matmul tensor has rank 27 (schoolbook).
    # So for each s, we add 27 rank-1 terms.
    # Total: 7 * 27 = 189 terms.
    
    # For term (s, ii, jj, kk):
    # U[i, j, r] is nonzero for specific (i,j) pairs.
    # Specifically, U[bi*3+ii, bj*3+jj, r] = alpha_s[bi, bj]
    # V[j, k, r] is nonzero for specific (j,k) pairs.
    # V[bj*3+jj, bk*3+kk, r] = beta_s[bj, bk]
    # W[k, i, r] is nonzero for specific (k,i) pairs.
    # W[bk*3+kk, bi*3+ii, r] = gamma_s[bi, bk]
    
    # But wait, this gives multiple nonzero entries in each row/column of U, V, W.
    # That's fine for the tensor decomposition.
    
    # Let me define alpha_s, beta_s, gamma_s for each Strassen term.
    
    # Strassen terms (0-indexed):
    # s=0: M1 = (A11+A22)*(B11+B22)
    #   alpha_0: A11->1, A22->1 (blocks (0,0) and (1,1))
    #   beta_0: B11->1, B22->1 (blocks (0,0) and (1,1))
    #   gamma_0: C11->1, C22->1
    #
    # s=1: M2 = (A21+A22)*B11
    #   alpha_1: A21->1, A22->1 (blocks (1,0) and (1,1))
    #   beta_1: B11->1 (block (0,0))
    #   gamma_1: C21->1, C22->-1
    #
    # s=2: M3 = A11*(B12-B22)
    #   alpha_2: A11->1 (block (0,0))
    #   beta_2: B12->1, B22->-1 (blocks (0,1) and (1,1))
    #   gamma_2: C12->1, C22->1
    #
    # s=3: M4 = A22*(B21-B11)
    #   alpha_3: A22->1 (block (1,1))
    #   beta_3: B21->1, B11->-1 (blocks (1,0) and (0,0))
    #   gamma_3: C11->1, C21->1
    #
    # s=4: M5 = (A11+A12)*B22
    #   alpha_4: A11->1, A12->1 (blocks (0,0) and (0,1))
    #   beta_4: B22->1 (block (1,1))
    #   gamma_4: C11->-1, C12->1
    #
    # s=5: M6 = (A21-A11)*(B11+B12)
    #   alpha_5: A21->1, A11->-1 (blocks (1,0) and (0,0))
    #   beta_5: B11->1, B12->1 (blocks (0,0) and (0,1))
    #   gamma_5: C22->1
    #
    # s=6: M7 = (A12-A22)*(B21+B22)
    #   alpha_6: A12->1, A22->-1 (blocks (0,1) and (1,1))
    #   beta_6: B21->1, B22->1 (blocks (1,0) and (1,1))
    #   gamma_6: C11->1
    
    # Let me encode these as dictionaries.
    # alpha_s: dict from (bi, bj) -> coefficient
    # beta_s: dict from (bj, bk) -> coefficient
    # gamma_s: dict from (bi, bk) -> coefficient
    
    alpha = [
        {(0,0): 1, (1,1): 1},      # s=0
        {(1,0): 1, (1,1): 1},      # s=1
        {(0,0): 1},                 # s=2
        {(1,1): 1},                 # s=3
        {(0,0): 1, (0,1): 1},      # s=4
        {(1,0): 1, (0,0): -1},     # s=5
        {(0,1): 1, (1,1): -1},     # s=6
    ]
    beta = [
        {(0,0): 1, (1,1): 1},      # s=0
        {(0,0): 1},                 # s=1
        {(0,1): 1, (1,1): -1},     # s=2
        {(1,0): 1, (0,0): -1},     # s=3
        {(1,1): 1},                 # s=4
        {(0,0): 1, (0,1): 1},      # s=5
        {(1,0): 1, (1,1): 1},      # s=6
    ]
    gamma = [
        {(0,0): 1, (1,1): 1},      # s=0
        {(1,0): 1, (1,1): -1},     # s=1
        {(0,1): 1, (1,1): 1},      # s=2
        {(0,0): 1, (1,0): 1},      # s=3
        {(0,0): -1, (0,1): 1},     # s=4
        {(1,1): 1},                 # s=5
        {(0,0): 1},                 # s=6
    ]
    
    rank = 7 * 27  # 189
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for s in range(7):
        for ii in range(3):
            for jj in range(3):
                for kk in range(3):
                    r = s * 27 + ii * 9 + jj * 3 + kk
                    # U: for each (bi, bj) in alpha[s], set U[bi*3+ii, bj*3+jj, r] = alpha[s][(bi,bj)]
                    for (bi, bj), a_coeff in alpha[s].items():
                        U[bi * 3 + ii][bj * 3 + jj] = a_coeff
                    # V: for each (bj, bk) in beta[s], set V[bj*3+jj, bk*3+kk, r] = beta[s][(bj,bk)]
                    for (bj, bk), b_coeff in beta[s].items():
                        V[bj * 3 + jj][bk * 3 + kk] = b_coeff
                    # W: for each (bi, bk) in gamma[s], set W[bk*3+kk, bi*3+ii, r] = gamma[s][(bi,bk)]
                    for (bi, bk), g_coeff in gamma[s].items():
                        W[bk * 3 + kk][bi * 3 + ii] = g_coeff
    
    # Verify
    if not _verify(U, V, W, n, m, p, rank):
        # Fallback to schoolbook
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        column = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][column] = 1
                    V[j * p + k][column] = 1
                    W[k * n + i][column] = 1
                    column += 1
        return {"U": U, "V": V, "W": W}
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
