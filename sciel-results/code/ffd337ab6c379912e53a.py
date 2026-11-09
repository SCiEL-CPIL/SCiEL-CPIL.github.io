"""Strassen-based 4x4 matrix multiplication tensor decomposition (rank 49)."""
import time
import random
import numpy as np

DIMENSIONS = [4, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    """Return a rank-49 exact tensor decomposition for 4x4 matrix multiplication.
    
    Uses recursive Strassen: 4x4 = 2x2 blocks of 2x2, each 2x2 using 7 multiplications.
    Total rank = 7 * 7 = 49 (vs schoolbook 64).
    
    Convention: U[i*m+j, r] for A[i,j], V[j*p+k, r] for B[j,k], W[k*n+i, r] for C[i,j].
    """
    n, m, p = 4, 4, 4
    
    # Strassen 2x2: C = A*B where A,B,C are 2x2
    # A = [[a11,a12],[a21,a22]], B = [[b11,b12],[b21,b22]]
    # C = [[c11,c12],[c21,c22]]
    #
    # 7 products:
    # m1 = (a11+a22)(b11+b22)
    # m2 = (a21+a22)b11
    # m3 = a11(b12-b22)
    # m4 = a22(b21-b11)
    # m5 = (a11+a12)b22
    # m6 = (a21-a11)(b11+b12)
    # m7 = (a12-a22)(b21+b22)
    #
    # c11 = m1 - m4 + m5 + m6
    # c12 = m3 + m5
    # c21 = m2 + m4
    # c22 = m1 - m2 + m3 + m7
    
    # For each of the 7 Strassen multiplications, define:
    # u_vec[4]: coefficients for A entries [a11,a12,a21,a22]
    # v_vec[4]: coefficients for B entries [b11,b12,b21,b22]
    # w_vec[4]: coefficients for C entries [c11,c12,c21,c22]
    
    # A entry order: (0,0)=a11, (0,1)=a12, (1,0)=a21, (1,1)=a22
    # B entry order: (0,0)=b11, (0,1)=b12, (1,0)=b21, (1,1)=b22
    # C entry order: (0,0)=c11, (0,1)=c12, (1,0)=c21, (1,1)=c22
    
    strassen_defs = [
        # (u_vec, v_vec, w_vec)
        # m1 = (a11+a22)(b11+b22)
        # contribution to c11: +1, c22: +1
        ([1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1]),
        # m2 = (a21+a22)b11
        # contribution to c21: +1, c22: -1
        ([0, 0, 1, 1], [1, 0, 0, 0], [0, 0, 1, -1]),
        # m3 = a11(b12-b22)
        # contribution to c12: +1, c22: +1
        ([1, 0, 0, 0], [0, 1, 0, -1], [0, 1, 0, 1]),
        # m4 = a22(b21-b11)
        # contribution to c11: -1, c21: +1
        ([0, 0, 0, 1], [0, 0, 1, -1], [-1, 0, 1, 0]),
        # m5 = (a11+a12)b22
        # contribution to c11: +1, c12: +1
        ([1, 1, 0, 0], [0, 0, 0, 1], [1, 1, 0, 0]),
        # m6 = (a21-a11)(b11+b12)
        # contribution to c11: +1
        ([-1, 0, 1, 0], [1, 1, 0, 0], [1, 0, 0, 0]),
        # m7 = (a12-a22)(b21+b22)
        # contribution to c22: +1
        ([0, 1, 0, -1], [0, 0, 1, 1], [0, 0, 0, 1]),
    ]
    
    rank = 49
    U = [[0] * rank for _ in range(n * m)]  # 16 x 49
    V = [[0] * rank for _ in range(m * p)]  # 16 x 49
    W = [[0] * rank for _ in range(p * n)]  # 16 x 49
    
    # 4x4 divided into 2x2 blocks:
    # Block (bi, bj) where bi in {0,1}, bj in {0,1}
    # A block (bi, bj): rows 2*bi to 2*bi+1, cols 2*bj to 2*bj+1
    # B block (bj, bk): rows 2*bj to 2*bj+1, cols 2*bk to 2*bk+1
    # C block (bi, bk): rows 2*bi to 2*bi+1, cols 2*bk to 2*bk+1
    
    # For each outer block pair (bi, bj) for A, (bj, bk) for B, (bi, bk) for C:
    #   The C block (bi,bk) gets contribution from A block (bi,bj) * B block (bj,bk)
    #   For each Strassen multiplication s (0..6):
    #     column = (bi*2 + bj)*7 + s  (for A/B)
    #     column = (bi*2 + bk)*7 + s  (for W)
    #     Wait, we need a single column index for each (bi,bj,bk,s) tuple.
    #     
    # Actually, the decomposition is: for each (bi,bj,bk) in {0,1}^3, we do a 2x2 Strassen
    # multiply of A[bi,bj] * B[bj,bk] contributing to C[bi,bk].
    # Each such 2x2 multiply uses 7 rank-1 terms.
    # Total: 8 * 7 = 56? No wait...
    # 
    # Hmm, let me reconsider. The 4x4 matrix multiply:
    # C[i,j] = sum_k A[i,k] * B[k,j]
    # 
    # Divide into 2x2 blocks:
    # C[bi,bk] = sum_bj A[bi,bj] * B[bj,bk]
    # 
    # So for each (bi,bk), we sum over bj=0,1: A[bi,bj]*B[bj,bk]
    # Each product A[bi,bj]*B[bj,bk] is a 2x2 matrix multiply, done via Strassen with 7 terms.
    # 
    # So total terms = 2 (bi) * 2 (bk) * 2 (bj) * 7 (Strassen) = 56 terms.
    # But that gives rank 56, not 49.
    #
    # Wait, the recursive Strassen for 4x4:
    # 4x4 = 2x2 blocks, each 2x2 multiply uses 7 products.
    # But the 7 products at the top level are themselves 2x2 matrix multiplies.
    # So total scalar multiplies = 7 * 7 = 49.
    #
    # Let me redo this properly.
    # 
    # Top-level Strassen on 4x4:
    # A = [[A11, A12], [A21, A22]] where each Aij is 2x2
    # B = [[B11, B12], [B21, B22]]
    # C = [[C11, C12], [C21, C22]]
    #
    # 7 "multiplications" (each is 2x2 matmul):
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B12-B22)
    # M4 = A22(B21-B11)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    #
    # C11 = M1 - M4 + M5 + M6
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M7
    #
    # Each Mx is a 2x2 matmul, which we do with 7 scalar products via Strassen.
    # So total scalar products = 7 * 7 = 49.
    #
    # For tensor decomposition:
    # Each scalar product is a rank-1 term.
    # For top-level multiplication t (0..6) and inner Strassen multiplication s (0..6):
    #   column = t * 7 + s
    #
    # The U factor for column (t,s):
    #   For each A[i,j] (i,j in 0..3):
    #     Determine which top-level block Aij contains (i,j):
    #       bi = i // 2, bj = j // 2
    #       local_r = i % 2, local_c = j % 2
    #       local_idx = local_r * 2 + local_c  (0..3)
    #     The top-level coefficient for A block (bi,bj) in multiplication t:
    #       From strassen_defs[t][0][block_idx] where block_idx = bi*2+bj
    #     The inner Strassen coefficient for A entry local_idx in multiplication s:
    #       From strassen_defs[s][0][local_idx]
    #     U[i,j, col] = top_coeff * inner_coeff
    #
    # Similarly for V (B matrix) and W (C matrix).
    
    for t in range(7):
        for s in range(7):
            col = t * 7 + s
            # U: A[i,j]
            for i in range(4):
                for j in range(4):
                    bi = i // 2
                    bj = j // 2
                    local_r = i % 2
                    local_c = j % 2
                    local_idx = local_r * 2 + local_c
                    block_idx = bi * 2 + bj
                    U[i * m + j][col] = strassen_defs[t][0][block_idx] * strassen_defs[s][0][local_idx]
            
            # V: B[k,j] where k is row of B (0..3), j is col of B (0..3)
            # V index: k*p + j = 4k + j
            # B block (bk, bj') where bk = k//2, bj' = j//2
            # Wait, B is [k][j] in our convention? Let me re-examine.
            # 
            # C[i,j] = sum_k A[i,k] * B[k,j]
            # U[i*m+j, r] corresponds to A[i,j]
            # V[j*p+k, r] corresponds to B[j,k]  -- so V row index is j*p+k where j is row of B, k is col
            # W[k*n+i, r] corresponds to C[i,j] where... let me re-read.
            #
            # "W[k*n+i,r]" - the task says output factor follows W[k*n+i,r]
            # C[i,j] = sum_k A[i,k]*B[k,j]
            # So W should be indexed such that W[k*n+i] gives coefficient for C[i,j]... 
            # Actually, the standard convention for tensor decomposition of matmul:
            # C[i,j] = sum_k sum_r U[i,k,r] * V[k,j,r] * W[i,j,r]
            # But here the shapes are U(n*m, rank), V(m*p, rank), W(p*n, rank)
            # So U[i*m+j, r] = U[i,j,r], V[j*p+k, r] = V[j,k,r], W[k*n+i, r] = W[k,i,r]
            # And the relation is: C[i,j] = sum_k sum_r U[i,k,r]*V[k,j,r]*W[?, ?, r]
            #
            # Actually for the tensor: the multiplication tensor T has T[i,j,k] = 1 if the term A[i,j]*B[j,k] contributes to C[i,k].
            # In the flat indexing: U[i*m+j, r], V[j*p+k, r], W[k*n+i, r]
            # Then C[i,k] = sum_{j,r} U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
            #
            # So W[k*n+i, r] is the coefficient for C[i,k] (i=row, k=col of C).
            # W row index = k*n + i = 4k + i where k is col of C, i is row of C.
            
            for k in range(4):
                for j in range(4):
                    bk = k // 2
                    bj = j // 2
                    local_r = k % 2
                    local_c = j % 2
                    local_idx = local_r * 2 + local_c
                    block_idx = bk * 2 + bj
                    V[k * p + j][col] = strassen_defs[t][1][block_idx] * strassen_defs[s][1][local_idx]
            
            # W: W[k*n+i, r] for C[i,k] where i=row, k=col
            for i in range(4):
                for k in range(4):
                    bi = i // 2
                    bk = k // 2
                    local_r = i % 2
                    local_c = k % 2
                    local_idx = local_r * 2 + local_c
                    block_idx = bi * 2 + bk
                    W[k * n + i][col] = strassen_defs[t][2][block_idx] * strassen_defs[s][2][local_idx]
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
