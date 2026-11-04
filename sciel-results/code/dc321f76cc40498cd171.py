import time
import numpy as np

DIMENSIONS = [4, 4, 5]
N, M, P = 4, 4, 5
NM = N * M
MP = M * P
PN = P * N

def _schoolbook_4x4x1():
    """Rank 16 decomposition for 4x4 @ 4x1 (column 4 of B)."""
    n, m, p = 4, 4, 1
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    c = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][c] = 1
                V[j * p + k][c] = 1
                W[k * n + i][c] = 1
                c += 1
    return U, V, W

def _strassen_4x4x4():
    """Rank 49 decomposition for 4x4 @ 4x4 (cols 0-3 of B)."""
    n = 4
    rank = 49
    U = np.zeros((n*n, rank))
    V = np.zeros((n*n, rank)) # First 4 cols of B
    W = np.zeros((n*n, rank)) # First 4 cols of C
    
    # Recursive Strassen construction
    # We define the 7 linear forms for 2x2 Strassen
    # u_coeffs[s][i,j], v_coeffs[s][i,j], w_coeffs[s][i,j]
    
    # 2x2 Strassen coefficients
    u2 = [
        [[1, 0], [0, 1]], # a+d
        [[0, 0], [1, 1]], # c+d
        [[1, 0], [0, 0]], # a
        [[0, 0], [0, 1]], # d
        [[1, 1], [0, 0]], # a+b
        [[0, 0], [1, -1]], # c-a
        [[0, 1], [0, -1]]  # b-d
    ]
    v2 = [
        [[1, 0], [0, 1]], # e+h
        [[1, 0], [0, 0]], # e
        [[0, 1], [0, -1]], # f-h
        [[0, 0], [1, -1]], # g-e
        [[0, 0], [0, 1]], # h
        [[1, 1], [0, 0]], # e+f
        [[0, 0], [1, 1]]  # g+h
    ]
    w2 = [
        [[1, 0], [0, 1]], # M1 contributes to C00, C11? 
        # C00 = M1 + M4 - M5 + M7
        # C01 = M3 + M5
        # C10 = M2 + M4
        # C11 = M1 - M2 + M3 + M6
        # So w_coeffs[out][s]
    ]
    # Correct w2 for 2x2:
    w2 = [
        [[1, 0], [0, 1]], # M1: C00+=1, C11+=1
        [[0, 0], [1, 0]], # M2: C10+=1
        [[0, 1], [0, 0]], # M3: C01+=1
        [[1, 0], [0, -1]], # M4: C00+=1, C11+=-1
        [[-1, 0], [0, 0]], # M5: C00+=-1, C01+=1? 
        # Let's re-verify 2x2 Strassen reconstruction
        # C00 = M1 + M4 - M5 + M7
        # C01 = M3 + M5
        # C10 = M2 + M4
        # C11 = M1 - M2 + M3 + M6
    ]
    w2 = [
        [[1, 0], [0, 1]], # s=0 (M1)
        [[0, 0], [1, 0]], # s=1 (M2)
        [[0, 1], [0, 0]], # s=2 (M3)
        [[1, 0], [0, -1]], # s=3 (M4)
        [[-1, 1], [0, 0]], # s=4 (M5)
        [[0, 0], [0, 1]], # s=5 (M6)
        [[1, 0], [0, 0]]  # s=6 (M7)
    ]
    # Check M5: C00 -= M5, C01 += M5. So [[-1, 1], [0, 0]] is correct for C00, C01.
    # Check M7: C00 += M7. So [[1, 0], [0, 0]] is correct.
    
    # 4x4 Strassen:
    # 7 outer terms. Each is a 2x2 matrix multiplication of linear forms.
    # We apply 2x2 Strassen to each outer term.
    
    # Outer coefficients for 4x4 (acting on 2x2 blocks)
    u4 = [
        [[1, 0], [0, 1]], # A00+A11
        [[0, 0], [1, 1]], # A10+A11
        [[1, 0], [0, 0]], # A00
        [[0, 0], [0, 1]], # A11
        [[1, 1], [0, 0]], # A00+A01
        [[0, 0], [1, -1]], # A10-A00
        [[0, 1], [0, -1]]  # A01-A11
    ]
    v4 = [
        [[1, 0], [0, 1]], # B00+B11
        [[1, 0], [0, 0]], # B00
        [[0, 1], [0, -1]], # B01-B11
        [[0, 0], [1, -1]], # B10-B00
        [[0, 0], [0, 1]], # B11
        [[1, 1], [0, 0]], # B00+B01
        [[0, 0], [1, 1]]  # B10+B11
    ]
    w4 = [
        [[1, 0], [0, 1]], # M1
        [[0, 0], [1, 0]], # M2
        [[0, 1], [0, 0]], # M3
        [[1, 0], [0, -1]], # M4
        [[-1, 1], [0, 0]], # M5
        [[0, 0], [0, 1]], # M6
        [[1, 0], [0, 0]]  # M7
    ]
    
    col = 0
    for k in range(7):
        # u4[k] is 2x2 matrix of coeffs for blocks.
        # Each block is 2x2.
        # So u4[k] defines a linear form on the 4x4 matrix A.
        # We need to expand this into a 16-length vector.
        U_vec = np.zeros(16)
        V_vec = np.zeros(16)
        
        # Map block (bi, bj) to indices
        # Block (0,0): 0,1,4,5
        # Block (0,1): 2,3,6,7
        # Block (1,0): 8,9,12,13
        # Block (1,1): 10,11,14,15
        block_indices = [
            [0, 1, 4, 5],
            [2, 3, 6, 7],
            [8, 9, 12, 13],
            [10, 11, 14, 15]
        ]
        
        for bi in range(2):
            for bj in range(2):
                coeff = u4[k][bi][bj]
                if coeff != 0:
                    for idx in block_indices[bi*2+bj]:
                        U_vec[idx] += coeff
                coeff_v = v4[k][bi][bj]
                if coeff_v != 0:
                    for idx in block_indices[bi*2+bj]:
                        V_vec[idx] += coeff_v
        
        # Now U_vec and V_vec are length 16.
        # We apply inner 2x2 Strassen to the "product" of these linear forms?
        # No, the outer Strassen term k is a product of two 2x2 matrices of linear forms.
        # Let L_k be the 2x2 matrix of linear forms defined by u4[k].
        # Let R_k be the 2x2 matrix of linear forms defined by v4[k].
        # We want to decompose L_k * R_k.
        # L_k * R_k is a 2x2 matrix of bilinear forms.
        # We use 2x2 Strassen to decompose this product into 7 rank-1 bilinear forms.
        
        # Extract the 4 linear forms for L_k (each is a 16-length vector? No, each is a 4-length vector on a 2x2 block)
        # Actually, u4[k][bi][bj] is a scalar coefficient for the block.
        # So L_k is a 2x2 matrix where entry (bi,bj) is the scalar coeff * Identity?
        # No, u4[k] is the coefficient for the block in the linear combination.
        # So L_k(bi, bj) is a linear form on A.
        # Specifically, L_k(bi, bj) = u4[k][bi][bj] * (sum of entries in block A_{bi,bj})?
        # No, u4[k] specifies which blocks to add.
        # So the linear form for block (bi,bj) of L_k is:
        #   if u4[k][bi][bj] != 0, it is u4[k][bi][bj] * (A_{bi,bj,00} + ...)?
        #   No, Strassen works on the matrix entries.
        #   u4[k] is a 2x2 matrix of scalars.
        #   The linear form is \sum_{bi,bj} u4[k][bi][bj] * A_{bi,bj}
        #   Where A_{bi,bj} is a 2x2 submatrix.
        #   This is a linear form on the 16 entries.
        
        # To apply inner Strassen, we need the 4 entries of the 2x2 matrix L_k.
        # Entry (bi, bj) of L_k is a linear form on A.
        # Let LF_L[bi][bj] be the 16-length vector for that linear form.
        
        LF_L = [[np.zeros(16) for _ in range(2)] for _ in range(2)]
        LF_R = [[np.zeros(16) for _ in range(2)] for _ in range(2)]
        
        for bi in range(2):
            for bj in range(2):
                c_l = u4[k][bi][bj]
                c_r = v4[k][bi][bj]
                if c_l != 0:
                    for idx in block_indices[bi*2+bj]:
                        LF_L[bi][bj][idx] += c_l
                if c_r != 0:
                    for idx in block_indices[bi*2+bj]:
                        LF_R[bi][bj][idx] += c_r
        
        # Now LF_L and LF_R are 2x2 matrices of 16-length vectors.
        # We apply 2x2 Strassen to multiply LF_L and LF_R.
        # This produces 7 terms.
        
        for s in range(7):
            # Inner Strassen coefficients u2[s], v2[s], w2[s]
            # u2[s] is 2x2 matrix of scalars.
            # The linear form for U in term s is:
            #   \sum_{bi,bj} u2[s][bi][bj] * LF_L[bi][bj]
            U_term = np.zeros(16)
            V_term = np.zeros(16)
            W_term = np.zeros(16)
            
            for bi in range(2):
                for bj in range(2):
                    c_u = u2[s][bi][bj]
                    c_v = v2[s][bi][bj]
                    if c_u != 0:
                        U_term += c_u * LF_L[bi][bj]
                    if c_v != 0:
                        V_term += c_v * LF_R[bi][bj]
            
            # W term:
            # The result of inner Strassen is a 2x2 matrix P.
            # P contributes to the outer result C via w4[k].
            # w4[k] is 2x2 matrix of scalars.
            # w2[s] is 2x2 matrix of scalars (contribution of inner term s to P).
            # The total contribution to C is:
            #   \sum_{ci,cj} w4[k][ci][cj] * w2[s][ci][cj] * P_{ci,cj}
            # But P_{ci,cj} is a bilinear form.
            # In the tensor decomposition, W is a linear form on C.
            # So W_term should be the linear form on C corresponding to the coefficient.
            
            # Wait, the tensor is \sum U V W.
            # U is linear form on A.
            # V is linear form on B.
            # W is linear form on C.
            
            # The bilinear form generated by U_term and V_term is:
            #   B(A,B) = U_term(A) * V_term(B)
            # This bilinear form corresponds to a specific entry in the 2x2 matrix P.
            # Actually, the inner Strassen term s produces a scalar product that contributes to multiple entries of P.
            # Specifically, P_{ci,cj} += w2[s][ci][cj] * (U_term(A) * V_term(B)).
            
            # So the contribution to C is:
            #   C_{outer_ci, outer_cj} += w4[k][outer_ci][outer_cj] * w2[s][inner_ci][inner_cj] * (U_term(A) * V_term(B))
            # This means W_term should be 1 at the indices corresponding to C_{outer_ci, outer_cj}
            # with coefficient w4[k][outer_ci][outer_cj] * w2[s][inner_ci][inner_cj].
            
            for ci in range(2):
                for cj in range(2):
                    coeff_w = w4[k][ci][cj] * w2[s][ci][cj]
                    if coeff_w != 0:
                        for idx in block_indices[ci*2+cj]:
                            W_term[idx] += coeff_w
            
            U[:, col] = U_term
            V[:, col] = V_term
            W[:, col] = W_term
            col += 1
            
    return U, V, W

def construct():
    deadline = time.time() + 110
    
    # Get Strassen 4x4x4 (cols 0-3)
    U_s, V_s, W_s = _strassen_4x4x4()
    rank_s = U_s.shape[1]
    
    # Get Schoolbook 4x4x1 (col 4)
    U_sb, V_sb, W_sb = _schoolbook_4x4x1()
    rank_sb = len(U_sb[0])
    
    total_rank = rank_s + rank_sb
    
    # Combine
    U = np.zeros((NM, total_rank))
    V = np.zeros((MP, total_rank))
    W = np.zeros((PN, total_rank))
    
    # Fill Strassen part (cols 0-48)
    U[:, :rank_s] = U_s
    # V_s is 16x49. V is 20x49. Rows 0-15 are B cols 0-3.
    V[:16, :rank_s] = V_s
    # W_s is 16x49. W is 20x49. Rows 0-15 are C cols 0-3.
    W[:16, :rank_s] = W_s
    
    # Fill Schoolbook part (cols 49-64)
    U[:, rank_s:] = np.array(U_sb)
    # V_sb is 4x16. V is 20x16.
    # V_sb row j corresponds to B[j, 4].
    # In V (20 rows), row j*5+4 corresponds to B[j, 4].
    for j in range(4):
        V[j*5+4, rank_s:] = V_sb[j]
        
    # W_sb is 4x16. W is 20x16.
    # W_sb row i corresponds to C[4, i]? No, W_sb row i is for C[4, i] in 1x4?
    # In _schoolbook_4x4x1, W has shape (1*4, 16) = (4, 16).
    # W[k*n+i] with k=0 (only col), i=0..3.
    # So W_sb[i] corresponds to C[0, i] in the 1x4 sub-problem.
    # In the full 4x5 problem, this is C[4, i].
    # In W (20 rows), row 4*4+i = 16+i corresponds to C[4, i].
    for i in range(4):
        W[16+i, rank_s:] = W_sb[i]
        
    # Convert to list of lists
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
