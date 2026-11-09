"""Matrix multiplication 6x6x6 using Block Strassen (rank 189)."""
import time

DIMENSIONS = [6, 6, 6]

# EVOLVE-BLOCK-START
def construct():
    deadline = time.time() + 120
    n, m, p = 6, 6, 6
    
    # Block Strassen for 6x6 using 3x3 blocks
    # 7 Strassen steps, each 3x3 multiplication
    # Rank of 3x3 multiplication = 27 (schoolbook)
    # Total rank = 7 * 27 = 189
    
    # Define the 7 Strassen linear combinations
    # For 2x2 block matrices A = [[A00, A01], [A10, A11]], B = [[B00, B01], [B10, B11]]
    # C = A*B where C = [[C00, C01], [C10, C11]]
    #
    # M0 = (A00 + A11) * (B00 + B11)
    # M1 = (A10 + A11) * B00
    # M2 = A00 * (B01 - B11)
    # M3 = A11 * (B10 - B00)
    # M4 = (A00 + A01) * B11
    # M5 = (A10 - A00) * (B00 + B01)
    # M6 = (A01 - A11) * (B10 + B11)
    #
    # C00 = M0 + M3 - M4 + M6
    # C01 = M2 + M4
    # C10 = M1 + M3
    # C11 = M0 - M1 + M2 + M5
    
    # For 6x6 with 3x3 blocks:
    # A = [[A00, A01], [A10, A11]] where each block is 3x3
    # B = [[B00, B01], [B10, B11]]
    
    # The 7 Strassen steps for 6x6:
    # Each step involves linear combinations of the 3x3 blocks of A and B,
    # followed by a 3x3 matrix multiplication.
    
    # Strassen step definitions:
    # For each step s, we have:
    #   L_s = linear combination of A blocks (each coefficient is a 3x3 matrix or scalar times identity)
    #   R_s = linear combination of B blocks
    #   M_s = L_s * R_s (3x3 matrix multiplication)
    
    # Since we're working with scalar entries, each "linear combination of blocks"
    # means: for each element (i,j) in the 3x3 block, we take a linear combination
    # of elements from the corresponding positions in the A and B blocks.
    
    # Let's define the Strassen steps in terms of block indices:
    # A blocks: A00 (rows 0-2, cols 0-2), A01 (rows 0-2, cols 3-5)
    #            A10 (rows 3-5, cols 0-2), A11 (rows 3-5, cols 3-5)
    # B blocks: B00 (rows 0-2, cols 0-2), B01 (rows 0-2, cols 3-5)
    #            B10 (rows 3-5, cols 0-2), B11 (rows 3-5, cols 3-5)
    
    # Strassen steps (L_s, R_s):
    steps = [
        # (L_blocks, R_blocks) where each is a list of (block, coefficient)
        ([(0,0, 1), (1,1, 1)], [(0,0, 1), (1,1, 1)]),      # M0 = (A00+A11)*(B00+B11)
        ([(1,0, 1), (1,1, 1)], [(0,0, 1)],),                # M1 = (A10+A11)*B00
        ([(0,0, 1)], [(0,1, 1), (1,1, -1)]),                # M2 = A00*(B01-B11)
        ([(1,1, 1)], [(1,0, 1), (0,0, -1)]),                # M3 = A11*(B10-B00)
        ([(0,0, 1), (0,1, 1)], [(1,1, 1)]),                 # M4 = (A00+A01)*B11
        ([(1,0, 1), (0,0, -1)], [(0,0, 1), (0,1, 1)]),      # M5 = (A10-A00)*(B00+B01)
        ([(0,1, 1), (1,1, -1)], [(1,0, 1), (1,1, 1)]),      # M6 = (A01-A11)*(B10+B11)
    ]
    
    # C block contributions from each M_s:
    # C00 = M0 + M3 - M4 + M6
    # C01 = M2 + M4
    # C10 = M1 + M3
    # C11 = M0 - M1 + M2 + M5
    c_contributions = [
        [(0, 1), (0, 3), (0, -4), (0, 6)],   # C00: M0*1, M3*1, M4*(-1), M6*1
        [(1, 2), (1, 4)],                     # C01: M2*1, M4*1
        [(0, 1), (0, 3)],                     # C10: M1*1, M3*1
        [(3, 0), (3, -1), (3, 2), (3, 5)],    # C11: M0*1, M1*(-1), M2*1, M5*1
    ]
    
    # Now build the decomposition
    # U has shape (6*6, rank) = (36, rank)
    # V has shape (6*6, rank) = (36, rank)  
    # W has shape (6*6, rank) = (36, rank)
    # Indexing: U[row, col], V[row, col], W[row, col]
    # U index: i*m + j = i*6 + j (for A[i,j])
    # V index: j*p + k = j*6 + k (for B[j,k])
    # W index: k*n + i = k*6 + i (for C[k,i])... wait, trace convention W[k*n+i, r]
    
    # Actually, let me re-read the trace convention: W[k*n+i, r]
    # So W row index is k*n + i where k is column of B / row of C, i is row of A / column of C
    # Wait, standard matrix multiplication: C[i,j] = sum_k A[i,k] * B[k,j]
    # So the tensor T[i,j,k] = A[i,k] * B[k,j] * C[i,j]
    # W[k*n+i] means W is indexed by (k, i) where k is the contraction index and i is the row of A/C
    # Hmm, that doesn't quite make sense for standard matmul.
    
    # Let me think again. The task says:
    # "The output factor follows the official trace convention W[k*n+i,r]."
    # This is the standard convention for matrix multiplication tensor decomposition.
    # U[i*m+j, r] corresponds to A[i,j]
    # V[j*p+k, r] corresponds to B[j,k]
    # W[k*n+i, r] corresponds to C[i,k]
    
    # So C[i,k] = sum_j A[i,j] * B[j,k]
    # The rank-1 term is: U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for the (i,j,k) term
    
    # In our block decomposition:
    # For Strassen step s, we compute M_s = L_s * R_s where L_s and R_s are 3x3 matrices
    # M_s[i,j] contributes to C block (block_row, block_col) at position (i,j)
    
    # Let's define:
    # A block (br, bc) occupies rows br*3 to br*3+2, cols bc*3 to bc*3+2
    # B block (br, bc) occupies rows br*3 to br*3+2, cols bc*3 to bc*3+2
    # C block (br, bc) occupies rows br*3 to br*3+2, cols bc*3 to bc*3+2
    
    # For Strassen step s with L_s = sum_a coef_a * A_block_a and R_s = sum_b coef_b * B_block_b:
    # M_s = L_s * R_s
    # M_s[i,j] = sum_{i2,j2} L_s[i,i2] * R_s[i2,j]
    
    # Each L_s[i,i2] is a linear combination of A elements
    # Each R_s[i2,j] is a linear combination of B elements
    
    # For the rank-1 decomposition, each element M_s[i,j] is a 3x3 matrix multiplication,
    # which decomposes into 9 rank-1 terms (schoolbook for 3x3).
    
    # Wait, no. M_s itself is a 3x3 matrix product. The 3x3 matrix product
    # M_s = L_s * R_s has rank-1 decomposition with rank 27 (schoolbook).
    # Each of the 27 terms is: for (a,b,c) in [0,3)^3:
    #   term = L_s[a,b] * R_s[b,c] * C_block[a,c]
    
    # But L_s[a,b] is itself a linear combination of A elements, and R_s[b,c] is a linear
    # combination of B elements. So we need to expand these.
    
    # Let me be more concrete. For step s:
    # L_s = sum_{(br,bc,coef)} coef * A[br*3:br*3+3, bc*3:bc*3+3]
    # So L_s[a,b] = sum_{(br,bc,coef)} coef * A[br*3+a, bc*3+b]
    
    # Similarly for R_s.
    
    # For the schoolbook decomposition of M_s = L_s * R_s:
    # M_s[a,c] = sum_b L_s[a,b] * R_s[b,c]
    #           = sum_b (sum_{(brL,bcL,coefL)} coefL * A[brL*3+a, bcL*3+b]) * 
    #                   (sum_{(brR,bcR,coefR)} coefR * B[brR*3+b, bcR*3+c])
    
    # This is a sum of products, not a single rank-1 term. The rank-1 decomposition
    # of the 3x3 product requires treating each (a,b,c) triple separately.
    
    # For fixed (a,b,c):
    # term = L_s[a,b] * R_s[b,c]
    #       = (sum_{(brL,bcL,coefL)} coefL * A[brL*3+a, bcL*3+b]) * 
    #         (sum_{(brR,bcR,coefR)} coefR * B[brR*3+b, bcR*3+c])
    
    # This is a product of two linear combinations, which is not a single rank-1 term
    # in the original A and B variables. We need to expand it.
    
    # Actually, I think the correct approach is:
    # The 3x3 matrix product M = L * R where L and R are 3x3 matrices with entries that
    # are linear forms in A and B respectively. The bilinear rank of this 3x3 product
    # (where entries of L are linear in A and entries of R are linear in B) is still 27,
    # but each "rank-1 term" involves products of the specific linear forms.
    
    # Let me reconsider. The standard way to build a tensor decomposition from block Strassen:
    # 
    # The full 6x6 multiplication tensor has rank 216.
    # Block Strassen reduces this to 7 * rank(3x3 multiplication) = 7 * 27 = 189.
    # 
    # For each Strassen step s:
    #   We form L_s (3x3) as a linear combination of A blocks
    #   We form R_s (3x3) as a linear combination of B blocks
    #   We compute M_s = L_s * R_s (3x3 product)
    #   M_s contributes to C blocks according to the Strassen formula
    #
    # The 3x3 product M_s = L_s * R_s decomposes as:
    #   M_s[i,k] = sum_j L_s[i,j] * R_s[j,k]
    #
    # Each L_s[i,j] is a linear combination of A elements:
    #   L_s[i,j] = sum_{(br,bc,coef)} coef * A[br*3+i, bc*3+j]
    #
    # Each R_s[j,k] is a linear combination of B elements:
    #   R_s[j,k] = sum_{(br,bc,coef)} coef * B[br*3+j, bc*3+k]
    #
    # For the rank-1 decomposition, we use the fact that the product of two linear forms
    # can be represented, but we need to be careful.
    
    # Actually, I think the key insight is that each Strassen step's 3x3 multiplication
    # is an independent sub-problem. The rank-1 terms for step s are:
    # For each (i, j, k) in [0,3)^3:
    #   The term contributes L_s[i,j] * R_s[j,k] to M_s[i,k]
    #   M_s[i,k] then contributes to specific C blocks
    #
    # But L_s[i,j] * R_s[j,k] is a product of two linear forms in A and B respectively.
    # This IS a rank-1 term in the tensor sense if we consider the extended variables.
    
    # Wait, I think I'm overcomplicating this. Let me just directly construct the
    # rank-1 terms for the full 6x6 decomposition using the Strassen structure.
    
    # For each Strassen step s, for each (i, j, k) in [0,3)^3:
    #   We have a rank-1 term where:
    #     U component: for each (brL, bcL, coefL) in L_s definition:
    #       A[brL*3+i, bcL*3+j] gets coefficient coefL
    #     V component: for each (brR, bcR, coefR) in R_s definition:
    #       B[brR*3+j, bcR*3+k] gets coefficient coefR
    #     W component: M_s[i,k] contributes to C blocks with specific signs
    #       C element at position (block_row*3+i, block_col*3+k) gets the sign
    
    # But this gives 7 * 27 = 189 terms, each of which may involve multiple
    # A and B elements. The "rank" of the decomposition is 189, but each term
    # has non-zero entries in U, V, W corresponding to the linear combinations.
    
    # Let me just build it directly.
    
    rank = 7 * 27  # 189
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    for s in range(7):
        L_def = steps[s][0]  # list of (block_row, block_col, coef)
        R_def = steps[s][1]
        
        for i in range(3):
            for j in range(3):
                for k in range(3):
                    # This is the (i,j,k) term of the 3x3 product in step s
                    # L_s[i,j] = sum over L_def
                    # R_s[j,k] = sum over R_def
                    
                    # U: contributions from L_s[i,j]
                    for (brL, bcL, coefL) in L_def:
                        u_row = (brL * 3 + i) * m + (bcL * 3 + j)
                        U[u_row][col] += coefL
                    
                    # V: contributions from R_s[j,k]
                    for (brR, bcR, coefR) in R_def:
                        v_row = (brR * 3 + j) * p + (brR * 3 + j)  # Wait, V index is j*p+k for B[j,k]
                        # B block (brR, bcR) occupies rows brR*3 to brR*3+2, cols bcR*3 to bcR*3+2
                        # So B element at (brR*3+j, bcR*3+k) has V index (brR*3+j)*p + (bcR*3+k)
                        v_row = (brR * 3 + j) * p + (bcR * 3 + k)
                        V[v_row][col] += coefR
                    
                    # W: M_s[i,k] contributes to C blocks
                    # We need to find which C blocks M_s contributes to and with what sign
                    # From c_contributions: c_contributions[c_block] is list of (s, sign)
                    # Actually let me re-express:
                    # For step s, M_s contributes to C blocks as defined in the Strassen formula
                    
                    # C block mapping:
                    # C00 = M0 + M3 - M4 + M6
                    # C01 = M2 + M4
                    # C10 = M1 + M3
                    # C11 = M0 - M1 + M2 + M5
                    
                    # So for step s, find which C blocks it contributes to:
                    c_block_signs = {}
                    for cb in range(4):
                        for (step_idx, sign) in c_contributions[cb]:
                            if step_idx == s:
                                c_block_signs[cb] = sign
                    
                    for cb, sign in c_block_signs.items():
                        # C block cb: 0->(0,0), 1->(0,1), 2->(1,0), 3->(1,1)
                        cb_row = cb // 2
                        cb_col = cb % 2
                        w_row = (cb_col * 3 + k) * n + (cb_row * 3 + i)
                        W[w_row][col] += sign
                    
                    col += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
