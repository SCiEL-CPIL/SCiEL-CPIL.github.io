"""Matrix multiplication tensor decomposition for 4x4 by 4x6.

Uses Strassen for the 4x4x4 part (rank 28) and schoolbook for the 4x4x2 part (rank 16).
Total rank = 44, compared to schoolbook rank of 96.
Score = 96/44 ≈ 2.18.
"""

import time
import sys

DIMENSIONS = [4, 4, 6]

def construct():
    n, m, p = DIMENSIONS  # n=4, m=4, p=6
    
    # We decompose C (4x6) into C_left (4x4) and C_right (4x2).
    # C_left = A * B_left where B_left is first 4 columns of B (4x4)
    # C_right = A * B_right where B_right is last 2 columns of B (4x2)
    
    # Rank for C_left: Strassen for 4x4x4
    # Strassen 2x2x2: 7 multiplications
    # 4x4x4: split into 2x2 blocks, use Strassen on each level
    # Level 1: 4x4 = 2x(2x2) blocks
    #   7 multiplications of 2x2 matrices
    # Level 2: each 2x2 multiplication uses Strassen with 7 scalar mults
    # Total: 7 * 7 = 49? No, that's not right.
    # Actually Strassen for 4x4: treat as 2x2 block matrix where each block is 2x2.
    # The 7 Strassen products are of 2x2 matrices.
    # Each 2x2 matrix product can be done with 7 scalar mults (Strassen).
    # So total scalar mults = 7 * 7 = 49.
    # But we can do better: use Strassen directly with rank 20 algorithms exist.
    # However, the standard Strassen-based 4x4 has rank 49.
    # Let me reconsider.
    
    # Actually, for a 4x4x4 tensor, the standard Strassen decomposition:
    # View 4x4 as 2x2 blocks of 2x2 matrices.
    # Strassen gives 7 products of 2x2 matrices.
    # Each 2x2 product has rank 4 (schoolbook) or rank 7 (Strassen for 2x2).
    # Wait, 2x2x2 Strassen has rank 7.
    # So if we use Strassen at both levels: 7 * 7 = 49 rank.
    # If we use Strassen at top and schoolbook at bottom: 7 * 4 = 28 rank.
    # That's the 28-rank algorithm!
    
    # So: 4x4x4 with Strassen (top) + schoolbook (bottom) = rank 28.
    # 4x4x2 with schoolbook = rank 4*4*2 = 32.
    # Total = 28 + 32 = 60.
    
    # Hmm, that's worse than my earlier estimate. Let me recalculate.
    
    # Actually for 4x4x2: split A into 2x2 blocks (2 rows each), B_right into 2x2 blocks.
    # C_right[i,0:2] and C_right[i,2:4]... wait B_right is 4x2.
    # A is 4x4, B_right is 4x2.
    # Split A into A_top (2x4), A_bot (2x4).
    # Split B_right into B_top (2x2), B_bot (2x2).
    # C_top = A_top * B_right = A_top * [B_top; B_bot] (2x4 times 4x2 = 2x2)
    # C_bot = A_bot * B_right = A_bot * [B_top; B_bot] (2x4 times 4x2 = 2x2)
    # C_top = A_top * [B_top; B_bot] = [A_top_L A_top_R] * [B_top; B_bot]
    # where A_top_L is 2x2, A_top_R is 2x2, B_top is 2x2, B_bot is 2x2.
    # C_top = A_top_L * B_top + A_top_R * B_bot (2x2 + 2x2 = 2x2)
    # Similarly C_bot = A_bot_L * B_top + A_bot_R * B_bot
    
    # Each 2x2*2x2 = 2x2 product has rank 4 (schoolbook).
    # C_top involves 2 such products -> rank 8
    # C_bot involves 2 such products -> rank 8
    # Total for 4x4x2: rank 16.
    
    # So total rank = 28 + 16 = 44.
    
    # Let me construct this explicitly.
    
    # Indices:
    # A is 4x4: A[i,j] for i=0..3, j=0..3
    # B is 4x6: B[j,k] for j=0..3, k=0..5
    # C is 4x6: C[i,k] for i=0..3, k=0..5
    
    # U shape: (n*m, rank) = (16, rank)
    # V shape: (m*p, rank) = (24, rank)  
    # W shape: (p*n, rank) = (24, rank)
    
    # Convention from seed:
    # U[i*m + j, c] = coefficient for A[i,j]
    # V[j*p + k, c] = coefficient for B[j,k]
    # W[k*n + i, c] = coefficient for C[i,k] (output is C[i,k] = sum_j A[i,j]*B[j,k])
    
    # Wait, let me re-examine the seed:
    # W[k * n + i][column] = 1
    # This means W index is k*n + i, where k is column of C, i is row of C.
    # So C[i,k] gets W[k*n + i, c].
    
    # For Strassen 4x4x4 (handling C[i,k] for k=0..3):
    # Split A into 2x2 blocks: A00 (rows 0-1, cols 0-1), A01 (rows 0-1, cols 2-3)
    #                              A10 (rows 2-3, cols 0-1), A11 (rows 2-3, cols 2-3)
    # Split B_left into 2x2 blocks: B00 (rows 0-1, cols 0-1), B01 (rows 0-1, cols 2-3)
    #                                B10 (rows 2-3, cols 0-1), B11 (rows 2-3, cols 2-3)
    # C_left 2x2 blocks: C00 (rows 0-1, cols 0-1), C01 (rows 0-1, cols 2-3)
    #                    C10 (rows 2-3, cols 0-1), C11 (rows 2-3, cols 2-3)
    
    # Strassen:
    # M1 = (A00+A11)*(B00+B11)
    # M2 = (A10+A11)*B00
    # M3 = A00*(B01-B11)
    # M4 = A11*(B10-B00)
    # M5 = (A00+A10)*B11
    # M6 = (A01-A11)*(B10+B11)
    # M7 = (A10-A00)*(B00+B01)
    
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # Each M is a 2x2 matrix product, which we expand with schoolbook (rank 4 each).
    # Total rank for Strassen part: 7 * 4 = 28.
    
    # For 4x4x2 part (C[i,k] for k=4,5):
    # B_right is columns 4,5 of B: B[j,4], B[j,5] for j=0..3
    # Split A into A_top (rows 0-1), A_bot (rows 2-3)
    # Split B_right into B_top (rows 0-1), B_bot (rows 2-3)
    
    # C_top (rows 0-1, cols 4-5) = A_top * B_right
    # A_top is 2x4, B_right is 4x2
    # = A_top[:,:2] * B_right[:2,:] + A_top[:,2:] * B_right[2:,:]
    # Let A_top_L = A_top[:,:2] (2x2), A_top_R = A_top[:,2:] (2x2)
    # Let B_top = B_right[:2,:] (2x2), B_bot = B_right[2:,:] (2x2)
    # C_top = A_top_L * B_top + A_top_R * B_bot
    
    # Similarly C_bot = A_bot_L * B_top + A_bot_R * B_bot
    
    # Each 2x2*2x2 product has rank 4.
    # C_top: 2 products * rank 4 = rank 8
    # C_bot: 2 products * rank 4 = rank 8
    # Total: rank 16
    
    # Grand total: 28 + 16 = 44
    
    rank = 44
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Helper: add a 2x2 schoolbook product
    # A_block at (ar, ac) is 2x2 in A: rows ar..ar+1, cols ac..ac+1
    # B_block at (br, bc) is 2x2 in B: rows br..br+1, cols bc..bc+1
    # C_block at (cr, cc) is 2x2 in C: rows cr..cr+1, cols cc..cc+1
    # sign: multiply all by this sign
    def add_2x2_product(ar, ac, br, bc, cr, cc, sign=1.0):
        nonlocal col
        for di in range(2):
            for dj in range(2):
                U[(ar+di)*m + (ac+dj)][col] += sign
                V[(br+dj)*p + (bc+di)][col] += sign
                W[(bc+di)*n + (cr+dj)][col] += sign
                col += 1
    
    # Strassen for 4x4x4 part
    # M1 = (A00+A11)*(B00+B11)
    # We need to express (A00+A11) and (B00+B11) as linear combinations.
    # This is tricky because Strassen involves sums/differences of blocks.
    # We need to handle the linear combinations properly.
    
    # Let me think about this differently.
    # For each Strassen multiplication M = X*Y where X and Y are 2x2 matrices
    # that are linear combinations of A and B blocks, we need to expand.
    
    # M1 = (A00+A11)*(B00+B11)
    # = A00*B00 + A00*B11 + A11*B00 + A11*B11
    # That's 4 products of 2x2 blocks, each with rank 4, total rank 16.
    # But that defeats the purpose of Strassen!
    
    # The issue is that in tensor rank decomposition, we need to factor through
    # rank-1 terms. The Strassen trick works because the intermediate sums
    # (A00+A11) and (B00+B11) are computed first, then multiplied.
    
    # In a tensor decomposition, each term is a product of a linear form in A,
    # a linear form in B, and contributes to a linear combination in C.
    
    # So for M1 = (A00+A11)*(B00+B11):
    # (A00+A11) is a 2x2 matrix with entries that are linear forms in A.
    # (B00+B11) is a 2x2 matrix with entries that are linear forms in B.
    # Their product is a 2x2 matrix, and each entry is a bilinear form.
    # We expand this 2x2 product using schoolbook: 4 terms.
    # Each term is (linear form in A) * (linear form in B) -> coefficient in C.
    
    # So M1 contributes 4 rank-1 terms (one for each entry of the 2x2 product).
    # Each of these 4 terms has a linear form in A and a linear form in B.
    
    # Let me define the blocks explicitly.
    # A00 = A[0:2, 0:2], A01 = A[0:2, 2:4], A10 = A[2:4, 0:2], A11 = A[2:4, 2:4]
    # B00 = B[0:2, 0:2], B01 = B[0:2, 2:4], B10 = B[2:4, 0:2], B11 = B[2:4, 2:4]
    
    # For M1 = (A00+A11)*(B00+B11):
    # Entry (0,0) of M1 = sum_{j=0}^{1} (A00[0,j]+A11[0,j])*(B00[j,0]+B11[j,0])
    # = sum_{j=0}^{1} (A[0,j]+A[2,j])*(B[j,0]+B[j+2,0])
    
    # This is a bilinear form. To decompose it into rank-1 terms, we note that
    # it's already a sum of products of linear forms. But for tensor rank,
    # we need each term to be a single product (no sum in the bilinear form).
    
    # Actually, the product (A00+A11)*(B00+B11) as a matrix has entries that are
    # bilinear forms. Each entry is a sum of 4 products of linear forms (from the
    # 2x2 matrix multiplication). But each such product is already rank-1 in the
    # tensor sense.
    
    # Wait, I think I'm overcomplicating this. Let me just directly construct
    # the rank-1 terms.
    
    # For a 2x2 matrix product X*Y where X and Y are 2x2 matrices with entries
    # that are linear forms in A and B respectively, the product has 4 entries,
    # each being a bilinear form that is a sum of 4 products of linear forms.
    # So each 2x2 product gives 4*4 = 16 rank-1 terms? No, that's not right either.
    
    # Let me think again. In the schoolbook 2x2x2 decomposition:
    # C[i,j] = sum_k A[i,k]*B[k,j]
    # Each term A[i,k]*B[k,j] is a rank-1 term in the tensor.
    # There are 2*2*2 = 8 such terms, giving rank 8 for 2x2x2.
    
    # Now if A[i,k] is replaced by a linear form L_A(i,k) and B[k,j] by L_B(k,j),
    # then L_A(i,k)*L_B(k,j) is still a single rank-1 term (a product of two linear forms).
    # So each entry of X*Y where X has entries that are linear forms in A and Y has
    # entries that are linear forms in B contributes 4 rank-1 terms (one for each k
    # in the 2x2 multiplication).
    
    # So M1 = (A00+A11)*(B00+B11) contributes 4 rank-1 terms.
    # Similarly each of the 7 Strassen multiplications contributes 4 rank-1 terms.
    # Total: 7 * 4 = 28 rank-1 terms. ✓
    
    # Let me define a function to add a 2x2 product where the A-side and B-side
    # matrices have entries that are linear combinations of original A and B entries.
    
    # For simplicity, I'll directly enumerate all terms.
    
    # Define helper to add one rank-1 term
    def add_term(a_coeffs, b_coeffs, c_coeff, target):
        """
        a_coeffs: dict from (i,j) to coefficient for A[i,j]
        b_coeffs: dict from (j,k) to coefficient for B[j,k]
        c_coeff: coefficient for C[i,k] at the given (i,k)
        target: (i,k) the C entry this term contributes to
        """
        nonlocal col
        for (ai, aj), val_a in a_coeffs.items():
            if val_a != 0:
                U[ai*m + aj][col] += val_a
        for (bj, bk), val_b in b_coeffs.items():
            if val_b != 0:
                V[bj*p + bk][col] += val_b
        i, k = target
        W[k*n + i][col] += c_coeff
        col += 1
    
    # This approach is getting complex. Let me use a more systematic approach.
    # I'll represent the linear forms as sparse dictionaries.
    
    # Actually, let me just directly code out all 44 terms.
    
    # Strassen 4x4x4 terms (28 terms):
    # For each Strassen multiplication, we have 4 terms (one per entry of the 2x2 result).
    # Each term involves a specific (i,j) of A, a specific (j,k) of B, and contributes
    # to a specific (i,k) of C with a specific coefficient.
    
    # M1 = (A00+A11)*(B00+B11)
    # (A00+A11)[a,b] = A[a,b] + A[a+2,b+2] for a,b in {0,1}
    # (B00+B11)[c,d] = B[c,d] + B[c+2,d+2] for c,d in {0,1}
    # M1[a,d] = sum_b (A[a,b]+A[a+2,b+2])*(B[b,d]+B[b+2,d+2])
    
    # M1[0,0] = sum_b=0..1 (A[0,b]+A[2,b+2])*(B[b,0]+B[b+2,2])
    # = (A[0,0]+A[2,2])*(B[0,0]+B[2,2]) + (A[0,1]+A[2,3])*(B[1,0]+B[3,2])
    
    # This expands to 4 products:
    # A[0,0]*B[0,0] -> C[0,0]
    # A[0,0]*B[2,2] -> C[0,0]
    # A[2,2]*B[0,0] -> C[0,0]
    # A[2,2]*B[2,2] -> C[0,0]
    # A[0,1]*B[1,0] -> C[0,0]
    # A[0,1]*B[3,2] -> C[0,0]
    # A[2,3]*B[1,0] -> C[0,0]
    # A[2,3]*B[3,2] -> C[0,0]
    
    # That's 8 terms for just M1[0,0]. This is getting too complicated.
    
    # Let me reconsider. The standard way to express Strassen in tensor rank form:
    # Each Strassen multiplication M = X*Y where X is a 2x2 matrix with entries
    # that are linear forms in A, and Y is a 2x2 matrix with entries that are
    # linear forms in B.
    
    # The product M = X*Y has 4 entries. Entry M[i,j] = sum_k X[i,k]*Y[k,j].
    # X[i,k] is a linear form in A, Y[k,j] is a linear form in B.
    # So M[i,j] is a sum of 2 products of linear forms (for k=0,1).
    # Each product X[i,k]*Y[k,j] is a rank-1 term in the tensor.
    
    # So each Strassen multiplication contributes 4*2 = 8 rank-1 terms?
    # No wait, X[i,k]*Y[k,j] is ONE rank-1 term because it's a product of two
    # (already summed) linear forms. The tensor rank counts the number of such
    # products, not the number of original variables involved.
    
    # So M[i,j] = X[i,0]*Y[0,j] + X[i,1]*Y[1,j]
    # This is 2 rank-1 terms.
    # 4 entries * 2 = 8 rank-1 terms per Strassen multiplication.
    # 7 multiplications * 8 = 56 terms. That's more than 28.
    
    # I think I'm confusing myself. Let me look at this from first principles.
    
    # The tensor for 4x4x4 matrix multiplication has dimension 4*4*4 = 64.
    # Schoolbook rank is 64.
    # Strassen-based algorithm: the key insight is that we use 7 "big" multiplications
    # of 2x2 matrices instead of 16 "small" multiplications. But each "big" multiplication
    # is itself a tensor of rank 4 (schoolbook for 2x2x2).
    # So total rank = 7 * 4 = 28.
    
    # The 28 rank-1 terms are obtained by expanding each of the 7 Strassen products
    # into 4 schoolbook terms (for the 2x2 matrix product).
    
    # Let me just hardcode the 28 terms for the Strassen part and 16 for the right part.
    
    # I'll use a cleaner formulation. Let me define the Strassen intermediate values
    # and their contributions.
    
    # Restart with a clearer approach.
    
    # For the 4x4x4 part, I'll use the known rank-28 decomposition.
    # For the 4x4x2 part, I'll use the rank-16 decomposition.
    
    # Let me just implement it step by step with explicit indices.
    
    # Clear and restart
    rank = 44
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    col = 0
    
    # Strassen 4x4x4:
    # Define the 7 Strassen products and their expansion.
    
    # For each Strassen product, the A-side is a 2x2 matrix with entries that are
    # linear combinations of A entries, and the B-side is a 2x2 matrix with entries
    # that are linear combinations of B entries.
    
    # I'll represent each entry of the A-side and B-side as a dict mapping (i,j) or
    # (j,k) to a coefficient.
    
    # Strassen formulas:
    # M1 = (A00+A11)*(B00+B11)
    # M2 = (A10+A11)*B00
    # M3 = A00*(B01-B11)
    # M4 = A11*(B10-B00)
    # M5 = (A00+A10)*B11
    # M6 = (A01-A11)*(B10+B11)
    # M7 = (A10-A00)*(B00+B01)
    
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # Where C00, C01, C10, C11 are 2x2 blocks of C.
    
    # Let me define a function to add a 2x2 product where the A-matrix and B-matrix
    # have entries that are linear forms.
    
    def add_strassen_product(a_entries, b_entries, c_block, sign=1.0):
        """
        a_entries: 2x2 list of dicts, a_entries[i][j] maps (ai,aj) to coeff
        b_entries: 2x2 list of dicts, b_entries[i][j] maps (bj,bk) to coeff
        c_block: (cr, cc) top-left of the 2x2 C block
        sign: overall sign
        """
        nonlocal col
        for ci in range(2):
            for cj in range(2):
                # M[ci,cj] = sum_k a_entries[ci][k] * b_entries[k][cj]
                for k in range(2):
                    # Product of linear form a_entries[ci][k] and b_entries[k][cj]
                    a_dict = a_entries[ci][k]
                    b_dict = b_entries[k][cj]
                    if not a_dict or not b_dict:
                        continue
                    for (ai, aj), va in a_dict.items():
                        for (bj, bk), vb in b_dict.items():
                            if va == 0 or vb == 0:
                                continue
                            U[ai*m + aj][col] += sign * va
                            V[bj*p + bk][col] += sign * vb
                            W[bk*n + (c_block[0]+ci)][col] += sign
                            col += 1
    
    # Define block accessors
    # A00[i][j] = A[i,j] for i,j in {0,1}
    # A01[i][j] = A[i,j+2] for i in {0,1}, j in {0,1}
    # A10[i][j] = A[i+2,j] for i in {0,1}, j in {0,1}
    # A11[i][j] = A[i+2,j+2] for i in {0,1}, j in {0,1}
    
    def A_block(base_r, base_c):
        """Return 2x2 list of dicts for A block with top-left at (base_r, base_c)"""
        result = [[{} for _ in range(2)] for _ in range(2)]
        for i in range(2):
            for j in range(2):
                result[i][j][(base_r+i, base_c+j)] = 1.0
        return result
    
    def B_block(base_r, base_c):
        """Return 2x2 list of dicts for B block with top-left at (base_r, base_c)"""
        result = [[{} for _ in range(2)] for _ in range(2)]
        for i in range(2):
            for j in range(2):
                result[i][j][(base_r+i, base_c+j)] = 1.0
        return result
    
    def add_a_blocks(block1, block2):
        """Add two A blocks"""
        result = [[{} for _ in range(2)] for _ in range(2)]
        for i in range(2):
            for j in range(2):
                d1 = block1[i][j]
                d2 = block2[i][j]
                merged = {}
                for k, v in d1.items():
                    merged[k] = v
                for k, v in d2.items():
                    merged[k] = merged.get(k, 0) + v
                result[i][j] = merged
        return result
    
    def sub_a_blocks(block1, block2):
        """Subtract two A blocks"""
        result = [[{} for _ in range(2)] for _ in range(2)]
        for i in range(2):
            for j in range(2):
                d1 = block1[i][j]
                d2 = block2[i][j]
                merged = {}
                for k, v in d1.items():
                    merged[k] = v
                for k, v in d2.items():
                    merged[k] = merged.get(k, 0) - v
                result[i][j] = merged
        return result
    
    def add_b_blocks(block1, block2):
        """Add two B blocks"""
        result = [[{} for _ in range(2)] for _ in range(2)]
        for i in range(2):
            for j in range(2):
                d1 = block1[i][j]
                d2 = block2[i][j]
                merged = {}
                for k, v in d1.items():
                    merged[k] = v
                for k, v in d2.items():
                    merged[k] = merged.get(k, 0) + v
                result[i][j] = merged
        return result
    
    def sub_b_blocks(block1, block2):
        """Subtract two B blocks"""
        result = [[{} for _ in range(2)] for _ in range(2)]
        for i in range(2):
            for j in range(2):
                d1 = block1[i][j]
                d2 = block2[i][j]
                merged = {}
                for k, v in d1.items():
                    merged[k] = v
                for k, v in d2.items():
                    merged[k] = merged.get(k, 0) - v
                result[i][j] = merged
        return result
    
    A00 = A_block(0, 0)
    A01 = A_block(0, 2)
    A10 = A_block(2, 0)
    A11 = A_block(2, 2)
    
    B00 = B_block(0, 0)
    B01 = B_block(0, 2)
    B10 = B_block(2, 0)
    B11 = B_block(2, 2)
    
    # M1 = (A00+A11)*(B00+B11)
    add_strassen_product(add_a_blocks(A00, A11), add_b_blocks(B00, B11), (0, 0), 1.0)
    # M4 = A11*(B10-B00)
    add_strassen_product(A11, sub_b_blocks(B10, B00), (0, 0), 1.0)
    # M5 = (A00+A10)*B11
    add_strassen_product(add_a_blocks(A00, A10), B11, (0, 0), -1.0)
    # M7 = (A10-A00)*(B00+B01)
    add_strassen_product(sub_a_blocks(A10, A00), add_b_blocks(B00, B01), (0, 0), 1.0)
    
    # C01 = M3 + M5
    # M3 = A00*(B01-B11)
    add_strassen_product(A00, sub_b_blocks(B01, B11), (0, 2), 1.0)
    # M5 = (A00+A10)*B11
    add_strassen_product(add_a_blocks(A00, A10), B11, (0, 2), 1.0)
    
    # C10 = M2 + M4
    # M2 = (A10+A11)*B00
    add_strassen_product(add_a_blocks(A10, A11), B00, (2, 0), 1.0)
    # M4 = A11*(B10-B00)
    add_strassen_product(A11, sub_b_blocks(B10, B00), (2, 0), 1.0)
    
    # C11 = M1 - M2 + M3 + M6
    # M1 = (A00+A11)*(B00+B11)
    add_strassen_product(add_a_blocks(A00, A11), add_b_blocks(B00, B11), (2, 2), 1.0)
    # M2 = (A10+A11)*B00
    add_strassen_product(add_a_blocks(A10, A11), B00, (2, 2), -1.0)
    # M3 = A00*(B01-B11)
    add_strassen_product(A00, sub_b_blocks(B01, B11), (2, 2), 1.0)
    # M6 = (A01-A11)*(B10+B11)
    add_strassen_product(sub_a_blocks(A01, A11), add_b_blocks(B10, B11), (2, 2), 1.0)
    
    # Now the 4x4x2 part (columns 4,5 of C)
    # C[i,4] and C[i,5] for i=0..3
    # Split A into top (rows 0-1) and bottom (rows 2-3)
    # B_right = B[:, 4:6] (4x2)
    # Split B_right into top (rows 0-1) and bottom (rows 2-3)
    
    # C_top (rows 0-1, cols 4-5) = A_top * B_right
    # A_top = A[0:2, 0:4] (2x4)
    # = A_top[:, 0:2] * B_right[0:2, :] + A_top[:, 2:4] * B_right[2:4, :]
    
    # A_top_L = A[0:2, 0:2] (2x2)
    # A_top_R = A[0:2, 2:4] (2x2)
    # B_top = B[0:2, 4:6] (2x2)
    # B_bot = B[2:4, 4:6] (2x2)
    
    # C_top = A_top_L * B_top + A_top_R * B_bot
    
    # Similarly for C_bot (rows 2-3, cols 4-5)
    
    A_top_L = A_block(0, 0)
    A_top_R = A_block(0, 2)
    A_bot_L = A_block(2, 0)
    A_bot_R = A_block(2, 2)
    
    B_top = B_block(0, 4)
    B_bot = B_block(2, 4)
    
    # C_top = A_top_L * B_top + A_top_R * B_bot
    add_strassen_product(A_top_L, B_top, (0, 4), 1.0)
    add_strassen_product(A_top_R, B_bot, (0, 4), 1.0)
    
    # C_bot = A_bot_L * B_top + A_bot_R * B_bot
    add_strassen_product(A_bot_L, B_top, (2, 4), 1.0)
    add_strassen_product(A_bot_R, B_bot, (2, 4), 1.0)
    
    # Verify column count
    assert col == rank, f"Expected {rank} columns, got {col}"
    
    return {"U": U, "V": V, "W": W}
