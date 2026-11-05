import numpy as np

DIMENSIONS = [6, 6, 6]

def construct():
    n, m, p = DIMENSIONS  # 6, 6, 6
    
    # Block structure: 3x3 blocks of 2x2
    # Block (a,b) of a 6x6 matrix corresponds to rows 2a..2a+1, cols 2b..2b+1
    # (0-indexed: rows 2a, 2a+1 and cols 2b, 2b+1)
    
    # Strassen for 3x3:
    # M1 = (A00 + A11 + A22) * (B00 + B11 + B22)  -- not standard, let me use a known 3x3 Strassen
    # Actually, Strassen's original algorithm is for 2x2. For 3x3, there are various algorithms.
    # Let me use a simple approach: treat 3x3 as a 2x2 of blocks (one block is 2x2, others are 1x1)
    # Or just use the schoolbook for 3x3 and Strassen for 2x2.
    
    # Actually, let me just use schoolbook for the outer 3x3 (9 multiplications) 
    # and Strassen for the inner 2x2 (7 multiplications each).
    # Total rank = 9 * 7 = 63
    
    # But wait, I need to verify this is a valid decomposition.
    
    # Let me try a different approach: just use the block structure with schoolbook at both levels.
    # Outer: 9 block multiplications
    # Inner: each block multiplication is 2x2 x 2x2 = 8 scalar multiplications (schoolbook)
    # Total: 9 * 8 = 72
    
    # But the schoolbook rank of 6x6 is 216, not 72. So the block structure doesn't give a simple product.
    
    # I think the issue is that the block multiplication is not independent. When we do
    # C_block[i,j] = sum_k A_block[i,k] * B_block[k,j]
    # each A_block[i,k] * B_block[k,j] is a 2x2 x 2x2 multiplication, which has 8 terms.
    # But these 8 terms are NOT independent across different (i,k,j) triples.
    
    # Actually, I think they ARE independent in the sense that each term in the 2x2 multiplication
    # corresponds to a specific scalar multiplication in the full 6x6 problem.
    
    # Let me just construct the decomposition explicitly.
    
    # For the full 6x6 multiplication:
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # 
    # In block notation (a = i//2, alpha = i%2, etc.):
    # C[2a+alpha, 2c+gamma] = sum_{b,beta} A[2a+alpha, 2b+beta] * B[2b+beta, 2c+gamma]
    #
    # = sum_{b} sum_{beta} A[2a+alpha, 2b+beta] * B[2b+beta, 2c+gamma]
    #
    # For fixed (a,b,c), this is a 2x2 x 2x2 multiplication:
    # C_block[a,c][alpha,gamma] = sum_{beta} A_block[a,b][alpha,beta] * B_block[b,c][beta,gamma]
    #
    # The 2x2 x 2x2 multiplication has 8 terms (schoolbook):
    # C[alpha,gamma] = sum_{beta} A[alpha,beta] * B[beta,gamma]
    # = A[0,0]*B[0,gamma] + A[0,1]*B[1,gamma] + A[1,0]*B[0,gamma] + A[1,1]*B[1,gamma]
    # (for each gamma)
    
    # So the full decomposition has 9 (outer) * 8 (inner) = 72 terms?
    # But the schoolbook has 216 terms. 72 != 216, so something is wrong.
    
    # OH! I see my error. The 2x2 x 2x2 multiplication has 4 outputs, each with 2 terms, so 8 terms total.
    # But in the full 6x6 problem, for each (a,b,c) triple, we have 4 output entries (alpha,gamma)
    # and 4 input pairs (alpha,beta) for each. So the number of scalar multiplications for one
    # (a,b,c) triple is 4 (outputs) * 2 (sum terms) = 8.
    # Total: 9 * 8 = 72.
    
    # But the schoolbook for 6x6 has 6*6*6 = 216 terms. 72 != 216.
    
    # The discrepancy is because in the schoolbook, each term A[i,j]*B[j,k] is a separate multiplication.
    # In the block structure, for fixed (a,b,c), the 8 terms are:
    # A[2a+0, 2b+0]*B[2b+0, 2c+0], A[2a+0, 2b+1]*B[2b+1, 2c+0],
    # A[2a+1, 2b+0]*B[2b+0, 2c+0], A[2a+1, 2b+1]*B[2b+1, 2c+0],
    # A[2a+0, 2b+0]*B[2b+0, 2c+1], A[2a+0, 2b+1]*B[2b+1, 2c+1],
    # A[2a+1, 2b+0]*B[2b+0, 2c+1], A[2a+1, 2b+1]*B[2b+1, 2c+1]
    
    # These are 8 specific scalar multiplications. Over all (a,b,c), we get 9*8 = 72 scalar multiplications.
    # But the schoolbook has 216 = 6*6*6 scalar multiplications.
    
    # 72 != 216, so the block structure does NOT give a valid decomposition with 72 terms!
    
    # Why? Because in the schoolbook, the term A[i,j]*B[j,k] contributes to C[i,k].
    # In the block structure, for fixed (a,b,c), the 8 terms contribute to C_block[a,c].
    # But C_block[a,c] is a 2x2 matrix, so it has 4 entries. Each entry is a sum of 2 products.
    # So the 8 terms for one (a,b,c) triple contribute to 4 output entries.
    
    # The total number of (input pair, output entry) triples is:
    # 9 (a,b,c) * 4 (alpha,gamma) * 2 (beta) = 72
    # But the schoolbook has 6*6*6 = 216 such triples.
    
    # 72 != 216. So the block structure gives FEWER terms than schoolbook?
    # That can't be right. Let me recount.
    
    # In the schoolbook: for each i,j,k in {1,...,6}, we have one term A[i,j]*B[j,k] contributing to C[i,k].
    # Total: 6*6*6 = 216 terms.
    
    # In the block structure: for each a,b,c in {1,2,3} and alpha,beta,gamma in {1,2},
    # we have a term A[2a+alpha, 2b+beta]*B[2b+beta, 2c+gamma] contributing to C[2a+alpha, 2c+gamma].
    # Total: 3*3*3*2*2*2 = 216 terms.
    
    # AH! I forgot the alpha and gamma indices. The 2x2 x 2x2 multiplication has:
    # For each output (alpha, gamma): sum over beta of A[alpha,beta]*B[beta,gamma]
    # So there are 2*2*2 = 8 terms for the 2x2 multiplication.
    # And for each (a,b,c) triple, there are 8 terms.
    # Total: 9 * 8 = 72? No, 9 * 8 = 72, but we need 216.
    
    # Wait, 3*3*3*2*2*2 = 216. And 9*8 = 72. These are different!
    # 3*3*3 = 27, and 2*2*2 = 8, so 27*8 = 216. Yes!
    # And 9*8 = 72. So 27*8 != 9*8. The error was in counting the outer terms.
    
    # The outer structure has 3*3*3 = 27 (a,b,c) triples, not 9.
    # For each (a,b,c), there are 2*2*2 = 8 inner terms.
    # Total: 27 * 8 = 216. Matches schoolbook!
    
    # So the block structure with schoolbook at both levels gives rank 216, same as naive schoolbook.
    # To reduce the rank, we need to use a lower-rank algorithm at one or both levels.
    
    # If we use Strassen for the inner 2x2 (rank 7 instead of 8):
    # Total rank = 27 * 7 = 189. THIS MATCHES THE OBSERVED RANK!
    
    # So the approach is:
    # 1. For each (a,b,c) triple (27 total), use Strassen's algorithm for 2x2 multiplication (7 terms)
    # 2. Total rank = 27 * 7 = 189
    
    # Let me implement this!
    
    # Strassen for 2x2:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B10-B11)
    # M4 = A11(B01-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B10)
    # M7 = (A01-A11)(B01+B11)
    # C00 = M1+M4-M5+M7
    # C01 = M3+M5
    # C10 = M2+M4
    # C11 = M1-M2+M3+M6
    
    # For the tensor decomposition, each M_t corresponds to a rank-1 term:
    # M1 = (A00+A11)(B00+B11)
    #   U contribution: A00 coefficient +1, A11 coefficient +1
    #   V contribution: B00 coefficient +1, B11 coefficient +1
    #   W contribution: C00 +1, C11 +1 (with signs from the final combination)
    
    # Actually, the W coefficients depend on how the M_t's are combined to get C.
    # C00 = M1+M4-M5+M7, so W for C00 gets +1 from M1, +1 from M4, -1 from M5, +1 from M7
    # etc.
    
    # Let me define the Strassen 2x2 decomposition explicitly.
    
    # For a 2x2 multiplication with inputs A[0,0],A[0,1],A[1,0],A[1,1] and B[0,0],B[0,1],B[1,0],B[1,1]
    # and outputs C[0,0],C[0,1],C[1,0],C[1,1]:
    
    # The 7 rank-1 terms are:
    # Term 1: U = [1,0,0,1] (A00+A11), V = [1,0,0,1] (B00+B11), W = [1,0,0,1] (C00+C11)
    #   Wait, W should have the coefficients for each output.
    #   C00 gets +1 from M1, C11 gets +1 from M1.
    #   So W = [1, 0, 0, 1] for C00,C01,C10,C11.
    
    # Term 2: U = [0,0,1,1] (A10+A11), V = [1,0,0,0] (B00), W = [0,0,1,0] (C10)
    #   C10 gets +1 from M2.
    
    # Term 3: U = [1,0,0,0] (A00), V = [0,0,1,-1] (B10-B11), W = [0,1,0,0] (C01)
    #   C01 gets +1 from M3.
    
    # Term 4: U = [0,0,0,1] (A11), V = [0,1,-1,0] (B01-B00), W = [1,0,0,0] (C00)
    #   C00 gets +1 from M4.
    
    # Term 5: U = [1,1,0,0] (A00+A01), V = [0,0,0,1] (B11), W = [-1,1,0,0] (-C00+C01)
    #   C00 gets -1 from M5, C01 gets +1 from M5.
    
    # Term 6: U = [0,0,1,-1] (A10-A00), V = [1,0,1,0] (B00+B10), W = [0,0,0,1] (C11)
    #   C11 gets +1 from M6.
    
    # Term 7: U = [0,1,0,-1] (A01-A11), V = [0,1,0,1] (B01+B11), W = [1,0,0,0] (C00)
    #   C00 gets +1 from M7.
    
    # Let me verify: C00 = M1+M4-M5+M7
    #   = (A00+A11)(B00+B11) + A11(B01-B00) - (A00+A01)B11 + (A01-A11)(B01+B11)
    #   = A00*B00 + A00*B11 + A11*B00 + A11*B11 + A11*B01 - A11*B00 - A00*B11 - A01*B11 + A01*B01 + A01*B11 - A11*B01 - A11*B11
    #   = A00*B00 + A01*B01
    #   ✓
    
    # C01 = M3+M5
    #   = A00(B10-B11) + (A00+A01)B11
    #   = A00*B10 - A00*B11 + A00*B11 + A01*B11
    #   = A00*B10 + A01*B11
    #   ✓
    
    # C10 = M2+M4
    #   = (A10+A11)B00 + A11(B01-B00)
    #   = A10*B00 + A11*B00 + A11*B01 - A11*B00
    #   = A10*B00 + A11*B01
    #   ✓
    
    # C11 = M1-M2+M3+M6
    #   = (A00+A11)(B00+B11) - (A10+A11)B00 + A00(B10-B11) + (A10-A00)(B00+B10)
    #   = A00*B00 + A00*B11 + A11*B00 + A11*B11 - A10*B00 - A11*B00 + A00*B10 - A00*B11 + A10*B00 + A10*B10 - A00*B00 - A00*B10
    #   = A10*B10 + A11*B11
    #   ✓
    
    # Great, the Strassen 2x2 decomposition is correct.
    
    # Now, for the full 6x6 decomposition:
    # For each (a,b,c) in {0,1,2}^3, we have a 2x2 x 2x2 multiplication.
    # The inputs are A_block[a,b] and B_block[b,c], and the output is C_block[a,c].
    # 
    # In the full 6x6 indexing:
    # A[2a+alpha, 2b+beta] for alpha,beta in {0,1}
    # B[2b+beta, 2c+gamma] for beta,gamma in {0,1}
    # C[2a+alpha, 2c+gamma] for alpha,gamma in {0,1}
    
    # The U matrix has shape (36, rank), where row index is i*m+j = i*6+j for A[i,j]
    # The V matrix has shape (36, rank), where row index is j*p+k = j*6+k for B[j,k]
    # The W matrix has shape (36, rank), where row index is k*n+i = k*6+i for C[i,k]
    
    # For the (a,b,c) block, the 7 Strassen terms give:
    # U rows: (2a+alpha)*6 + (2b+beta) for the A entries
    # V rows: (2b+beta)*6 + (2c+gamma) for the B entries
    # W rows: (2c+gamma)*6 + (2a+alpha) for the C entries
    
    # Let me construct the full decomposition.
    
    rank = 27 * 7  # 189
    U = [[0] * rank for _ in range(36)]
    V = [[0] * rank for _ in range(36)]
    W = [[0] * rank for _ in range(36)]
    
    # Strassen 2x2 coefficients
    # U coefficients for A[0,0],A[0,1],A[1,0],A[1,1]
    # V coefficients for B[0,0],B[0,1],B[1,0],B[1,1]
    # W coefficients for C[0,0],C[0,1],C[1,0],C[1,1]
    strassen_terms = [
        # (U_coeffs, V_coeffs, W_coeffs)
        ([1,0,0,1], [1,0,0,1], [1,0,0,1]),  # M1
        ([0,0,1,1], [1,0,0,0], [0,0,1,0]),  # M2
        ([1,0,0,0], [0,0,1,-1], [0,1,0,0]),  # M3
        ([0,0,0,1], [0,1,-1,0], [1,0,0,0]),  # M4
        ([1,1,0,0], [0,0,0,1], [-1,1,0,0]),  # M5
        ([0,0,1,-1], [1,0,1,0], [0,0,0,1]),  # M6
        ([0,1,0,-1], [0,1,0,1], [1,0,0,0]),  # M7
    ]
    
    col = 0
    for a in range(3):
        for b in range(3):
            for c in range(3):
                for (u_coeffs, v_coeffs, w_coeffs) in strassen_terms:
                    # U: A[2a+alpha, 2b+beta]
                    for alpha in range(2):
                        for beta in range(2):
                            U[(2*a+alpha)*6 + (2*b+beta)][col] = u_coeffs[alpha*2+beta]
                    # V: B[2b+beta, 2c+gamma]
                    for beta in range(2):
                        for gamma in range(2):
                            V[(2*b+beta)*6 + (2*c+gamma)][col] = v_coeffs[beta*2+gamma]
                    # W: C[2a+alpha, 2c+gamma]
                    for alpha in range(2):
                        for gamma in range(2):
                            W[(2*c+gamma)*6 + (2*a+alpha)][col] = w_coeffs[alpha*2+gamma]
                    col += 1
    
    return {"U": U, "V": V, "W": W}
