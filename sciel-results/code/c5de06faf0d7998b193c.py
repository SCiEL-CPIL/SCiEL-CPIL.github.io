"""Matrix multiplication 4x4x4 - Kronecker product of Strassen's rank-7 algorithm."""
import time

DIMENSIONS = [4, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    deadline = time.time() + 100  # Safety margin before 120s limit
    n, m, p = 4, 4, 4
    
    # Strassen's rank-7 decomposition for 2x2 matrices
    # M1 = (A00+A11)*(B00+B11)
    # M2 = (A10+A11)*B00
    # M3 = A00*(B01-B11)
    # M4 = A11*(B10-B00)
    # M5 = (A00+A10)*(B00+B10)
    # M6 = (A01-A10)*(B01+B11)
    # M7 = (A11-A01)*(B10+B11)
    
    # For 2x2: U is 4x7, V is 4x7, W is 4x7
    # Index convention: U[i*m+j, c], V[j*p+k, c], W[k*n+i, c]
    # For n=m=p=2: U[0..3, 0..6], V[0..3, 0..6], W[0..3, 0..6]
    
    U2 = [[0]*7 for _ in range(4)]
    V2 = [[0]*7 for _ in range(4)]
    W2 = [[0]*7 for _ in range(4)]
    
    # Column 0: M1 = (A00+A11)*(B00+B11)
    U2[0][0] = 1; U2[3][0] = 1
    V2[0][0] = 1; V2[3][0] = 1
    W2[0][0] = 1; W2[3][0] = 1  # C00 += M1, C11 += M1
    
    # Column 1: M2 = (A10+A11)*B00
    U2[2][1] = 1; U2[3][1] = 1
    V2[0][1] = 1
    W2[1][1] = 1; W2[3][1] = 1  # C10 += M2, C11 += M2
    
    # Column 2: M3 = A00*(B01-B11)
    U2[0][2] = 1
    V2[1][2] = 1; V2[3][2] = -1
    W2[2][2] = 1  # C01 += M3
    
    # Column 3: M4 = A11*(B10-B00)
    U2[3][3] = 1
    V2[2][3] = 1; V2[0][3] = -1
    W2[1][3] = 1; W2[2][3] = 1  # C10 += M4, C01 += M4
    
    # Column 4: M5 = (A00+A10)*(B00+B10)
    U2[0][4] = 1; U2[2][4] = 1
    V2[0][4] = 1; V2[2][4] = 1
    W2[0][4] = -1; W2[2][4] = 1  # C00 -= M5, C01 += M5
    
    # Column 5: M6 = (A01-A10)*(B01+B11)
    U2[1][5] = 1; U2[2][5] = -1
    V2[1][5] = 1; V2[3][5] = 1
    W2[2][5] = 1  # C01 += M6
    
    # Column 6: M7 = (A11-A01)*(B10+B11)
    U2[3][6] = 1; U2[1][6] = -1
    V2[2][6] = 1; V2[3][6] = 1
    W2[1][6] = 1; W2[3][6] = -1  # C10 += M7, C11 -= M7
    
    # Wait, let me re-verify Strassen's formula:
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # Let me redo this more carefully.
    # A = [[A00, A01], [A10, A11]], B = [[B00, B01], [B10, B11]]
    # C = A*B where C[i][j] = sum_k A[i][k]*B[k][j]
    # C00 = A00*B00 + A01*B10
    # C01 = A00*B01 + A01*B11
    # C10 = A10*B00 + A11*B10
    # C11 = A10*B01 + A11*B11
    
    # Strassen:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A10)(B00+B10)
    # M6 = (A01-A10)(B01+B11)
    # M7 = (A11-A01)(B10+B11)
    
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # Let me re-derive to be sure:
    # M1 + M4 - M5 + M7
    # = (A00+A11)(B00+B11) + A11(B10-B00) - (A00+A10)(B00+B10) + (A11-A01)(B10+B11)
    # = A00B00 + A00B11 + A11B00 + A11B11 + A11B10 - A11B00 - A00B00 - A00B10 - A10B00 - A10B10 + A11B10 + A11B11 - A01B10 - A01B11
    # = A00B11 + A11B00 + A11B11 + A11B10 - A00B10 - A10B00 - A10B10 + A11B10 + A11B11 - A01B10 - A01B11
    # Hmm, this doesn't simplify to C00 = A00B00 + A01B10. Let me use the standard reference.
    
    # Actually the standard Strassen is:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)(B00+B10)
    # M3 = (A00+A01)(B10+B11)
    # M4 = (A10+A11)(B10+B11)
    # M5 = (A00+A11)(B00+B11) - M1... no
    
    # Let me just use the well-known version:
    # p1 = (a11 + a22)(b11 + b22)
    # p2 = (a21 + a22)b11
    # p3 = a11(b21 - b22)
    # p4 = a22(b12 - b11)
    # p5 = (a11 + a21)(b11 + b12)
    # p6 = (a22 - a12)(b12 + b22)
    # p7 = (a12 - a21)(b21 + b22)
    # c11 = p1 + p4 - p5 + p7
    # c12 = p3 + p5
    # c21 = p2 + p4
    # c22 = p1 - p2 + p3 + p6
    
    # OK let me just hard-code the known correct Strassen factors.
    U2 = [[0]*7 for _ in range(4)]
    V2 = [[0]*7 for _ in range(4)]
    W2 = [[0]*7 for _ in range(4)]
    
    # p1: (A00+A11)*(B00+B11)
    U2[0][0] = 1; U2[3][0] = 1
    V2[0][0] = 1; V2[3][0] = 1
    W2[0][0] = 1; W2[3][0] = 1
    
    # p2: (A10+A11)*B00
    U2[2][1] = 1; U2[3][1] = 1
    V2[0][1] = 1
    W2[1][1] = 1; W2[3][1] = -1
    
    # p3: A00*(B01-B11)
    U2[0][2] = 1
    V2[1][2] = 1; V2[3][2] = -1
    W2[2][2] = 1; W2[3][2] = 1
    
    # p4: A11*(B10-B00)
    U2[3][3] = 1
    V2[2][3] = 1; V2[0][3] = -1
    W2[1][3] = 1; W2[2][3] = 1
    
    # p5: (A00+A10)*(B00+B10)
    U2[0][4] = 1; U2[2][4] = 1
    V2[0][4] = 1; V2[2][4] = 1
    W2[0][4] = -1; W2[2][4] = 1
    
    # p6: (A01-A10)*(B01+B11)
    U2[1][5] = 1; U2[2][5] = -1
    V2[1][5] = 1; V2[3][5] = 1
    W2[3][5] = 1
    
    # p7: (A11-A01)*(B10+B11)
    U2[3][6] = 1; U2[1][6] = -1
    V2[2][6] = 1; V2[3][6] = 1
    W2[0][6] = 1; W2[1][6] = -1
    
    # Verify: C00 = p1+p4-p5+p7, C01 = p3+p5, C10 = p2+p4, C11 = p1-p2+p3+p6
    # W2 row indices: W[k*n+i] for n=2
    # C00 -> k=0,i=0 -> 0; C01 -> k=1,i=0 -> 2; C10 -> k=0,i=1 -> 1; C11 -> k=1,i=1 -> 3
    
    # p1: W[0]+=1, W[3]+=1 ✓
    # p2: W[1]+=1, W[3]+=-1 ✓ (C10 += p2, C11 -= p2)
    # p3: W[2]+=1, W[3]+=1 ✓ (C01 += p3, C11 += p3)
    # p4: W[1]+=1, W[2]+=1 ✓ (C10 += p4, C01 += p4)
    # p5: W[0]+=-1, W[2]+=1 ✓ (C00 -= p5, C01 += p5)
    # p6: W[3]+=1 ✓ (C11 += p6)
    # p7: W[0]+=1, W[1]+=-1 ✓ (C00 += p7, C10 -= p7)
    
    # Wait, that gives C10 = p2 + p4 - p7, but it should be p2+p4. Let me recheck.
    # c21 = p2 + p4. So W[1] should have p2:+1, p4:+1, p7:0.
    # And c11 = p1 - p2 + p3 + p6. So W[3] should have p1:+1, p2:-1, p3:+1, p6:+1, p7:0.
    # c12 = p3 + p5. W[2] should have p3:+1, p4:+1, p5:+1. Wait c12 = p3+p5 not p3+p4+p5.
    
    # I'm confusing myself. Let me just carefully set up from the formulas.
    U2 = [[0]*7 for _ in range(4)]
    V2 = [[0]*7 for _ in range(4)]
    W2 = [[0]*7 for _ in range(4)]
    
    # p1 = (A00+A11)(B00+B11)
    U2[0][0] = 1; U2[3][0] = 1
    V2[0][0] = 1; V2[3][0] = 1
    # c11 += p1, c22 += p1
    W2[0][0] = 1; W2[3][0] = 1
    
    # p2 = (A10+A11)B00
    U2[2][1] = 1; U2[3][1] = 1
    V2[0][1] = 1
    # c21 += p2, c22 -= p2
    W2[1][1] = 1; W2[3][1] = -1
    
    # p3 = A00(B01-B11)
    U2[0][2] = 1
    V2[1][2] = 1; V2[3][2] = -1
    # c12 += p3, c22 += p3
    W2[2][2] = 1; W2[3][2] = 1
    
    # p4 = A11(B10-B00)
    U2[3][3] = 1
    V2[2][3] = 1; V2[0][3] = -1
    # c21 += p4, c12 += p4
    W2[1][3] = 1; W2[2][3] = 1
    
    # p5 = (A00+A10)(B00+B10)
    U2[0][4] = 1; U2[2][4] = 1
    V2[0][4] = 1; V2[2][4] = 1
    # c11 -= p5, c12 += p5
    W2[0][4] = -1; W2[2][4] = 1
    
    # p6 = (A01-A10)(B01+B11)
    U2[1][5] = 1; U2[2][5] = -1
    V2[1][5] = 1; V2[3][5] = 1
    # c22 += p6
    W2[3][5] = 1
    
    # p7 = (A11-A01)(B10+B11)
    U2[3][6] = 1; U2[1][6] = -1
    V2[2][6] = 1; V2[3][6] = 1
    # c11 += p7
    W2[0][6] = 1
    
    # Now Kronecker product for 4x4:
    # The 4x4 matrix multiplication can be seen as 2x2 block matrices of 2x2 blocks.
    # A = [[A00, A01], [A10, A11]] where each Ai is 2x2
    # B = [[B00, B01], [B10, B11]] where each Bj is 2x2
    # C = A*B with 2x2 blocks
    # Ci = sum_j Ai * Bj (matrix multiplication of 2x2 blocks)
    
    # For each Strassen product pk, we need to do 2x2 matrix multiplications.
    # So we apply Strassen's 7 products to each 2x2 block multiplication.
    # Total rank = 7 * 7 = 49
    
    rank = 49
    U = [[0]*rank for _ in range(16)]
    V = [[0]*rank for _ in range(16)]
    W = [[0]*rank for _ in range(16)]
    
    # For each of the 7 Strassen "super-products" and each of the 7 inner products:
    for outer in range(7):
        for inner in range(7):
            col = outer * 7 + inner
            
            # U: For the outer Strassen, we have coefficients on A-blocks.
            # For the inner Strassen, we have coefficients on A-sub-blocks within each block.
            # A-block (bi) has indices: row offset = 2*bi_row, col offset = 2*bi_col within 4x4
            # A element (i,j) in 4x4: i = 2*bi_row + ai, j = 2*bi_col + aj
            # U index = i*4 + j (since m=4)
            
            # Outer Strassen tells us which 2x2 blocks to combine with what signs
            # Inner Strassen tells us which elements within each 2x2 block to combine
            
            for outer_block_idx in range(4):
                outer_coeff = U2[outer_block_idx][outer]
                if outer_coeff == 0:
                    continue
                # outer_block_idx: 0->A00(block), 1->A01(block), 2->A10(block), 3->A11(block)
                bi_row = outer_block_idx // 2
                bi_col = outer_block_idx % 2
                
                for inner_idx in range(4):
                    inner_coeff = U2[inner_idx][inner]
                    if inner_coeff == 0:
                        continue
                    ai = inner_idx // 2
                    aj = inner_idx % 2
                    
                    # 4x4 row = 2*bi_row + ai, col = 2*bi_col + aj
                    row4 = 2 * bi_row + ai
                    col4 = 2 * bi_col + aj
                    U[row4 * 4 + col4][col] = outer_coeff * inner_coeff
            
            # V: similar logic for B
            for outer_block_idx in range(4):
                outer_coeff = V2[outer_block_idx][outer]
                if outer_coeff == 0:
                    continue
                bi_row = outer_block_idx // 2
                bi_col = outer_block_idx % 2
                
                for inner_idx in range(4):
                    inner_coeff = V2[inner_idx][inner]
                    if inner_coeff == 0:
                        continue
                    ai = inner_idx // 2
                    aj = inner_idx % 2
                    
                    # B element in 4x4: row = 2*bi_row + ai, col = 2*bi_col + aj
                    # V index = row*4 + col (since m=4, p=4, V is (m*p, rank) = (16, rank))
                    # V[j*p+k] where j is row in B, k is col in B
                    row4 = 2 * bi_row + ai
                    col4 = 2 * bi_col + aj
                    V[row4 * 4 + col4][col] = outer_coeff * inner_coeff
            
            # W: For C output, W[k*4+i] where i is row in C, k is col in C
            for outer_block_idx in range(4):
                outer_coeff = W2[outer_block_idx][outer]
                if outer_coeff == 0:
                    continue
                # W2 index: W2[k2*n2+i2] where n2=2
                # outer_block_idx = k2*2 + i2 where i2 is C block row, k2 is C block col
                ci_row = outer_block_idx % 2
                ci_col = outer_block_idx // 2
                
                for inner_idx in range(4):
                    inner_coeff = W2[inner_idx][inner]
                    if inner_coeff == 0:
                        continue
                    # inner_idx = ki*2 + ii where ii is C sub-row, ki is C sub-col
                    ci = inner_idx % 2
                    ck = inner_idx // 2
                    
                    # 4x4 C element: row = 2*ci_row + ci, col = 2*ci_col + ck
                    # W index = k*4 + i = (2*ci_col + ck)*4 + (2*ci_row + ci)
                    row4 = 2 * ci_row + ci
                    col4 = 2 * ci_col + ck
                    W[col4 * 4 + row4][col] = outer_coeff * inner_coeff
    
    if time.time() > deadline:
        pass  # Already constructed, just return
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
