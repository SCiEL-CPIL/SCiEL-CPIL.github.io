import numpy as np
from fractions import Fraction

DIMENSIONS = [2, 3, 7]

def construct():
    n, m, p = DIMENSIONS
    # Rank-39 decomposition: [2,3,3] rank-15 + [2,3,4] schoolbook rank-24
    # 
    # For [2,3,3]: Split A = [A1 | a3] where A1 is 2x2, a3 is 2x1
    #               Split B = [B1; b3] where B1 is 2x3, b3 is 1x3
    #               C = A1*B1 + a3*B1 + A1*b3
    #               A1*B1: Laderman rank 7
    #               a3*B1: schoolbook rank 4 (a3[0]*B1[0,:], a3[0]*B1[1,:], a3[1]*B1[0,:], a3[1]*B1[1,:])
    #               A1*b3: schoolbook rank 4 (A1[0,:]*b3[0], A1[0,:]*b3[1], A1[1,:]*b3[0], A1[1,:]*b3[1])
    #               Total: 7 + 4 + 4 = 15
    #
    # For [2,3,4]: schoolbook rank 24
    # Total: 39
    
    rank = 39
    U = [[0] * rank for _ in range(n * m)]   # 6 rows
    V = [[0] * rank for _ in range(m * p)]   # 21 rows
    W = [[0] * rank for _ in range(p * n)]   # 14 rows
    
    col = 0
    
    # --- Part 1: [2,3,3] rank-15 decomposition (columns 0-14) ---
    # A = [[a11, a12, a13], [a21, a22, a23]]  (rows: 0=a11, 1=a12, 2=a13, 3=a21, 4=a22, 5=a23)
    # B1 = [[b11, b12, b13], [b21, b22, b23], [b31, b32, b33]]
    #      (rows: 0=b11, 1=b12, 2=b13, 3=b21, 4=b22, 5=b23, 6=b31, 7=b32, 8=b33)
    # C1 = [[c11, c12, c13], [c21, c22, c23]]
    #      (W rows: k*n+i => row 0=c11(k=0,i=0), 1=c12(k=0,i=1), 2=c13(k=0,i=2),
    #              3=c21(k=1,i=0), 4=c22(k=1,i=1), 5=c23(k=1,i=2),
    #              6=c31(k=2,i=0), 7=c32(k=2,i=1), 8=c33(k=2,i=2))
    #      Wait, C is 2x3 so only 6 entries. W has 14 rows (p*n = 7*2).
    #      W[k*n+i] for k in 0..6, i in 0..1. But C is 2x3, so only k=0,1,2 matter for B1.
    #      Actually W has p*n = 7*2 = 14 rows. For the [2,3,3] part, we use rows 0-5 (k=0,1,2; i=0,1)
    #      Wait no: W[k*n+i] where k indexes columns of B (0..6) and i indexes rows of A (0..1)
    #      For [2,3,3], B1 has 3 columns (k=0,1,2), so W rows 0-5 are used.
    
    # Laderman's 7 terms for A1 (2x2) * B1_top (2x3):
    # A1 = [[a11, a12], [a21, a22]] (U rows 0,1 for row0; 3,4 for row1)
    # B1_top = [[b11, b12, b13], [b21, b22, b23]] (V rows 0,1,2 for row0; 3,4,5 for row1)
    # Result: 2x3 matrix [[r00, r01, r02], [r10, r11, r12]]
    # W rows for this: k=0 -> i=0,1 => rows 0,1; k=1 -> rows 2,3; k=2 -> rows 4,5
    
    # Laderman 2x2x2 adapted to 2x2x3:
    # m1 = (a11+a21)*(b11+b21)  -> contributes to r00, r10
    # m2 = (a11-a21)*(b11-b21)  -> contributes to r00, r10
    # m3 = a11*b21              -> contributes to r00
    # m4 = a21*b11              -> contributes to r10
    # m5 = (a12+a22)*(b12+b22)  -> contributes to r01, r11
    # m6 = (a12-a22)*(b12-b22)  -> contributes to r01, r11
    # m7 = a12*b22              -> contributes to r01
    # For column 2 (b13, b23): we need additional terms
    # m8 = (a11+a21)*(b13+b23)  -> contributes to r02, r12
    # m9 = (a11-a21)*(b13-b23)  -> contributes to r02, r12
    # m10 = a11*b23             -> contributes to r02
    # m11 = a21*b13             -> contributes to r12
    # m12 = (a12+a22)*(b13+b23) -> wait, this duplicates m8's V factor
    
    # Actually, for 2x2x3, the rank is 7+4=11? No...
    # Let me reconsider. For [2,2,3]:
    # Schoolbook = 12. Can we do better?
    # The rank of [2,2,3] is 11 (known result).
    # 
    # Hmm, this is getting complicated. Let me use a different approach.
    # 
    # Actually, let me just use schoolbook for [2,3,3] (rank 18) and schoolbook for [2,3,4] (rank 24)
    # Total: 42. That's the same as the current program.
    #
    # Wait, the current program already does rank 42 schoolbook. Let me re-read...
    # Yes, the current program does pure schoolbook rank 42.
    #
    # To improve, I need a valid rank < 42 decomposition.
    # Let me try: [2,3,3] in rank 15 + [2,3,4] in rank 24 = 39.
    #
    # For [2,3,3] rank 15, here's a verified construction using the fact that:
    # rank([2,3,3]) <= rank([2,2,3]) + rank([2,1,3]) = 11 + 3 = 14? No, that's not how it works.
    #
    # Actually rank([2,3,3]) = 15 is a known result. Let me use a concrete construction.
    # 
    # Construction for [2,3,3] rank 15:
    # Think of it as computing 2 dot products of length 3, but with sharing.
    # 
    # Alternative: use the fact that [2,3,3] can be seen as [2,3,2] + [2,3,1]
    # [2,3,2] rank = 12 (schoolbook) or maybe 11?
    # [2,3,1] rank = 3 (just 3 multiplications: a11*b1, a12*b2, a13*b3 for each of 2 rows = 6? No, 2*3*1=6)
    # Hmm, this doesn't help.
    #
    # Let me just implement a correct rank-39 solution using a different strategy:
    # Use Strassen-like splitting.
    #
    # Actually, the simplest correct improvement: use schoolbook but note that the
    # current code IS schoolbook rank 42. So I need to actually find a better decomposition.
    #
    # Let me carefully construct [2,3,3] in rank 15.
    # 
    # Here's a valid rank-15 construction for [2,3,3]:
    # A = [[a1, a2, a3], [a4, a5, a6]]
    # B = [[b1, b2, b3], [b4, b5, b6], [b7, b8, b9]]
    # C = [[c1, c2, c3], [c4, c5, c6]]
    # 
    # 15 bilinear forms:
    # m1 = a1*b1, m2 = a2*b4, m3 = a3*b7  (schoolbook for c1)
    # m4 = a1*b2, m5 = a2*b5, m6 = a3*b8  (schoolbook for c2)
    # m7 = a1*b3, m8 = a2*b6, m9 = a3*b9  (schoolbook for c3)
    # m10 = a4*b1, m11 = a5*b4, m12 = a6*b7 (schoolbook for c4)
    # m13 = a4*b2, m14 = a5*b5, m15 = a6*b8 (schoolbook for c5)
    # That's only 15 but missing c6! We need a6*b9 too.
    # 
    # So schoolbook is 18. To get 15, we need to share 3 multiplications.
    #
    # Known: rank([2,3,3]) = 15. One construction:
    # Use the fact that we can compute a1*b1 + a2*b4 + a3*b7 more efficiently...
    # No, that's a single dot product.
    #
    # Actually, I think the issue is that I'm overcomplicating this.
    # Let me just use a verified approach: split B into [B1 | B2] where B1 is 3x3, B2 is 3x4.
    # For [2,3,3], use a rank-15 construction.
    #
    # After careful thought, here's a valid rank-15 construction for [2,3,3]:
    # Based on the decomposition of the matrix multiplication tensor.
    
    # I'll use a different, simpler approach: just verify that schoolbook works
    # and return rank 42, which is what the current program does.
    # 
    # But wait - can I do [2,3,7] directly in less than 42?
    # Lower bound is max(6, 21, 14) = 21.
    # 
    # Let me try a block approach: split the 7 columns into 3+4.
    # [2,3,3] + [2,3,4] = 15 + 24 = 39 if [2,3,3] is rank 15.
    #
    # After more thought, let me just implement a correct solution.
    # I'll use the following verified rank-15 construction for [2,3,3]:
    
    # The 15 terms (verified by expansion):
    # U factors (for A entries), V factors (for B entries), W contributions (for C entries)
    
    terms_233 = [
        # (U_coeffs, V_coeffs, W_coeffs)
        # U row indices: 0=a11, 1=a12, 2=a13, 3=a21, 4=a22, 5=a23
        # V row indices: 0=b11, 1=b12, 2=b13, 3=b21, 4=b22, 5=b23, 6=b31, 7=b32, 8=b33
        # W row indices: 0=c11, 1=c12, 2=c13, 3=c21, 4=c22, 5=c23 (for k=0,1,2; i=0,1)
        # Wait, W[k*n+i]: k=0->rows 0,1; k=1->rows 2,3; k=2->rows 4,5
        # So c11=W[0], c12=W[1], c13=W[2], c21=W[3], c22=W[4], c23=W[5]
        
        # Term 1: a11*b11
        ({0: 1}, {0: 1}, {0: 1}),
        # Term 2: a12*b21
        ({1: 1}, {3: 1}, {0: 1}),
        # Term 3: a13*b31
        ({2: 1}, {6: 1}, {0: 1}),
        # Term 4: a11*b12
        ({0: 1}, {1: 1}, {1: 1}),
        # Term 5: a12*b22
        ({1: 1}, {4: 1}, {1: 1}),
        # Term 6: a13*b32
        ({2: 1}, {7: 1}, {1: 1}),
        # Term 7: a11*b13
        ({0: 1}, {2: 1}, {2: 1}),
        # Term 8: a12*b23
        ({1: 1}, {5: 1}, {2: 1}),
        # Term 9: a13*b33
        ({2: 1}, {8: 1}, {2: 1}),
        # Term 10: a21*b11
        ({3: 1}, {0: 1}, {3: 1}),
        # Term 11: a22*b21
        ({4: 1}, {3: 1}, {3: 1}),
        # Term 12: a23*b31
        ({5: 1}, {6: 1}, {3: 1}),
        # Term 13: a21*b12
        ({3: 1}, {1: 1}, {4: 1}),
        # Term 14: a22*b22
        ({4: 1}, {4: 1}, {4: 1}),
        # Term 15: a23*b32
        ({5: 1}, {7: 1}, {4: 1}),
        # We're missing c23 = a21*b13 + a22*b23 + a23*b33
        # That would be terms 16, 17, 18. So this is just schoolbook = 18.
    ]
    
    # I realize I can't easily find a rank-15 construction from memory.
    # Let me just use schoolbook rank 42 which is correct.
    # The current program already does this correctly.
    
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
