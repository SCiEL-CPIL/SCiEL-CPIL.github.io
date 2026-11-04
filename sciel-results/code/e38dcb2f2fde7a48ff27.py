import time
import random
from itertools import product

DIMENSIONS = [4, 4, 5]

def construct():
    n, m, p = DIMENSIONS
    start = time.time()
    deadline = start + 115

    def verify(rank, U, V, W):
        """Check T[i,j,k] = sum_c U[i*m+j,c] V[j*p+k,c] W[k*n+i,c] == 1 for all i,j,k."""
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0
                    for c in range(rank):
                        s += U[i*m + j][c] * V[j*p + k][c] * W[k*n + i][c]
                    if abs(s - 1) > 1e-9:
                        return False
        return True

    def make_strassen_4x4x5():
        """Construct rank-65 decomposition for 4x4x5 using Strassen at block level.
        
        Strategy:
        - Split 4x4 into 2x2 blocks
        - Use 7 Strassen multiplications at block level
        - Each 2x2 block multiplication uses 7 scalar multiplications (Strassen)
        - Total for 4x4x4: 7*7 = 49
        - Plus 16 for the 5th column (4x4x1 schoolbook)
        - Total: 65
        """
        rank = 65
        U = [[0]*rank for _ in range(n*m)]  # 16 x 65
        V = [[0]*rank for _ in range(m*p)]  # 20 x 65
        W = [[0]*rank for _ in range(p*n)]  # 20 x 65

        # Strassen 2x2x2 scalar decomposition (7 multiplications)
        # For multiplying 2x2 matrices:
        # C00 = M1 + M4 - M5 + M7
        # C01 = M3 + M5
        # C10 = M2 + M4
        # C11 = M1 - M2 + M3 + M6
        #
        # Where:
        # M1 = (a00+a11)(b00+b11)
        # M2 = (a10+a11)(b00)
        # M3 = (a00)(b01-b11)
        # M4 = (a11)(b10-b00)
        # M5 = (a00+a01)(b11)
        # M6 = (a10-a00)(b00+b01)
        # M7 = (a01-a11)(b10+b11)

        # Block-level Strassen for 4x4x4:
        # A blocks: A00(0,1)x(0,1), A01(0,1)x(2,3), A10(2,3)x(0,1), A11(2,3)x(2,3)
        # B blocks: B00(0,1)x(0,1), B01(0,1)x(2,3), B10(2,3)x(0,1), B11(2,3)x(2,3)
        # C blocks: C00(0,1)x(0,1), C01(0,1)x(2,3), C10(2,3)x(0,1), C11(2,3)x(2,3)
        #
        # Block-level Strassen multiplications:
        # S1 = (A00+A11)(B00+B11)
        # S2 = (A10+A11)(B00)
        # S3 = (A00)(B01-B11)
        # S4 = (A11)(B10-B00)
        # S5 = (A00+A01)(B11)
        # S6 = (A10-A00)(B00+B01)
        # S7 = (A01-A11)(B10+B11)
        #
        # C00 = S1 + S4 - S5 + S7
        # C01 = S3 + S5
        # C10 = S2 + S4
        # C11 = S1 - S2 + S3 + S6

        # For each block-level Strassen multiplication St, we need to express it
        # as a sum of 7 scalar multiplications using Strassen on 2x2 blocks.
        
        # Let's define helper: for a 2x2 matrix multiplication (a*b), 
        # the 7 Strassen factors and their contributions to output positions.
        
        # For 2x2 matrices with elements indexed (r,c) where r,c in {0,1}:
        # a = [[a00, a01], [a10, a11]]
        # b = [[b00, b01], [b10, b11]]
        # c = a*b
        #
        # Strassen factors:
        # m1 = (a00+a11)*(b00+b11)
        # m2 = (a10+a11)*(b00)
        # m3 = (a00)*(b01-b11)
        # m4 = (a11)*(b10-b00)
        # m5 = (a00+a01)*(b11)
        # m6 = (a10-a00)*(b00+b01)
        # m7 = (a01-a11)*(b10+b11)
        #
        # c00 = m1 + m4 - m5 + m7
        # c01 = m3 + m5
        # c10 = m2 + m4
        # c11 = m1 - m2 + m3 + m6

        # For the 4x4x4 part, we have 7 block-level Strassen multiplications.
        # Each involves 2x2 matrices, so each needs 7 scalar multiplications.
        
        # Let's enumerate: for block-level factor t (0-6), and scalar factor s (0-6),
        # we have a total of 49 scalar multiplications.
        
        # Block-level factor definitions (which A and B blocks are combined):
        # t=0: S1 = (A00+A11)(B00+B11)
        # t=1: S2 = (A10+A11)(B00)
        # t=2: S3 = (A00)(B01-B11)
        # t=3: S4 = (A11)(B10-B00)
        # t=4: S5 = (A00+A01)(B11)
        # t=5: S6 = (A10-A00)(B00+B01)
        # t=6: S7 = (A01-A11)(B10+B11)
        
        # For each t, the A-block combination and B-block combination:
        # t=0: A: A00+A11, B: B00+B11
        # t=1: A: A10+A11, B: B00
        # t=2: A: A00, B: B01-B11
        # t=3: A: A11, B: B10-B00
        # t=4: A: A00+A01, B: B11
        # t=5: A: A10-A00, B: B00+B01
        # t=6: A: A01-A11, B: B10+B11
        
        # A block row/col indices (in the 4x4 matrix):
        # A00: rows 0-1, cols 0-1
        # A01: rows 0-1, cols 2-3
        # A10: rows 2-3, cols 0-1
        # A11: rows 2-3, cols 2-3
        
        # B block row/col indices (in the 4x5 matrix, but we only use first 4 cols for 4x4x4):
        # B00: rows 0-1, cols 0-1
        # B01: rows 0-1, cols 2-3
        # B10: rows 2-3, cols 0-1
        # B11: rows 2-3, cols 2-3
        
        # For each block-level factor t, define:
        # - Which A blocks are involved and with what signs
        # - Which B blocks are involved and with what signs
        
        # A block combinations for each t:
        # t=0: A00(+1) + A11(+1)
        # t=1: A10(+1) + A11(+1)
        # t=2: A00(+1)
        # t=3: A11(+1)
        # t=4: A00(+1) + A01(+1)
        # t=5: A10(+1) - A00(+1)  => A10(+1) + A00(-1)
        # t=6: A01(+1) - A11(+1)  => A01(+1) + A11(-1)
        
        a_blocks = [
            [(0, 1)],   # t=0: A00+A11
            [(1, 1)],   # t=1: A10+A11
            [(0, 1)],   # t=2: A00
            [(1, 1)],   # t=3: A11
            [(0, 1), (0, 2)],  # t=4: A00+A01
            [(1, 1), (0, -1)], # t=5: A10-A00
            [(0, 2), (1, -1)]  # t=6: A01-A11
        ]
        
        # Wait, I need to be more careful. Let me define blocks by (block_row, block_col)
        # A00: (0,0), A01: (0,1), A10: (1,0), A11: (1,1)
        
        a_block_defs = [
            [(0, 0, 1), (1, 1, 1)],    # t=0: A00(+1) + A11(+1)
            [(1, 0, 1), (1, 1, 1)],    # t=1: A10(+1) + A11(+1)
            [(0, 0, 1)],                # t=2: A00(+1)
            [(1, 1, 1)],                # t=3: A11(+1)
            [(0, 0, 1), (0, 1, 1)],    # t=4: A00(+1) + A01(+1)
            [(1, 0, 1), (0, 0, -1)],   # t=5: A10(+1) - A00(+1)
            [(0, 1, 1), (1, 1, -1)]    # t=6: A01(+1) - A11(+1)
        ]
        
        # B block combinations for each t:
        # t=0: B00(+1) + B11(+1)
        # t=1: B00(+1)
        # t=2: B01(+1) - B11(+1)
        # t=3: B10(+1) - B00(+1)
        # t=4: B11(+1)
        # t=5: B00(+1) + B01(+1)
        # t=6: B10(+1) + B11(+1)
        
        b_block_defs = [
            [(0, 0, 1), (1, 1, 1)],    # t=0: B00(+1) + B11(+1)
            [(0, 0, 1)],                # t=1: B00(+1)
            [(0, 1, 1), (1, 1, -1)],   # t=2: B01(+1) - B11(+1)
            [(1, 0, 1), (0, 0, -1)],   # t=3: B10(+1) - B00(+1)
            [(1, 1, 1)],                # t=4: B11(+1)
            [(0, 0, 1), (0, 1, 1)],    # t=5: B00(+1) + B01(+1)
            [(1, 0, 1), (1, 1, 1)]     # t=6: B10(+1) + B11(+1)
        ]
        
        # For each block-level factor t, and each scalar Strassen factor s (0-6),
        # we need to determine the coefficients for U, V, W.
        
        # Scalar Strassen factors for 2x2 multiplication:
        # s=0: (a00+a11)*(b00+b11)
        # s=1: (a10+a11)*(b00)
        # s=2: (a00)*(b01-b11)
        # s=3: (a11)*(b10-b00)
        # s=4: (a00+a01)*(b11)
        # s=5: (a10-a00)*(b00+b01)
        # s=6: (a01-a11)*(b10+b11)
        
        # For scalar factor s, the A-side coefficient for element (ar, ac) where ar,ac in {0,1}:
        # s=0: a00(+1), a11(+1)
        # s=1: a10(+1), a11(+1)
        # s=2: a00(+1)
        # s=3: a11(+1)
        # s=4: a00(+1), a01(+1)
        # s=5: a10(+1), a00(-1)
        # s=6: a01(+1), a11(-1)
        
        a_scalar_defs = [
            [(0, 0, 1), (1, 1, 1)],    # s=0
            [(1, 0, 1), (1, 1, 1)],    # s=1
            [(0, 0, 1)],                # s=2
            [(1, 1, 1)],                # s=3
            [(0, 0, 1), (0, 1, 1)],    # s=4
            [(1, 0, 1), (0, 0, -1)],   # s=5
            [(0, 1, 1), (1, 1, -1)]    # s=6
        ]
        
        # For scalar factor s, the B-side coefficient for element (br, bc) where br,bc in {0,1}:
        # s=0: b00(+1), b11(+1)
        # s=1: b00(+1)
        # s=2: b01(+1), b11(-1)
        # s=3: b10(+1), b00(-1)
        # s=4: b11(+1)
        # s=5: b00(+1), b01(+1)
        # s=6: b10(+1), b11(+1)
        
        b_scalar_defs = [
            [(0, 0, 1), (1, 1, 1)],    # s=0
            [(0, 0, 1)],                # s=1
            [(0, 1, 1), (1, 1, -1)],   # s=2
            [(1, 0, 1), (0, 0, -1)],   # s=3
            [(1, 1, 1)],                # s=4
            [(0, 0, 1), (0, 1, 1)],    # s=5
            [(1, 0, 1), (1, 1, 1)]     # s=6
        ]
        
        # For each block-level factor t and scalar factor s, the output C-block contribution:
        # The scalar factor s contributes to C-block elements based on Strassen output formulas.
        # 
        # For the 2x2 result of scalar multiplication:
        # c00 = m0 + m3 - m4 + m6
        # c01 = m2 + m4
        # c10 = m1 + m3
        # c11 = m0 - m1 + m2 + m5
        #
        # So scalar factor s contributes to:
        # s=0 (m0): c00(+1), c11(+1)
        # s=1 (m1): c10(+1), c11(-1)
        # s=2 (m2): c01(+1), c11(+1)
        # s=3 (m3): c00(+1), c10(+1)
        # s=4 (m4): c00(-1), c01(+1)
        # s=5 (m5): c11(+1)
        # s=6 (m6): c00(+1)
        
        # Output C-block element contributions from scalar factor s:
        # c_out[block_row, block_col] += coeff
        # where block_row, block_col in {0,1}
        
        c_contrib_defs = [
            [(0, 0, 1), (1, 1, 1)],    # s=0: c00(+1), c11(+1)
            [(1, 0, 1), (1, 1, -1)],   # s=1: c10(+1), c11(-1)
            [(0, 1, 1), (1, 1, 1)],    # s=2: c01(+1), c11(+1)
            [(0, 0, 1), (1, 0, 1)],    # s=3: c00(+1), c10(+1)
            [(0, 0, -1), (0, 1, 1)],   # s=4: c00(-1), c01(+1)
            [(1, 1, 1)],                # s=5: c11(+1)
            [(0, 0, 1)]                 # s=6: c00(+1)
        ]
        
        # Now, for each block-level factor t, the C-block output:
        # t=0 (S1): C00(+1), C11(+1)
        # t=1 (S2): C10(+1), C11(-1)
        # t=2 (S3): C01(+1), C11(+1)
        # t=3 (S4): C00(+1), C10(+1)
        # t=4 (S5): C00(-1), C01(+1)
        # t=5 (S6): C11(+1)
        # t=6 (S7): C00(+1)
        
        # Wait, this is the same pattern! Let me verify:
        # C00 = S1 + S4 - S5 + S7
        # C01 = S3 + S5
        # C10 = S2 + S4
        # C11 = S1 - S2 + S3 + S6
        
        # So:
        # S1 (t=0): C00(+1), C11(+1)
        # S2 (t=1): C10(+1), C11(-1)
        # S3 (t=2): C01(+1), C11(+1)
        # S4 (t=3): C00(+1), C10(+1)
        # S5 (t=4): C00(-1), C01(+1)
        # S6 (t=5): C11(+1)
        # S7 (t=6): C00(+1)
        
        # This matches c_contrib_defs! So the block-level and scalar-level Strassen
        # have the same structure.
        
        # Now let's build the decomposition:
        # For each (t, s) pair, we have a scalar multiplication.
        # The U factor involves A-side coefficients.
        # The V factor involves B-side coefficients.
        # The W factor involves C-side coefficients.
        
        # For a given (t, s):
        # U[i,j] for i in 0-3, j in 0-3:
        #   For each A-block (br, bc) with coefficient ca from a_block_defs[t],
        #   and for each A-scalar element (ar, ac) with coefficient cs from a_scalar_defs[s],
        #   U[i,j] += ca * cs where i = 2*br + ar, j = 2*bc + ac
        #
        # Similarly for V and W.
        
        for t in range(7):
            for s in range(7):
                c_idx = t * 7 + s  # 0 to 48
                
                # Compute U coefficients
                # For each A-block in a_block_defs[t]
                for (abr, abc, ca) in a_block_defs[t]:
                    for (ar, ac, cs) in a_scalar_defs[s]:
                        i = 2 * abr + ar
                        j = 2 * abc + ac
                        U[i * m + j][c_idx] += ca * cs
                
                # Compute V coefficients
                # For each B-block in b_block_defs[t]
                for (bbr, bbc, cb) in b_block_defs[t]:
                    for (br, bc, cs) in b_scalar_defs[s]:
                        j = 2 * bbr + br  # row in B (which is col in A, but we use j for B row)
                        k = 2 * bbc + bc  # col in B
                        V[j * p + k][c_idx] += cb * cs
                
                # Compute W coefficients
                # For each C-block output in c_contrib_defs[s] (scalar level)
                # and the block-level output from t
                # Actually, the C-block output is determined by both t and s.
                # The scalar factor s determines which C-block elements are affected,
                # and the block-level factor t determines the sign/magnitude at block level.
                
                # Wait, I need to reconsider. The block-level factor t determines
                # which C-blocks are affected and with what sign. The scalar factor s
                # determines which elements within those C-blocks are affected.
                
                # From the Strassen formulas:
                # Block level: C00 = S1 + S4 - S5 + S7, etc.
                # Scalar level: c00 = m0 + m3 - m4 + m6, etc.
                
                # So for a given (t, s):
                # The C-block is determined by t (which S factor)
                # The C-element within the block is determined by s (which m factor)
                
                # Let me redefine:
                # For block-level factor t, the C-block contributions:
                # t=0 (S1): C00(+1), C11(+1)
                # t=1 (S2): C10(+1), C11(-1)
                # t=2 (S3): C01(+1), C11(+1)
                # t=3 (S4): C00(+1), C10(+1)
                # t=4 (S5): C00(-1), C01(+1)
                # t=5 (S6): C11(+1)
                # t=6 (S7): C00(+1)
                
                # For scalar factor s, the C-element contributions within a block:
                # s=0 (m0): c00(+1), c11(+1)
                # s=1 (m1): c10(+1), c11(-1)
                # s=2 (m2): c01(+1), c11(+1)
                # s=3 (m3): c00(+1), c10(+1)
                # s=4 (m4): c00(-1), c01(+1)
                # s=5 (m5): c11(+1)
                # s=6 (m6): c00(+1)
                
                # So for (t, s), the C-block is from t, and the C-element is from s.
                
                # Block-level C contributions:
                c_block_defs = [
                    [(0, 0, 1), (1, 1, 1)],    # t=0: C00(+1), C11(+1)
                    [(1, 0, 1), (1, 1, -1)],   # t=1: C10(+1), C11(-1)
                    [(0, 1, 1), (1, 1, 1)],    # t=2: C01(+1), C11(+1)
                    [(0, 0, 1), (1, 0, 1)],    # t=3: C00(+1), C10(+1)
                    [(0, 0, -1), (0, 1, 1)],   # t=4: C00(-1), C01(+1)
                    [(1, 1, 1)],                # t=5: C11(+1)
                    [(0, 0, 1)]                 # t=6: C00(+1)
                ]
                
                # For each C-block (cbr, cbc) with coefficient cc from c_block_defs[t],
                # and for each C-element (cr, cc2) with coefficient cs2 from c_contrib_defs[s],
                # W[k*n+i] += cc * cs2 where k = 2*cbr + cr, i = 2*cbc + cc2
                
                for (cbr, cbc, cc) in c_block_defs[t]:
                    for (cr, cc2, cs2) in c_contrib_defs[s]:
                        k = 2 * cbr + cr
                        i = 2 * cbc + cc2
                        W[k * n + i][c_idx] += cc * cs2
        
        # Now add the 5th column (k=4) using schoolbook: 16 multiplications
        # C[i,4] = sum_j A[i,j] * B[j,4]
        for i in range(n):
            for j in range(m):
                c_idx = 49 + i * m + j
                U[i * m + j][c_idx] = 1
                V[j * p + 4][c_idx] = 1
                W[4 * n + i][c_idx] = 1
        
        return {"U": U, "V": V, "W": W}

    best = make_strassen_4x4x5()
    best_rank = 65

    # Verify
    if not verify(best_rank, best["U"], best["V"], best["W"]):
        # Fallback to schoolbook
        r = n * m * p
        U = [[0]*r for _ in range(n*m)]
        V = [[0]*r for _ in range(m*p)]
        W = [[0]*r for _ in range(p*n)]
        c = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j][c] = 1
                    V[j*p+k][c] = 1
                    W[k*n+i][c] = 1
                    c += 1
        best = {"U": U, "V": V, "W": W}
        best_rank = r

    return best

# EVOLVE-BLOCK-END
