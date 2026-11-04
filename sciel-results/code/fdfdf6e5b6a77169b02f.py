"""Matrix multiplication tensor decomposition for 2x4 by 4x8 matrices."""
import numpy as np

DIMENSIONS = [2, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = 2, 4, 8
    # Rank 56 decomposition using Strassen's algorithm embedded in block structure.
    # A is 2x4, B is 4x8.
    # Split A into A1 (2x2) and A2 (2x2).
    # Split B into B11 (2x4), B12 (2x4), B21 (2x4), B22 (2x4).
    # C = [A1*B11 + A2*B21, A1*B12 + A2*B22].
    # Each term is a 2x2 * 2x4 multiplication.
    # Rank of 2x2 * 2x4 is 14 (2 * Strassen rank 7).
    # Total rank 4 * 14 = 56.
    
    rank = 56
    U = np.zeros((8, rank))
    V = np.zeros((32, rank))
    W = np.zeros((16, rank))
    
    # Helper to add Strassen factors for X (2x2) * Y (2x4) -> Z (2x4)
    # X indices in U: x_rows (2x2), y_rows in V: y_rows (2x4), z_rows in W: z_rows (2x4)
    # x_rows: list of 4 indices (i*m + j) for the 2x2 block of A
    # y_rows: list of 8 indices (j*p + k) for the 2x4 block of B
    # z_rows: list of 8 indices (k*n + i) for the 2x4 block of C
    
    def add_strassen_224(col_offset, x_rows, y_rows, z_rows):
        # y_rows is 2x4, so y_rows[0:2] is row 0 of Y (4 cols), y_rows[2:4] is row 1?
        # No, V is indexed by (j, k). j is row of B (0..3), k is col of B (0..7).
        # For a 2x4 block of B, we have 2 rows of B and 4 cols of B.
        # Let's say the block occupies B rows r0, r1 and cols c0..c3.
        # The indices in V are:
        # (r0, c0), (r0, c1), (r0, c2), (r0, c3)
        # (r1, c0), (r1, c1), (r1, c2), (r1, c3)
        # So y_rows should be a list of 8 integers.
        
        # X is 2x2. Indices in U:
        # (x0, x1) for row 0 of X
        # (x2, x3) for row 1 of X
        # x_rows list: [u_r0c0, u_r0c1, u_r1c0, u_r1c1]
        
        # Z is 2x4. Indices in W:
        # W is indexed by (k, i). k is col of C, i is row of C.
        # C rows are 0,1. C cols are c0..c3.
        # Indices:
        # (c0, 0), (c1, 0), (c2, 0), (c3, 0)  -> Row 0 of Z
        # (c0, 1), (c1, 1), (c2, 1), (c3, 1)  -> Row 1 of Z
        # z_rows list: [w_c0r0, w_c1r0, w_c2r0, w_c3r0, w_c0r1, w_c1r1, w_c2r1, w_c3r1]
        
        # Strassen for X * Y
        # X = [[X00, X01], [X10, X11]]
        # Y = [[Y00, Y01, Y02, Y03], [Y10, Y11, Y12, Y13]]
        # Z = [[Z00, Z01, Z02, Z03], [Z10, Z11, Z12, Z13]]
        
        # We treat Y as two 2x2 matrices Y1 = [[Y00, Y01], [Y10, Y11]] and Y2 = [[Y02, Y03], [Y12, Y13]]
        # Z1 = X * Y1, Z2 = X * Y2
        
        # Strassen for X * Y1 (7 mults)
        # M1 = (X00+X11)(Y00+Y11)
        # M2 = (X10+X11)Y00
        # M3 = X00(Y01-Y11)
        # M4 = X11(Y10-Y00)
        # M5 = (X00+X01)Y11
        # M6 = (X10-X00)(Y00+Y01)
        # M7 = (X01-X11)(Y10+Y11)
        
        # Z00 = M1 + M4 - M5 + M7
        # Z01 = M3 + M5
        # Z10 = M2 + M4
        # Z11 = M1 - M2 + M3 + M6
        
        # Indices in local 2x2 Y1:
        # y00 = y_rows[0]
        # y01 = y_rows[1]
        # y10 = y_rows[4]
        # y11 = y_rows[5]
        
        # Indices in local 2x2 X:
        # x00 = x_rows[0]
        # x01 = x_rows[1]
        # x10 = x_rows[2]
        # x11 = x_rows[3]
        
        # Indices in local 2x2 Z1 (part of Z):
        # z00 = z_rows[0]
        # z01 = z_rows[1]
        # z10 = z_rows[4]
        # z11 = z_rows[5]
        
        # 1. M1
        U[x00, col_offset] += 1; U[x11, col_offset] += 1
        V[y00, col_offset] += 1; V[y11, col_offset] += 1
        W[z00, col_offset] += 1; W[z11, col_offset] += 1
        
        # 2. M2
        U[x10, col_offset+1] += 1; U[x11, col_offset+1] += 1
        V[y00, col_offset+1] += 1
        W[z10, col_offset+1] += 1
        
        # 3. M3
        U[x00, col_offset+2] += 1
        V[y01, col_offset+2] += 1; V[y11, col_offset+2] -= 1
        W[z01, col_offset+2] += 1
        
        # 4. M4
        U[x11, col_offset+3] += 1
        V[y10, col_offset+3] += 1; V[y00, col_offset+3] -= 1
        W[z00, col_offset+3] += 1; W[z10, col_offset+3] += 1
        
        # 5. M5
        U[x00, col_offset+4] += 1; U[x01, col_offset+4] += 1
        V[y11, col_offset+4] += 1
        W[z00, col_offset+4] -= 1; W[z01, col_offset+4] += 1
        
        # 6. M6
        U[x10, col_offset+5] += 1; U[x00, col_offset+5] -= 1
        V[y00, col_offset+5] += 1; V[y01, col_offset+5] += 1
        W[z11, col_offset+5] += 1
        
        # 7. M7
        U[x01, col_offset+6] += 1; U[x11, col_offset+6] -= 1
        V[y10, col_offset+6] += 1; V[y11, col_offset+6] += 1
        W[z00, col_offset+6] += 1; W[z11, col_offset+6] -= 1
        
        # Now for Y2 (cols 2,3 of Y)
        # y02 = y_rows[2]
        # y03 = y_rows[3]
        # y12 = y_rows[6]
        # y13 = y_rows[7]
        
        # z02 = z_rows[2]
        # z03 = z_rows[3]
        # z12 = z_rows[6]
        # z13 = z_rows[7]
        
        # 8. M8 (M1 for Y2)
        U[x00, col_offset+7] += 1; U[x11, col_offset+7] += 1
        V[y02, col_offset+7] += 1; V[y13, col_offset+7] += 1
        W[z02, col_offset+7] += 1; W[z13, col_offset+7] += 1
        
        # 9. M9 (M2 for Y2)
        U[x10, col_offset+8] += 1; U[x11, col_offset+8] += 1
        V[y02, col_offset+8] += 1
        W[z12, col_offset+8] += 1
        
        # 10. M10 (M3 for Y2)
        U[x00, col_offset+9] += 1
        V[y03, col_offset+9] += 1; V[y13, col_offset+9] -= 1
        W[z03, col_offset+9] += 1
        
        # 11. M11 (M4 for Y2)
        U[x11, col_offset+10] += 1
        V[y12, col_offset+10] += 1; V[y02, col_offset+10] -= 1
        W[z02, col_offset+10] += 1; W[z12, col_offset+10] += 1
        
        # 12. M12 (M5 for Y2)
        U[x00, col_offset+11] += 1; U[x01, col_offset+11] += 1
        V[y13, col_offset+11] += 1
        W[z02, col_offset+11] -= 1; W[z03, col_offset+11] += 1
        
        # 13. M13 (M6 for Y2)
        U[x10, col_offset+12] += 1; U[x00, col_offset+12] -= 1
        V[y02, col_offset+12] += 1; V[y03, col_offset+12] += 1
        W[z13, col_offset+12] += 1
        
        # 14. M14 (M7 for Y2)
        U[x01, col_offset+13] += 1; U[x11, col_offset+13] -= 1
        V[y12, col_offset+13] += 1; V[y13, col_offset+13] += 1
        W[z02, col_offset+13] += 1; W[z13, col_offset+13] -= 1

    # Define blocks
    # A1: rows 0,1; cols 0,1. U indices:
    # A1[0,0] -> 0*4+0 = 0
    # A1[0,1] -> 0*4+1 = 1
    # A1[1,0] -> 1*4+0 = 4
    # A1[1,1] -> 1*4+1 = 5
    x_rows_A1 = [0, 1, 4, 5]
    
    # A2: rows 0,1; cols 2,3. U indices:
    # A2[0,0] (global A[0,2]) -> 0*4+2 = 2
    # A2[0,1] (global A[0,3]) -> 0*4+3 = 3
    # A2[1,0] (global A[1,2]) -> 1*4+2 = 6
    # A2[1,1] (global A[1,3]) -> 1*4+3 = 7
    x_rows_A2 = [2, 3, 6, 7]
    
    # B11: rows 0,1; cols 0-3. V indices:
    # (0,0)->0, (0,1)->1, (0,2)->2, (0,3)->3
    # (1,0)->8, (1,1)->9, (1,2)->10, (1,3)->11
    y_rows_B11 = [0, 1, 2, 3, 8, 9, 10, 11]
    
    # B12: rows 0,1; cols 4-7. V indices:
    # (0,4)->4, (0,5)->5, (0,6)->6, (0,7)->7
    # (1,4)->12, (1,5)->13, (1,6)->14, (1,7)->15
    y_rows_B12 = [4, 5, 6, 7, 12, 13, 14, 15]
    
    # B21: rows 2,3; cols 0-3. V indices:
    # (2,0)->16, (2,1)->17, (2,2)->18, (2,3)->19
    # (3,0)->24, (3,1)->25, (3,2)->26, (3,3)->27
    y_rows_B21 = [16, 17, 18, 19, 24, 25, 26, 27]
    
    # B22: rows 2,3; cols 4-7. V indices:
    # (2,4)->20, (2,5)->21, (2,6)->22, (2,7)->23
    # (3,4)->28, (3,5)->29, (3,6)->30, (3,7)->31
    y_rows_B22 = [20, 21, 22, 23, 28, 29, 30, 31]
    
    # C Left (cols 0-3):
    # C[0,0]->0*2+0=0, C[0,1]->1*2+0=2, C[0,2]->2*2+0=4, C[0,3]->3*2+0=6
    # C[1,0]->0*2+1=1, C[1,1]->1*2+1=3, C[1,2]->2*2+1=5, C[1,3]->3*2+1=7
    # W indices: k*n+i. k is col, i is row.
    # k=0, i=0 -> 0
    # k=1, i=0 -> 2
    # k=2, i=0 -> 4
    # k=3, i=0 -> 6
    # k=0, i=1 -> 1
    # k=1, i=1 -> 3
    # k=2, i=1 -> 5
    # k=3, i=1 -> 7
    z_rows_C_left = [0, 2, 4, 6, 1, 3, 5, 7]
    
    # C Right (cols 4-7):
    # k=4, i=0 -> 8
    # k=5, i=0 -> 10
    # k=6, i=0 -> 12
    # k=7, i=0 -> 14
    # k=4, i=1 -> 9
    # k=5, i=1 -> 11
    # k=6, i=1 -> 13
    # k=7, i=1 -> 15
    z_rows_C_right = [8, 10, 12, 14, 9, 11, 13, 15]
    
    # Term 1: A1 * B11 -> C_left
    add_strassen_224(0, x_rows_A1, y_rows_B11, z_rows_C_left)
    # Term 2: A2 * B21 -> C_left
    add_strassen_224(14, x_rows_A2, y_rows_B21, z_rows_C_left)
    # Term 3: A1 * B12 -> C_right
    add_strassen_224(28, x_rows_A1, y_rows_B12, z_rows_C_right)
    # Term 4: A2 * B22 -> C_right
    add_strassen_224(42, x_rows_A2, y_rows_B22, z_rows_C_right)
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
