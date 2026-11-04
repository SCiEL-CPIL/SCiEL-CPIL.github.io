"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 9]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 4, 4, 9
    
    # Use rank-49 Strassen-based decomposition for each 4x4x4 block
    # B is 4x9, so we have two 4x4 blocks and one 4x1 column
    # For 4x1: rank 4 (each row of A times the single B column)
    # Total: 49 + 49 + 4 = 102
    
    rank = 102
    U = [[0] * rank for _ in range(n * m)]  # 16 x 102
    V = [[0] * rank for _ in range(m * p)]  # 36 x 102
    W = [[0] * rank for _ in range(p * n)]  # 36 x 102
    
    col = 0
    
    def add_2x2x2_strassen(U, V, W, a_rows, b_rows, c_rows, c_start):
        """
        Add rank-7 Strassen for 2x2x2.
        a_rows: [row for A[0][0], A[0][1], A[1][0], A[1][1]]
        b_rows: [row for B[0][0], B[0][1], B[1][0], B[1][1]]
        c_rows: [row for C[0][0], C[0][1], C[1][0], C[1][1]]
        """
        c = c_start
        # M1 = (A11+A22)(B11+B22)
        U[a_rows[0]][c] += 1; U[a_rows[3]][c] += 1
        V[b_rows[0]][c] += 1; V[b_rows[3]][c] += 1
        W[c_rows[0]][c] += 1; W[c_rows[3]][c] += 1
        c += 1
        # M2 = (A21+A22)B11
        U[a_rows[2]][c] += 1; U[a_rows[3]][c] += 1
        V[b_rows[0]][c] += 1
        W[c_rows[2]][c] += 1; W[c_rows[3]][c] += -1
        c += 1
        # M3 = A11(B21-B22)
        U[a_rows[0]][c] += 1
        V[b_rows[2]][c] += 1; V[b_rows[3]][c] += -1
        W[c_rows[1]][c] += 1; W[c_rows[3]][c] += 1
        c += 1
        # M4 = A22(B12-B11)
        U[a_rows[3]][c] += 1
        V[b_rows[1]][c] += 1; V[b_rows[0]][c] += -1
        W[c_rows[0]][c] += 1; W[c_rows[2]][c] += 1
        c += 1
        # M5 = (A11+A12)B22
        U[a_rows[0]][c] += 1; U[a_rows[1]][c] += 1
        V[b_rows[3]][c] += 1
        W[c_rows[0]][c] += -1; W[c_rows[1]][c] += 1
        c += 1
        # M6 = (A21-A11)(B11+B12)
        U[a_rows[2]][c] += 1; U[a_rows[0]][c] += -1
        V[b_rows[0]][c] += 1; V[b_rows[1]][c] += 1
        W[c_rows[3]][c] += 1
        c += 1
        # M7 = (A12-A22)(B21+B22)
        U[a_rows[1]][c] += 1; U[a_rows[3]][c] += -1
        V[b_rows[2]][c] += 1; V[b_rows[3]][c] += 1
        W[c_rows[0]][c] += 1
        c += 1
        return c
    
    def add_4x4x4_rank49(U, V, W, a_off, b_off, c_off, c_start):
        """Add rank-49 decomposition for 4x4x4 using Strassen at both levels."""
        c = c_start
        
        # A block row indices for 4x4 A: A[i][j] at U[a_off + i*4 + j]
        # A11 = [[A00,A01],[A10,A11]] -> rows [a_off+0, a_off+1, a_off+4, a_off+5]
        # A12 = [[A02,A03],[A12,A13]] -> rows [a_off+2, a_off+3, a_off+6, a_off+7]
        # A21 = [[A20,A21],[A30,A31]] -> rows [a_off+8, a_off+9, a_off+12, a_off+13]
        # A22 = [[A22,A23],[A32,A33]] -> rows [a_off+10, a_off+11, a_off+14, a_off+15]
        
        A11 = [a_off+0, a_off+1, a_off+4, a_off+5]
        A12 = [a_off+2, a_off+3, a_off+6, a_off+7]
        A21 = [a_off+8, a_off+9, a_off+12, a_off+13]
        A22 = [a_off+10, a_off+11, a_off+14, a_off+15]
        
        B11 = [b_off+0, b_off+1, b_off+4, b_off+5]
        B12 = [b_off+2, b_off+3, b_off+6, b_off+7]
        B21 = [b_off+8, b_off+9, b_off+12, b_off+13]
        B22 = [b_off+10, b_off+11, b_off+14, b_off+15]
        
        C11 = [c_off+0, c_off+1, c_off+4, c_off+5]
        C12 = [c_off+2, c_off+3, c_off+6, c_off+7]
        C21 = [c_off+8, c_off+9, c_off+12, c_off+13]
        C22 = [c_off+10, c_off+11, c_off+14, c_off+15]
        
        # Strassen formulas:
        # M1 = (A11+A22)(B11+B22)
        # M2 = (A21+A22)B11
        # M3 = A11(B21-B22)
        # M4 = A22(B12-B11)
        # M5 = (A11+A12)B22
        # M6 = (A21-A11)(B11+B12)
        # M7 = (A12-A22)(B21+B22)
        
        # C11 = M1 + M4 - M5 + M7
        # C12 = M3 + M5
        # C21 = M2 + M4
        # C22 = M1 - M2 + M3 + M6
        
        # For each Mi, we need to compute a 2x2 matrix product with linear combinations.
        # We'll add 7 Strassen sub-products for each Mi, with appropriate coefficients.
        
        # Helper to add a 2x2x2 Strassen with scaled inputs
        def add_scaled_2x2x2_strassen(U, V, W, a_coeffs, b_coeffs, c_rows, c_start):
            """
            a_coeffs: dict mapping (row_idx) -> coeff for A block
            b_coeffs: dict mapping (row_idx) -> coeff for B block
            c_rows: [row for C[0][0], C[0][1], C[1][0], C[1][1]]
            """
            c = c_start
            # Get the 4 A rows and their coefficients
            a00 = a_coeffs.get(0, 0)
            a01 = a_coeffs.get(1, 0)
            a10 = a_coeffs.get(2, 0)
            a11 = a_coeffs.get(3, 0)
            b00 = b_coeffs.get(0, 0)
            b01 = b_coeffs.get(1, 0)
            b10 = b_coeffs.get(2, 0)
            b11 = b_coeffs.get(3, 0)
            
            # We need actual row indices. Let's pass them differently.
            # This approach is getting too complex. Let me use a different method.
            return c
        
        # Given the extreme complexity, let me just do schoolbook for each 2x2x2.
        # Schoolbook 2x2x2 has rank 8 (not 7). So 7*8 = 56 per 4x4x4.
        # Total: 56 + 56 + 4 = 116. Still better than 144.
        
        def add_2x2x2_schoolbook(U, V, W, a_rows, b_rows, c_rows, c_start):
            """Add rank-8 schoolbook for 2x2x2."""
            c = c_start
            # C[0][0] = A[0][0]*B[0][0] + A[0][1]*B[1][0]
            # C[0][1] = A[0][0]*B[0][1] + A[0][1]*B[1][1]
            # C[1][0] = A[1][0]*B[0][0] + A[1][1]*B[1][0]
            # C[1][1] = A[1][0]*B[0][1] + A[1][1]*B[1][1]
            for i in range(2):
                for j in range(2):
                    for k in range(2):
                        U[a_rows[i*2+k]][c] += 1
                        V[b_rows[k*2+j]][c] += 1
                        W[c_rows[i*2+j]][c] += 1
                        c += 1
            return c
        
        # Now use schoolbook for each of the 7 Strassen products
        # M1 = (A11+A22)(B11+B22)
        # This expands to: A11*B11 + A11*B22 + A22*B11 + A22*B22
        # Each contributes to C11 and C22
        
        # A11*B11 -> C11
        c = add_2x2x2_schoolbook(U, V, W, A11, B11, C11, c)
        # A11*B22 -> C22
        c = add_2x2x2_schoolbook(U, V, W, A11, B22, C22, c)
        # A22*B11 -> C11
        c = add_2x2x2_schoolbook(U, V, W, A22, B11, C11, c)
        # A22*B22 -> C22
        c = add_2x2x2_schoolbook(U, V, W, A22, B22, C22, c)
        
        # M2 = (A21+A22)B11 = A21*B11 + A22*B11
        # A21*B11 -> C21
        c = add_2x2x2_schoolbook(U, V, W, A21, B11, C21, c)
        # A22*B11 -> C22
        c = add_2x2x2_schoolbook(U, V, W, A22, B11, C22, c)
        
        # M3 = A11(B21-B22) = A11*B21 - A11*B22
        # A11*B21 -> C12
        c = add_2x2x2_schoolbook(U, V, W, A11, B21, C12, c)
        # -A11*B22 -> C12
        c = add_2x2x2_schoolbook(U, V, W, A11, B22, C12, c)
        # Need to negate: multiply W contributions by -1
        # This is getting messy. Let me just accept this won't work cleanly.
        
        return c
    
    # This is too error-prone. Let me just use schoolbook rank 144.
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
# EVOLVE-BLOCK-END
