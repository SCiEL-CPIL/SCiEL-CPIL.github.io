"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 9]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 4, 4, 9
    
    # Use Strassen-based decomposition for 4x4x4 (rank 28) for first 8 columns
    # and schoolbook (rank 16) for the last column.
    # Total rank: 28 + 28 + 16 = 72
    
    rank = 72
    U = [[0] * rank for _ in range(n * m)]  # 16 x 72
    V = [[0] * rank for _ in range(m * p)]  # 36 x 72
    W = [[0] * rank for _ in range(p * n)]  # 36 x 72
    
    def add_strassen_2x2(U, V, W, a_row_offset, b_row_offset, c_row_offset, col_start):
        """
        Add Strassen 2x2x2 decomposition.
        a_row_offset: row offset in U for the 2x2 A block (4 rows)
        b_row_offset: row offset in V for the 2x2 B block (4 rows)
        c_row_offset: row offset in W for the 2x2 C block (4 rows)
        col_start: starting column index
        Returns: new column index
        """
        c = col_start
        # M1 = (A11+A22)(B11+B22)
        U[a_row_offset + 0][c] = 1
        U[a_row_offset + 3][c] = 1
        V[b_row_offset + 0][c] = 1
        V[b_row_offset + 3][c] = 1
        W[c_row_offset + 0][c] = 1   # C11
        W[c_row_offset + 3][c] = 1   # C22
        c += 1
        
        # M2 = (A21+A22)B11
        U[a_row_offset + 2][c] = 1
        U[a_row_offset + 3][c] = 1
        V[b_row_offset + 0][c] = 1
        W[c_row_offset + 2][c] = 1   # C21
        W[c_row_offset + 3][c] = -1  # C22
        c += 1
        
        # M3 = A11(B21-B22)
        U[a_row_offset + 0][c] = 1
        V[b_row_offset + 2][c] = 1
        V[b_row_offset + 3][c] = -1
        W[c_row_offset + 1][c] = 1   # C12
        W[c_row_offset + 3][c] = 1   # C22
        c += 1
        
        # M4 = A22(B12-B11)
        U[a_row_offset + 3][c] = 1
        V[b_row_offset + 1][c] = 1
        V[b_row_offset + 0][c] = -1
        W[c_row_offset + 0][c] = 1   # C11
        W[c_row_offset + 2][c] = 1   # C21
        c += 1
        
        # M5 = (A11+A12)B22
        U[a_row_offset + 0][c] = 1
        U[a_row_offset + 1][c] = 1
        V[b_row_offset + 3][c] = 1
        W[c_row_offset + 0][c] = -1  # C11
        W[c_row_offset + 1][c] = 1   # C12
        c += 1
        
        # M6 = (A21-A11)(B11+B12)
        U[a_row_offset + 2][c] = 1
        U[a_row_offset + 0][c] = -1
        V[b_row_offset + 0][c] = 1
        V[b_row_offset + 1][c] = 1
        W[c_row_offset + 3][c] = 1   # C22
        c += 1
        
        # M7 = (A12-A22)(B21+B22)
        U[a_row_offset + 1][c] = 1
        U[a_row_offset + 3][c] = -1
        V[b_row_offset + 2][c] = 1
        V[b_row_offset + 3][c] = 1
        W[c_row_offset + 0][c] = 1   # C11
        c += 1
        
        return c
    
    def add_4x4x4_strassen(U, V, W, a_offset, b_offset, c_offset, col_start):
        """
        Add 28-rank Strassen-based decomposition for 4x4x4.
        a_offset: row offset in U for the 4x4 A block (16 rows)
        b_offset: row offset in V for the 4x4 B block (16 rows)
        c_offset: row offset in W for the 4x4 C block (16 rows)
        col_start: starting column
        Returns: new column
        """
        c = col_start
        
        # For 4x4 matrix A stored with row index = i*4 + j:
        # A11 = A[0:2, 0:2] -> indices 0,1,4,5
        # A12 = A[0:2, 2:4] -> indices 2,3,6,7
        # A21 = A[2:4, 0:2] -> indices 8,9,12,13
        # A22 = A[2:4, 2:4] -> indices 10,11,14,15
        
        # Strassen 2x2 block multiplication:
        # M1 = (A11+A22)(B11+B22)
        # M2 = (A21+A22)B11
        # M3 = A11(B21-B22)
        # M4 = A22(B12-B11)
        # M5 = (A11+A12)B22
        # M6 = (A21-A11)(B11+B12)
        # M7 = (A12-A22)(B21+B22)
        
        # Each Mi is a 2x2 matrix product, computed with schoolbook (4 scalar mults)
        
        # M1 = (A11+A22)(B11+B22)
        # (A11+A22) has entries at A11 and A22 positions
        # We need to compute (A11+A22) * (B11+B22)
        # This is a 2x2 * 2x2 product
        
        # For product (A11+A22)(B11+B22):
        # A11+A22: 
        #   [A11[0,0]+A22[0,0], A11[0,1]+A22[0,1]]
        #   [A11[1,0]+A22[1,0], A11[1,1]+A22[1,1]]
        # In U indices (relative to a_offset):
        #   A11[0,0]=0, A11[0,1]=1, A11[1,0]=4, A11[1,1]=5
        #   A22[0,0]=10, A22[0,1]=11, A22[1,0]=14, A22[1,1]=15
        
        # B11+B22:
        #   B11[0,0]=0, B11[0,1]=1, B11[1,0]=4, B11[1,1]=5
        #   B22[0,0]=10, B22[0,1]=11, B22[1,0]=14, B22[1,1]=15
        
        # We compute 4 scalar products for this 2x2*2x2:
        # P1 = (A11[0,0]+A22[0,0])*(B11[0,0]+B22[0,0])
        #     = (A[0,0]+A[2,2])*(B[0,0]+B[2,2])
        # P2 = (A11[0,0]+A22[0,0])*(B11[0,1]+B22[0,1])
        #     = (A[0,0]+A[2,2])*(B[0,1]+B[2,3])
        # P3 = (A11[1,0]+A22[1,0])*(B11[0,0]+B22[0,0])
        #     = (A[1,0]+A[3,2])*(B[0,0]+B[2,2])
        # P4 = (A11[1,0]+A22[1,0])*(B11[0,1]+B22[0,1])
        #     = (A[1,0]+A[3,2])*(B[0,1]+B[2,3])
        # P5 = (A11[0,1]+A22[0,1])*(B11[1,0]+B22[1,0])
        #     = (A[0,1]+A[2,3])*(B[1,0]+B[3,2])
        # P6 = (A11[0,1]+A22[0,1])*(B11[1,1]+B22[1,1])
        #     = (A[0,1]+A[2,3])*(B[1,1]+B[3,3])
        # P7 = (A11[1,1]+A22[1,1])*(B11[1,0]+B22[1,0])
        #     = (A[1,1]+A[3,3])*(B[1,0]+B[3,2])
        # P8 = (A11[1,1]+A22[1,1])*(B11[1,1]+B22[1,1])
        #     = (A[1,1]+A[3,3])*(B[1,1]+B[3,3])
        
        # Wait, that's 8 products, not 4. I made an error.
        # For 2x2*2x2 schoolbook, we need 4 products:
        # C[0,0] = A[0,0]*B[0,0] + A[0,1]*B[1,0]
        # C[0,1] = A[0,0]*B[0,1] + A[0,1]*B[1,1]
        # C[1,0] = A[1,0]*B[0,0] + A[1,1]*B[1,0]
        # C[1,1] = A[1,0]*B[0,1] + A[1,1]*B[1,1]
        
        # So for (A11+A22)(B11+B22), the 4 scalar products are:
        # s1 = (A11+A22)[0,0] * (B11+B22)[0,0]
        # s2 = (A11+A22)[0,0] * (B11+B22)[1,0]  -- wait no
        # Actually for 2x2 schoolbook:
        # q1 = A[0,0]*B[0,0]
        # q2 = A[0,1]*B[1,0]
        # q3 = A[1,0]*B[0,0]
        # q4 = A[1,1]*B[1,0]
        # q5 = A[0,0]*B[0,1]
        # q6 = A[0,1]*B[1,1]
        # q7 = A[1,0]*B[0,1]
        # q8 = A[1,1]*B[1,1]
        
        # That's 8 products for 2x2*2x2 schoolbook! Not 4.
        # I was confused. 2x2*2x2 schoolbook has rank 4? No, it has rank 4 only if we use Strassen.
        # Schoolbook 2x2*2x2 has rank 4? Let me reconsider.
        
        # Actually, 2x2*2x2 matrix multiplication has rank 7 (Strassen) or 4 (schoolbook uses 4 multiplications? No!)
        # Schoolbook 2x2*2x2: C[i,j] = sum_k A[i,k]*B[k,j], so 4*2 = 8 scalar multiplications.
        # But we can do better. The rank of 2x2*2x2 is 7 (Strassen).
        
        # So my earlier claim of rank 28 for 4x4x4 was wrong.
        # If each 2x2*2x2 has rank 7, then 7*7 = 49 for 4x4x4.
        
        # Let me just use schoolbook for everything to be safe.
        # 4x4*4x9 schoolbook: rank = 4*4*9 = 144
        
        return c
    
    # Given the complexity, let me just use schoolbook.
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
