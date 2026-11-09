"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Rank 42 construction based on Strassen's algorithm for 2x2x2 blocks
    rank = 42
    U = [[0] * rank for _ in range(n * m)]  # 8 x 42
    V = [[0] * rank for _ in range(m * p)]  # 24 x 42
    W = [[0] * rank for _ in range(p * n)]  # 12 x 42
    
    col = 0
    
    # Helper to add a Strassen 2x2x2 block
    def add_strassen_block(a_row_offset, b_col_offset, c_col_offset, start_col):
        # a_row_offset: offset in A (0 or 2, since A is 2x4, rows are 0,1. Wait.
        # A is 2x4.
        # We split A into A1 (cols 0,1) and A2 (cols 2,3).
        # A1 is 2x2. Rows 0,1. Cols 0,1.
        # A2 is 2x2. Rows 0,1. Cols 2,3.
        
        # B is 4x6.
        # We split B into B1 (rows 0,1) and B2 (rows 2,3).
        # B1 is 2x6.
        # B2 is 2x6.
        
        # C is 2x6.
        # C1 = A1 B1.
        # C2 = A2 B2.
        # C = C1 + C2.
        
        # So we need to add 2 blocks of M_2,2,6.
        pass

    # We will manually unroll the two M_2,2,6 blocks.
    
    # Block 1: A1 (2x2) * B1 (2x6)
    # A1 indices in global A:
    # A1[0,0] -> U[0, col]
    # A1[0,1] -> U[1, col]
    # A1[1,0] -> U[4, col]  (Row 1, Col 0 of A1? No. A is 2x4.
    # A[i, j].
    # A1 is A[0:2, 0:2].
    # A1[0,0] is A[0,0] -> U[0]
    # A1[0,1] is A[0,1] -> U[1]
    # A1[1,0] is A[1,0] -> U[4]
    # A1[1,1] is A[1,1] -> U[5]
    
    # B1 is B[0:2, :]. B is 4x6.
    # B1[0,0] is B[0,0] -> V[0]
    # B1[0,1] is B[0,1] -> V[1]
    # B1[1,0] is B[1,0] -> V[6]
    # B1[1,1] is B[1,1] -> V[7]
    
    # C1 is C[0:2, :]. C is 2x6.
    # C1[0,0] is C[0,0] -> W[0]
    # C1[0,1] is C[0,1] -> W[1]
    # C1[1,0] is C[1,0] -> W[2]
    # C1[1,1] is C[1,1] -> W[3]
    
    # We split B1 (2x6) into three 2x2 blocks.
    # B1_k = B[0:2, 2*k : 2*k+2]
    # For k=0,1,2.
    
    for block_id in range(2): # Two M_2,2,6 blocks
        # Determine A and B offsets
        if block_id == 0:
            # A1: A[0:2, 0:2]
            a_col_base = 0
            # B1: B[0:2, :]
            b_row_base = 0
        else:
            # A2: A[0:2, 2:4]
            a_col_base = 2
            # B2: B[2:4, :]
            b_row_base = 2
            
        # Now decompose M_2,2,6 using 3 Strassen blocks
        for k in range(3):
            # B_k cols are 2*k and 2*k+1
            b_col_base = 2 * k
            
            # Strassen coefficients
            # M1 = A00 (B00 + B11)
            U[a_col_base + 0][col] = 1 # A00 is A[0, a_col_base]
            # Wait, A indices:
            # A1[0,0] is global A[0,0]. U[0].
            # A1[0,1] is global A[0,1]. U[1].
            # A1[1,0] is global A[1,0]. U[4].
            # A1[1,1] is global A[1,1]. U[5].
            
            # So:
            # A00 -> U[a_row_base + 0 + a_col_base] ?
            # A is 2x4. U index = i*m + j.
            # A1[0,0] -> i=0, j=0 -> U[0].
            # A1[0,1] -> i=0, j=1 -> U[1].
            # A1[1,0] -> i=1, j=0 -> U[4].
            # A1[1,1] -> i=1, j=1 -> U[5].
            
            # For block 1 (a_col_base=0):
            # A00: U[0]
            # A01: U[1]
            # A10: U[4]
            # A11: U[5]
            
            # For block 2 (a_col_base=2):
            # A00: U[0*4+2] = U[2]
            # A01: U[0*4+3] = U[3]
            # A10: U[1*4+2] = U[6]
            # A11: U[1*4+3] = U[7]
            
            # B indices:
            # V index = j*p + k.
            # B1[0,0] -> j=0, k=0 -> V[0].
            # B1[0,1] -> j=0, k=1 -> V[1].
            # B1[1,0] -> j=1, k=0 -> V[6].
            # B1[1,1] -> j=1, k=1 -> V[7].
            
            # For block 1 (b_row_base=0, b_col_base=2k):
            # B00: V[0*6 + 2k]
            # B01: V[0*6 + 2k+1]
            # B10: V[1*6 + 2k]
            # B11: V[1*6 + 2k+1]
            
            # For block 2 (b_row_base=2, b_col_base=2k):
            # B00: V[2*6 + 2k]
            # B01: V[2*6 + 2k+1]
            # B10: V[3*6 + 2k]
            # B11: V[3*6 + 2k+1]
            
            # C indices:
            # W index = k*n + i.
            # C is 2x6.
            # C1[0,0] -> i=0, k=0 -> W[0].
            # C1[0,1] -> i=0, k=1 -> W[1].
            # C1[1,0] -> i=1, k=0 -> W[2].
            # C1[1,1] -> i=1, k=1 -> W[3].
            
            # For block k (cols 2k, 2k+1):
            # C00: W[2k*2 + 0] = W[4k]
            # C01: W[2k*2 + 1] = W[4k+1]
            # C10: W[(2k+1)*2 + 0] = W[4k+2]
            # C11: W[(2k+1)*2 + 1] = W[4k+3]
            
            # Define local indices
            u00 = a_col_base # i=0, j=a_col_base
            u01 = a_col_base + 1
            u10 = 4 + a_col_base # i=1, j=a_col_base
            u11 = 4 + a_col_base + 1
            
            v00 = b_row_base * 6 + b_col_base
            v01 = b_row_base * 6 + b_col_base + 1
            v10 = (b_row_base + 1) * 6 + b_col_base
            v11 = (b_row_base + 1) * 6 + b_col_base + 1
            
            w00 = b_col_base * 2 # i=0, k=b_col_base
            w01 = b_col_base * 2 + 1
            w10 = (b_col_base + 1) * 2
            w11 = (b_col_base + 1) * 2 + 1
            
            # M1 = A00 (B00 + B11)
            U[u00][col] += 1
            V[v00][col] += 1
            V[v11][col] += 1
            # W coeffs for M1: C00+=1, C11+=1
            W[w00][col] += 1
            W[w11][col] += 1
            col += 1
            
            # M2 = (A00 + A10) B11
            U[u00][col] += 1
            U[u10][col] += 1
            V[v11][col] += 1
            # W coeffs for M2: C00-=1, C11+=1
            W[w00][col] -= 1
            W[w11][col] += 1
            col += 1
            
            # M3 = (A01 + A11) B00
            U[u01][col] += 1
            U[u11][col] += 1
            V[v00][col] += 1
            # W coeffs for M3: C01+=1, C11-=1
            W[w01][col] += 1
            W[w11][col] -= 1
            col += 1
            
            # M4 = A11 (B11 - B00)
            U[u11][col] += 1
            V[v11][col] += 1
            V[v00][col] -= 1
            # W coeffs for M4: C00+=1, C11+=1
            W[w00][col] += 1
            W[w11][col] += 1
            col += 1
            
            # M5 = (A00 + A01) B10
            U[u00][col] += 1
            U[u01][col] += 1
            V[v10][col] += 1
            # W coeffs for M5: C10+=1, C11+=1
            W[w10][col] += 1
            W[w11][col] += 1
            col += 1
            
            # M6 = (A10 - A00) (B00 + B01)
            U[u10][col] += 1
            U[u00][col] -= 1
            V[v00][col] += 1
            V[v01][col] += 1
            # W coeffs for M6: C00+=1
            W[w00][col] += 1
            col += 1
            
            # M7 = (A01 - A11) (B10 + B11)
            U[u01][col] += 1
            U[u11][col] -= 1
            V[v10][col] += 1
            V[v11][col] += 1
            # W coeffs for M7: C01+=1, C10+=1
            W[w01][col] += 1
            W[w10][col] += 1
            col += 1
            
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
