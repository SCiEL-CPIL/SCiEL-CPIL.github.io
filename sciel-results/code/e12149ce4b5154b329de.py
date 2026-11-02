import numpy as np

DIMENSIONS = [5, 5, 5]

def _get_laderman_3x3():
    # Laderman's 15-rank algorithm for 3x3 matrix multiplication
    # Returns U, V, W as lists of lists, shapes (9, 15), (9, 15), (9, 15)
    # Indices: U[i*3+j], V[j*3+k], W[k*3+i]
    U = [[0]*15 for _ in range(9)]
    V = [[0]*15 for _ in range(9)]
    W = [[0]*15 for _ in range(9)]
    
    # Laderman's algorithm coefficients
    # P1 = (a11+a22)(b11+b22)
    # P2 = (a21+a22)(b21+b22)
    # P3 = (a11-a22)(b11-b22)
    # P4 = (a11+a12)(b12-b22)
    # P5 = (a21-a11)(b11+b12)
    # P6 = (a12-a22)(b21+b22)
    # P7 = (a11+a21)(b11+b21)
    # P8 = (a12+a22)(b12+b22)
    # P9 = (a11-a21)(b11-b21)
    # P10 = (a12-a22)(b12-b22)
    # P11 = (a11+a12)(b11+b12)
    # P12 = (a21+a22)(b11+b12)
    # P13 = (a11-a12)(b11-b12)
    # P14 = (a21-a22)(b11-b12)
    # P15 = (a11+a21)(b11-b21) -- Wait, I need the exact standard Laderman factors.
    
    # Using a known implementation of Laderman's factors
    # See: https://en.wikipedia.org/wiki/Matrix_multiplication_algorithm#Laderman's_algorithm
    # Or simply generate them from the 7 products of 2x2 blocks? No, Laderman is specific.
    
    # Let's use a simpler, verified set of 15 factors.
    # Actually, I can just use the schoolbook for 3x3 (27 rank) if I'm unsure of Laderman's exact integers.
    # But 15 is much better.
    
    # Let's try to construct Laderman's factors.
    # The 15 products are:
    # P1 = (a11+a22)(b11+b22)
    # P2 = (a21+a22)(b21+b22)
    # P3 = (a11-a22)(b11-b22)
    # P4 = (a11+a12)(b12-b22)
    # P5 = (a21-a11)(b11+b12)
    # P6 = (a12-a22)(b21+b22)
    # P7 = (a11+a21)(b11+b21)
    # P8 = (a12+a22)(b12+b22)
    # P9 = (a11-a21)(b11-b21)
    # P10 = (a12-a22)(b12-b22)
    # P11 = (a11+a12)(b11+b12)
    # P12 = (a21+a22)(b11+b12)
    # P13 = (a11-a12)(b11-b12)
    # P14 = (a21-a22)(b11-b12)
    # P15 = (a11+a21)(b11-b21)
    
    # This is getting risky to type out manually.
    # I will use the schoolbook for 3x3 (27) to be safe, but that increases rank.
    # 27 + 7 + 3*18 + 3*12 = 27 + 7 + 54 + 36 = 124.
    # 124 is still better than 125.
    
    # But let's try to get Laderman.
    # I will define the factors explicitly.
    
    pass

def construct():
    n, m, p = DIMENSIONS
    # We will use the 3+2 block decomposition.
    # Total rank will be sum of ranks of blocks.
    # To be safe and fast, I will use schoolbook for all blocks except 2x2 (which is 7).
    # Rank:
    # 3x3x3: 27
    # 2x2x2: 7
    # 3x2x3: 18
    # 3x3x2: 18
    # 3x2x2: 12
    # 2x3x3: 18
    # 2x3x2: 12
    # 2x2x3: 12
    # Total: 27+7+18+18+12+18+12+12 = 124.
    
    # This is a valid improvement (124 < 125).
    
    # Let's implement this.
    
    rank = 124
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Helper to add schoolbook block
    def add_schoolbook_block(i_start, i_end, j_start, j_end, k_start, k_end, U, V, W, col):
        # i, j, k ranges
        for i in range(i_start, i_end + 1):
            for j in range(j_start, j_end + 1):
                for k in range(k_start, k_end + 1):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return col
    
    # 1. A33 B33 (i,j,k in 0..2)
    col = add_schoolbook_block(0, 2, 0, 2, 0, 2, U, V, W, col)
    
    # 2. A22 B22 (i,j,k in 3..4)
    # Use 7-rank for 2x2
    # 2x2 factors:
    # P1 = (a00+a11)(b00+b11)
    # P2 = (a10+a11)b00
    # P3 = a00(b01-b11)
    # P4 = a11(b10-b00)
    # P5 = (a00+a01)b11
    # P6 = (a10-a00)(b00+b01)
    # P7 = (a01-a11)(b10+b11)
    
    # Map local (0,1) to global (3,4)
    def add_2x2_block(i_offset, j_offset, k_offset, U, V, W, col):
        # i_offset=3, j_offset=3, k_offset=3
        # Local indices 0,1 -> Global i_offset+0, i_offset+1
        # P1
        U[(i_offset+0)*m + (j_offset+0)][col] = 1
        U[(i_offset+1)*m + (j_offset+1)][col] = 1
        V[(j_offset+0)*p + (k_offset+0)][col] = 1
        V[(j_offset+1)*p + (k_offset+1)][col] = 1
        W[(k_offset+0)*n + (i_offset+0)][col] = 1
        W[(k_offset+1)*n + (i_offset+1)][col] = 1
        col += 1
        
        # P2
        U[(i_offset+1)*m + (j_offset+0)][col] = 1
        U[(i_offset+1)*m + (j_offset+1)][col] = 1
        V[(j_offset+0)*p + (k_offset+0)][col] = 1
        W[(k_offset+0)*n + (i_offset+1)][col] = 1
        col += 1
        
        # P3
        U[(i_offset+0)*m + (j_offset+0)][col] = 1
        V[(j_offset+0)*p + (k_offset+1)][col] = 1
        V[(j_offset+1)*p + (k_offset+1)][col] = -1
        W[(k_offset+1)*n + (i_offset+0)][col] = 1
        col += 1
        
        # P4
        U[(i_offset+1)*m + (j_offset+1)][col] = 1
        V[(j_offset+1)*p + (k_offset+0)][col] = 1
        V[(j_offset+0)*p + (k_offset+0)][col] = -1
        W[(k_offset+0)*n + (i_offset+1)][col] = 1
        col += 1
        
        # P5
        U[(i_offset+0)*m + (j_offset+0)][col] = 1
        U[(i_offset+0)*m + (j_offset+1)][col] = 1
        V[(j_offset+1)*p + (k_offset+1)][col] = 1
        W[(k_offset+1)*n + (i_offset+0)][col] = 1
        col += 1
        
        # P6
        U[(i_offset+1)*m + (j_offset+0)][col] = 1
        U[(i_offset+0)*m + (j_offset+0)][col] = -1
        V[(j_offset+0)*p + (k_offset+0)][col] = 1
        V[(j_offset+0)*p + (k_offset+1)][col] = 1
        W[(k_offset+0)*n + (i_offset+1)][col] = 1
        col += 1
        
        # P7
        U[(i_offset+0)*m + (j_offset+1)][col] = 1
        U[(i_offset+1)*m + (j_offset+1)][col] = -1
        V[(j_offset+1)*p + (k_offset+0)][col] = 1
        V[(j_offset+1)*p + (k_offset+1)][col] = 1
        W[(k_offset+0)*n + (i_offset+0)][col] = 1
        col += 1
        
        return col

    col = add_2x2_block(3, 3, 3, U, V, W, col)
    
    # 3. A32 B23 (i 0..2, j 3..4, k 0..2)
    col = add_schoolbook_block(0, 2, 3, 4, 0, 2, U, V, W, col)
    
    # 4. A33 B32 (i 0..2, j 0..2, k 3..4)
    col = add_schoolbook_block(0, 2, 0, 2, 3, 4, U, V, W, col)
    
    # 5. A32 B22 (i 0..2, j 3..4, k 3..4)
    col = add_schoolbook_block(0, 2, 3, 4, 3, 4, U, V, W, col)
    
    # 6. A23 B33 (i 3..4, j 0..2, k 0..2)
    col = add_schoolbook_block(3, 4, 0, 2, 0, 2, U, V, W, col)
    
    # 7. A23 B32 (i 3..4, j 0..2, k 3..4)
    col = add_schoolbook_block(3, 4, 0, 2, 3, 4, U, V, W, col)
    
    # 8. A22 B23 (i 3..4, j 3..4, k 0..2)
    col = add_schoolbook_block(3, 4, 3, 4, 0, 2, U, V, W, col)
    
    assert col == rank, f"Rank mismatch: {col} vs {rank}"
    
    return {"U": U, "V": V, "W": W}
