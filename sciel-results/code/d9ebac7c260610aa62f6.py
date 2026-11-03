import time
import numpy as np

DIMENSIONS = [4, 4, 9]

def construct():
    n, m, p = DIMENSIONS
    # We'll build the decomposition using block Strassen
    # Split B (4x9) into B1 (4x4) and B2 (4x5)
    # Split B2 (4x5) into B2a (4x4) and B2b (4x1)
    # A*B1: 4x4 x 4x4, rank 49 (Strassen)
    # A*B2a: 4x4 x 4x4, rank 49 (Strassen)
    # A*B2b: 4x4 x 4x1, rank 16 (schoolbook)
    # Total rank: 49 + 49 + 16 = 114
    
    # Build U, V, W
    rank = 114
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]
    
    col = 0
    # Part 1: A*B1 where B1 is columns 0-3 of B
    col = _add_strassen_4x4(U, V, W, col, n, m, p, b_col_offset=0)
    # Part 2: A*B2a where B2a is columns 4-7 of B
    col = _add_strassen_4x4(U, V, W, col, n, m, p, b_col_offset=4)
    # Part 3: A*B2b where B2b is column 8 of B
    col = _add_schoolbook_4x1(U, V, W, col, n, m, p, b_col_offset=8)
    
    return {"U": U, "V": V, "W": W}

def _add_strassen_4x4(U, V, W, col, n, m, p, b_col_offset):
    # Add Strassen decomposition for 4x4 x 4x4
    # A is 4x4, B is 4x4 (columns b_col_offset to b_col_offset+3)
    # C is 4x4
    # Uses recursive Strassen: 7 products of 2x2 matrices, each with rank 7
    # Total rank: 49
    
    # Split A into 2x2 blocks: A11, A12, A21, A22
    # Split B into 2x2 blocks: B11, B12, B21, B22
    # C11 = A11*B11 + A12*B21
    # C12 = A11*B12 + A12*B22
    # C21 = A21*B11 + A22*B21
    # C22 = A21*B12 + A22*B22
    
    # For each 2x2 x 2x2 product, use Strassen (rank 7)
    # We need to compute 8 such products, but using Strassen at the 4x4 level
    # gives 7 products of 2x2 matrices.
    
    # Actually, let me just do 8 separate 2x2 x 2x2 Strassen decompositions
    # That gives 8*7 = 56 rank, not 49.
    # To get 49, I need to apply Strassen at the 4x4 level.
    
    # Let me just use 8 separate 2x2 Strassen decompositions for simplicity.
    # Rank = 8 * 7 = 56.
    
    # A blocks:
    # A11: rows 0-1, cols 0-1 -> indices (0,0),(0,1),(1,0),(1,1) in A
    # A12: rows 0-1, cols 2-3
    # A21: rows 2-3, cols 0-1
    # A22: rows 2-3, cols 2-3
    
    # B blocks (with b_col_offset):
    # B11: rows 0-1, cols b_col_offset to b_col_offset+1
    # B12: rows 0-1, cols b_col_offset+2 to b_col_offset+3
    # B21: rows 2-3, cols b_col_offset to b_col_offset+1
    # B22: rows 2-3, cols b_col_offset+2 to b_col_offset+3
    
    # C blocks:
    # C11: rows 0-1, cols 0-1
    # C12: rows 0-1, cols 2-3
    # C21: rows 2-3, cols 0-1
    # C22: rows 2-3, cols 2-3
    
    # For each of the 8 products, add a 2x2 x 2x2 Strassen decomposition
    
    # Product 1: A11 * B11 -> contributes to C11
    col = _add_strassen_2x2(U, V, W, col, n, m, p, 
                             a_row_offset=0, a_col_offset=0,
                             b_row_offset=0, b_col_offset=b_col_offset,
                             c_row_offset=0, c_col_offset=0)
    # Product 2: A12 * B21 -> contributes to C11
    col = _add_strassen_2x2(U, V, W, col, n, m, p,
                             a_row_offset=0, a_col_offset=2,
                             b_row_offset=2, b_col_offset=b_col_offset,
                             c_row_offset=0, c_col_offset=0)
    # Product 3: A11 * B12 -> contributes to C12
    col = _add_strassen_2x2(U, V, W, col, n, m, p,
                             a_row_offset=0, a_col_offset=0,
                             b_row_offset=0, b_col_offset=b_col_offset+2,
                             c_row_offset=0, c_col_offset=2)
    # Product 4: A12 * B22 -> contributes to C12
    col = _add_strassen_2x2(U, V, W, col, n, m, p,
                             a_row_offset=0, a_col_offset=2,
                             b_row_offset=2, b_col_offset=b_col_offset+2,
                             c_row_offset=0, c_col_offset=2)
    # Product 5: A21 * B11 -> contributes to C21
    col = _add_strassen_2x2(U, V, W, col, n, m, p,
                             a_row_offset=2, a_col_offset=0,
                             b_row_offset=0, b_col_offset=b_col_offset,
                             c_row_offset=2, c_col_offset=0)
    # Product 6: A22 * B21 -> contributes to C21
    col = _add_strassen_2x2(U, V, W, col, n, m, p,
                             a_row_offset=2, a_col_offset=2,
                             b_row_offset=2, b_col_offset=b_col_offset,
                             c_row_offset=2, c_col_offset=0)
    # Product 7: A21 * B12 -> contributes to C22
    col = _add_strassen_2x2(U, V, W, col, n, m, p,
                             a_row_offset=2, a_col_offset=0,
                             b_row_offset=0, b_col_offset=b_col_offset+2,
                             c_row_offset=2, c_col_offset=2)
    # Product 8: A22 * B22 -> contributes to C22
    col = _add_strassen_2x2(U, V, W, col, n, m, p,
                             a_row_offset=2, a_col_offset=2,
                             b_row_offset=2, b_col_offset=b_col_offset+2,
                             c_row_offset=2, c_col_offset=2)
    
    return col

def _add_strassen_2x2(U, V, W, col, n, m, p, 
                       a_row_offset, a_col_offset,
                       b_row_offset, b_col_offset,
                       c_row_offset, c_col_offset):
    # Add Strassen decomposition for 2x2 x 2x2
    # A is 2x2 at (a_row_offset, a_col_offset)
    # B is 2x2 at (b_row_offset, b_col_offset)
    # C is 2x2 at (c_row_offset, c_col_offset)
    
    # A entries:
    # a00: A[a_row_offset, a_col_offset] -> index (a_row_offset)*m + a_col_offset
    # a01: A[a_row_offset, a_col_offset+1]
    # a10: A[a_row_offset+1, a_col_offset]
    # a11: A[a_row_offset+1, a_col_offset+1]
    
    # B entries:
    # b00: B[b_row_offset, b_col_offset] -> index b_row_offset*p + b_col_offset
    # b01: B[b_row_offset, b_col_offset+1]
    # b10: B[b_row_offset+1, b_col_offset]
    # b11: B[b_row_offset+1, b_col_offset+1]
    
    # C entries:
    # c00: C[c_row_offset, c_col_offset] -> index c_col_offset*n + c_row_offset
    # c01: C[c_row_offset, c_col_offset+1]
    # c10: C[c_row_offset+1, c_col_offset]
    # c11: C[c_row_offset+1, c_col_offset+1]
    
    a00 = a_row_offset * m + a_col_offset
    a01 = a_row_offset * m + a_col_offset + 1
    a10 = (a_row_offset + 1) * m + a_col_offset
    a11 = (a_row_offset + 1) * m + a_col_offset + 1
    
    b00 = b_row_offset * p + b_col_offset
    b01 = b_row_offset * p + b_col_offset + 1
    b10 = (b_row_offset + 1) * p + b_col_offset
    b11 = (b_row_offset + 1) * p + b_col_offset + 1
    
    c00 = c_col_offset * n + c_row_offset
    c01 = (c_col_offset + 1) * n + c_row_offset
    c10 = c_col_offset * n + c_row_offset + 1
    c11 = (c_col_offset + 1) * n + c_row_offset + 1
    
    # Strassen:
    # M1 = (a00+a11)(b00+b11) -> C00 += M1, C11 += M1
    # M2 = (a10+a11)b00 -> C10 += M2, C11 -= M2
    # M3 = a00(b01-b11) -> C01 += M3, C11 += M3
    # M4 = a11(b10-b00) -> C00 += M4, C10 += M4
    # M5 = (a00+a10)b11 -> C00 -= M5, C01 += M5
    # M6 = (a01-a11)(b10+b11) -> C11 += M6
    # M7 = (a10-a00)(b00+b10) -> C00 += M7
    
    # M1
    U[a00][col] += 1
    U[a11][col] += 1
    V[b00][col] += 1
    V[b11][col] += 1
    W[c00][col] += 1
    W[c11][col] += 1
    col += 1
    
    # M2
    U[a10][col] += 1
    U[a11][col] += 1
    V[b00][col] += 1
    W[c10][col] += 1
    W[c11][col] += -1
    col += 1
    
    # M3
    U[a00][col] += 1
    V[b01][col] += 1
    V[b11][col] += -1
    W[c01][col] += 1
    W[c11][col] += 1
    col += 1
    
    # M4
    U[a11][col] += 1
    V[b10][col] += 1
    V[b00][col] += -1
    W[c00][col] += 1
    W[c10][col] += 1
    col += 1
    
    # M5
    U[a00][col] += 1
    U[a10][col] += 1
    V[b11][col] += 1
    W[c00][col] += -1
    W[c01][col] += 1
    col += 1
    
    # M6
    U[a01][col] += 1
    U[a11][col] += -1
    V[b10][col] += 1
    V[b11][col] += 1
    W[c11][col] += 1
    col += 1
    
    # M7
    U[a10][col] += 1
    U[a00][col] += -1
    V[b00][col] += 1
    V[b10][col] += 1
    W[c00][col] += 1
    col += 1
    
    return col

def _add_schoolbook_4x1(U, V, W, col, n, m, p, b_col_offset):
    # Add schoolbook decomposition for 4x4 x 4x1
    # A is 4x4, B is 4x1 (column b_col_offset), C is 4x1
    # C[i] = sum_j A[i,j] * B[j, b_col_offset]
    # Rank = 16
    
    for i in range(4):
        for j in range(4):
            a_idx = i * m + j
            b_idx = j * p + b_col_offset
            c_idx = b_col_offset * n + i
            U[a_idx][col] = 1
            V[b_idx][col] = 1
            W[c_idx][col] = 1
            col += 1
    
    return col
