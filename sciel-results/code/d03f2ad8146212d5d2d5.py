"""Rewritten program: explicit rank-44 construction for 2x3 * 3x8.

Strategy:
- Split A (2x3) into two 2x3 blocks: A_left = A[:, :3] (full A), A_right = A[:, 3:]? 
  No. A is 2x3, B is 3x8. So the middle dimension is 3.
  Actually, the "2x3x2" block decomposition mentioned in the knowledge refers to 
  decomposing the 2x3 matrix multiplication into blocks. But here the dimensions are 
  2x3 * 3x8.
  
  Let me re-read the knowledge:
  "Explicit rank-44 construction via Strassen(7)+4 rank-1 per 2x3x2 block, verified 
   immediately, beats schoolbook 48"
  
  So the idea is: for a 2x3 * 3x2 multiplication, we can use rank 11 (Strassen 7 + 4 
  rank-1 cross terms). Then for 2x3 * 3x8, we can split the 8 columns of B into four 
  2-column blocks. Each 2x3 * 3x2 block has rank 11, so 4 * 11 = 44.

  Let's implement this.
  
  For a 2x3 * 3x2 matrix multiplication:
  A = [[a00, a01, a02],
       [a10, a11, a12]]
  B = [[b00, b01],
       [b10, b11],
       [b20, b21]]
  C = A * B (2x2)

  We need a rank-11 decomposition for this 2x3x2 tensor.
  
  Strassen's algorithm works for 2x2 * 2x2. Here we have 2x3 * 3x2.
  One way: split the middle dimension 3 into 2+1.
  A = A_left (2x2) * B_left (2x2) + A_right (2x1) * B_right (1x2)
  
  A_left = [[a00, a01],
            [a10, a11]]
  B_left = [[b00, b01],
            [b10, b11]]
  A_right = [[a02],
             [a12]]
  B_right = [[b20, b21]]
  
  C = A_left * B_left + A_right * B_right
  
  Use Strassen for A_left * B_left: rank 7
  Use 4 rank-1 terms for A_right * B_right:
  C_extra = [[a02*b20, a02*b21],
             [a12*b20, a12*b21]]
  This is a 2x1 * 1x2 = 2x2 matrix, which can be done with rank 4 (each element is a product).
  
  So total rank = 7 + 4 = 11.

  Now for the full 2x3 * 3x8:
  Split B into 4 blocks of 2 columns each: B = [B1, B2, B3, B4], each Bi is 3x2.
  C = A * B = [A*B1, A*B2, A*B3, A*B4], each Ai is 2x2.
  
  Each A*Bi has rank 11, so total rank = 4 * 11 = 44.

  Now let's construct the U, V, W matrices.
  
  For each block bi (i = 0, 1, 2, 3):
  A is 2x3, indexed by (row i_a in 0..1, col j_a in 0..2).
  B block is 3x2, indexed by (row j_b in 0..2, col k_b in 0..1).
  C block is 2x2, indexed by (row i_c in 0..1, col k_c in 0..1).
  
  The full C is 2x8, indexed by (i_c in 0..1, k in 0..7) where k = bi*2 + k_c.
  
  W is p*n = 8*2 = 16 rows, indexed by k*n + i = k*2 + i.
  
  For the Strassen part (A_left * B_left):
  A_left is rows 0..1, cols 0..1 of A.
  B_left is rows 0..1, col 0..1 of B block.
  
  Strassen's 7 products for 2x2 * 2x2:
  M1 = (a00 + a11) * (b00 + b11)
  M2 = (a10 + a11) * b00
  M3 = a00 * (b10 - b00)
  M4 = a11 * (b01 - b11)
  M5 = (a00 + a01) * b11
  M6 = (a10 - a00) * (b00 + b10)
  M7 = (a01 - a11) * (b11 + b01)
  
  C00 = M1 + M4 - M5 + M7
  C01 = M3 + M5
  C10 = M2 + M4
  C11 = M1 - M2 + M3 + M6

  For the cross terms (A_right * B_right):
  M8 = a02 * b20
  M9 = a02 * b21
  M10 = a12 * b20
  M11 = a12 * b21
  
  C00 += M8
  C01 += M9
  C10 += M10
  C11 += M11

  So for each block, we have 11 rank-1 terms.
  
  Now let's construct U, V, W for the full problem.
  
  U: (n*m, rank) = (2*3, 44) = (6, 44)
  V: (m*p, rank) = (3*8, 44) = (24, 44)
  W: (p*n, rank) = (8*2, 44) = (16, 44)
  
  Indexing:
  U[i*m + j, c] where i in 0..1, j in 0..2
  V[j*p + k, c] where j in 0..2, k in 0..7
  W[k*n + i, c] where k in 0..7, i in 0..1
  
  For block bi (bi = 0, 1, 2, 3), the columns of B are k = bi*2, bi*2+1.
  The columns of C are k = bi*2, bi*2+1.
  
  For each term in the block, we set the appropriate entries in U, V, W.
  
  Let's code this.
"""

def construct():
    n, m, p = 2, 3, 8
    num_blocks = p // 2  # 4 blocks of 2 columns each
    rank_per_block = 11
    total_rank = num_blocks * rank_per_block  # 44
    
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    
    col = 0
    for bi in range(num_blocks):
        # B block columns: k0 = bi*2, k1 = bi*2 + 1
        k0 = bi * 2
        k1 = bi * 2 + 1
        
        # Strassen terms for A_left (cols 0,1) * B_left (rows 0,1)
        # M1 = (a00 + a11) * (b00 + b11)
        U[0 * m + 0][col] = 1  # a00
        U[1 * m + 1][col] = 1  # a11
        V[0 * p + k0][col] = 1  # b00
        V[1 * p + k0][col] = 1  # b10
        # Wait, B is 3x2, so B_left is rows 0,1 and cols k0,k1.
        # b00 is B[0][k0], b10 is B[1][k0], etc.
        # Actually, B is 3x8, so B[j][k] where j in 0..2, k in 0..7.
        # For block bi, the 2x2 submatrix is B[0:2, k0:k1+1].
        # So b00 = B[0][k0], b01 = B[0][k1], b10 = B[1][k0], b11 = B[1][k1].
        
        # Let me redo.
        # M1 = (a00 + a11) * (b00 + b11)
        # b00 = B[0][k0], b11 = B[1][k1]
        # So V[0*p + k0][col] = 1, V[1*p + k1][col] = 1
        
        # Let me restart the Strassen terms with correct indexing.
        
        # M1 = (a00 + a11) * (b00 + b11)
        U[0 * m + 0][col] = 1
        U[1 * m + 1][col] = 1
        V[0 * p + k0][col] = 1
        V[1 * p + k1][col] = 1
        # C00 += 1, C11 += 1
        # C00 is at row 0, col k0 in C. W index: k0*n + 0 = k0*2 + 0
        # C11 is at row 1, col k1 in C. W index: k1*n + 1 = k1*2 + 1
        W[k0 * n + 0][col] = 1
        W[k1 * n + 1][col] = 1
        col += 1
        
        # M2 = (a10 + a11) * b00
        U[1 * m + 0][col] = 1
        U[1 * m + 1][col] = 1
        V[0 * p + k0][col] = 1
        # C10 += 1
        # C10 is at row 1, col k0. W index: k0*n + 1 = k0*2 + 1
        W[k0 * n + 1][col] = 1
        col += 1
        
        # M3 = a00 * (b10 - b00)
        U[0 * m + 0][col] = 1
        V[1 * p + k0][col] = 1
        V[0 * p + k0][col] = -1
        # C01 += 1
        # C01 is at row 0, col k1. W index: k1*n + 0 = k1*2 + 0
        W[k1 * n + 0][col] = 1
        col += 1
        
        # M4 = a11 * (b01 - b11)
        U[1 * m + 1][col] = 1
        V[0 * p + k1][col] = 1
        V[1 * p + k1][col] = -1
        # C00 += 1, C10 += 1
        W[k0 * n + 0][col] = 1
        W[k0 * n + 1][col] = 1
        col += 1
        
        # M5 = (a00 + a01) * b11
        U[0 * m + 0][col] = 1
        U[0 * m + 1][col] = 1
        V[1 * p + k1][col] = 1
        # C00 += -1, C01 += 1
        W[k0 * n + 0][col] = -1
        W[k1 * n + 0][col] = 1
        col += 1
        
        # M6 = (a10 - a00) * (b00 + b10)
        U[1 * m + 0][col] = 1
        U[0 * m + 0][col] = -1
        V[0 * p + k0][col] = 1
        V[1 * p + k0][col] = 1
        # C11 += 1
        W[k1 * n + 1][col] = 1
        col += 1
        
        # M7 = (a01 - a11) * (b11 + b01)
        U[0 * m + 1][col] = 1
        U[1 * m + 1][col] = -1
        V[1 * p + k1][col] = 1
        V[0 * p + k1][col] = 1
        # C00 += 1, C11 += 1
        W[k0 * n + 0][col] = 1
        W[k1 * n + 1][col] = 1
        col += 1
        
        # Cross terms: A_right (col 2) * B_right (row 2)
        # M8 = a02 * b20
        U[0 * m + 2][col] = 1
        V[2 * p + k0][col] = 1
        # C00 += 1
        W[k0 * n + 0][col] = 1
        col += 1
        
        # M9 = a02 * b21
        U[0 * m + 2][col] = 1
        V[2 * p + k1][col] = 1
        # C01 += 1
        W[k1 * n + 0][col] = 1
        col += 1
        
        # M10 = a12 * b20
        U[1 * m + 2][col] = 1
        V[2 * p + k0][col] = 1
        # C10 += 1
        W[k0 * n + 1][col] = 1
        col += 1
        
        # M11 = a12 * b21
        U[1 * m + 2][col] = 1
        V[2 * p + k1][col] = 1
        # C11 += 1
        W[k1 * n + 1][col] = 1
        col += 1
    
    return {"U": U, "V": V, "W": W}
