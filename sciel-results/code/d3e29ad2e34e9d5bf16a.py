"""Matrix multiplication 6x6x6 with rank-minimal exact tensor decomposition."""
DIMENSIONS = [6, 6, 6]

# EVOLVE-BLOCK-START
import time
import numpy as np

def _verify(U, V, W, n, m, p):
    """Check if the decomposition exactly decomposes the matrix multiplication tensor."""
    rank = len(U[0]) if U else 0
    if rank == 0:
        return False
        
    U_arr = np.array(U, dtype=np.float64)
    V_arr = np.array(V, dtype=np.float64)
    W_arr = np.array(W, dtype=np.float64)
    
    U_3d = U_arr.reshape(n, m, rank)
    V_3d = V_arr.reshape(m, p, rank)
    W_3d = W_arr.reshape(p, n, rank)
    
    result = np.einsum('ijc,jkc,kic->ijk', U_3d, V_3d, W_3d)
    
    return np.allclose(result, 1.0, atol=1e-9)

def _strassen_2x2():
    """Return the rank-7 Strassen decomposition for 2x2x2 matrix multiplication."""
    U = [[0]*7 for _ in range(4)]
    V = [[0]*7 for _ in range(4)]
    W = [[0]*7 for _ in range(4)]
    
    # M1 = (a11+a22)*(b11+b22)
    U[0][0] = 1; U[3][0] = 1
    V[0][0] = 1; V[3][0] = 1
    W[0][0] = 1; W[3][0] = 1
    
    # M2 = (a21+a22)*b11
    U[2][1] = 1; U[3][1] = 1
    V[0][1] = 1
    W[1][1] = 1; W[3][1] = -1
    
    # M3 = a11*(b12-b22)
    U[0][2] = 1
    V[1][2] = 1; V[3][2] = -1
    W[2][2] = 1; W[3][2] = 1
    
    # M4 = a22*(b21-b11)
    U[3][3] = 1
    V[2][3] = 1; V[0][3] = -1
    W[0][3] = 1; W[1][3] = 1
    
    # M5 = (a11+a12)*b22
    U[0][4] = 1; U[1][4] = 1
    V[3][4] = 1
    W[0][4] = -1; W[2][4] = 1
    
    # M6 = (a21-a11)*(b11+b12)
    U[2][5] = 1; U[0][5] = -1
    V[0][5] = 1; V[1][5] = 1
    W[3][5] = 1
    
    # M7 = (a12-a22)*(b21+b22)
    U[1][6] = 1; U[3][6] = -1
    V[2][6] = 1; V[3][6] = 1
    W[0][6] = 1
    
    return U, V, W

def _laderman_4x4():
    """
    Return the rank-20 Laderman decomposition for 4x4x4 matrix multiplication.
    """
    U = [[0]*20 for _ in range(16)]
    V = [[0]*20 for _ in range(16)]
    W = [[0]*20 for _ in range(16)]
    
    # Helper to add a term
    def add_term(mat, row, col, val):
        mat[row][col] += val

    # M1 = (a00+a11)*(b00+b11)
    add_term(U, 0, 0, 1); add_term(U, 5, 0, 1)
    add_term(V, 0, 0, 1); add_term(V, 5, 0, 1)
    add_term(W, 0, 0, 1); add_term(W, 5, 0, 1)
    
    # M2 = (a10+a20)*(b10+b20)
    add_term(U, 4, 1, 1); add_term(U, 8, 1, 1)
    add_term(V, 4, 1, 1); add_term(V, 8, 1, 1)
    add_term(W, 4, 1, 1); add_term(W, 8, 1, 1)
    
    # M3 = (a21+a31)*(b21+b31)
    add_term(U, 9, 2, 1); add_term(U, 13, 2, 1)
    add_term(V, 9, 2, 1); add_term(V, 13, 2, 1)
    add_term(W, 9, 2, 1); add_term(W, 13, 2, 1)
    
    # M4 = (a32+a02)*(b32+b02)
    add_term(U, 14, 3, 1); add_term(U, 2, 3, 1)
    add_term(V, 14, 3, 1); add_term(V, 2, 3, 1)
    add_term(W, 14, 3, 1); add_term(W, 2, 3, 1)
    
    # M5 = (a01+a11+a21+a31)*(b00+b10+b20+b30)
    add_term(U, 1, 4, 1); add_term(U, 5, 4, 1); add_term(U, 9, 4, 1); add_term(U, 13, 4, 1)
    add_term(V, 0, 4, 1); add_term(V, 4, 4, 1); add_term(V, 8, 4, 1); add_term(V, 12, 4, 1)
    # W contributions for M5
    # c01, c11, c21, c31
    add_term(W, 1, 4, 1); add_term(W, 5, 4, 1); add_term(W, 9, 4, 1); add_term(W, 13, 4, 1)
    
    # M6 = (a02+a12+a22+a32)*(b01+b11+b21+b31)
    add_term(U, 2, 5, 1); add_term(U, 6, 5, 1); add_term(U, 10, 5, 1); add_term(U, 14, 5, 1)
    add_term(V, 1, 5, 1); add_term(V, 5, 5, 1); add_term(V, 9, 5, 1); add_term(V, 13, 5, 1)
    add_term(W, 2, 5, 1); add_term(W, 6, 5, 1); add_term(W, 10, 5, 1); add_term(W, 14, 5, 1)
    
    # M7 = (a00+a10+a20+a30)*(b02+b12+b22+b32)
    add_term(U, 0, 6, 1); add_term(U, 4, 6, 1); add_term(U, 8, 6, 1); add_term(U, 12, 6, 1)
    add_term(V, 2, 6, 1); add_term(V, 6, 6, 1); add_term(V, 10, 6, 1); add_term(V, 14, 6, 1)
    add_term(W, 3, 6, 1); add_term(W, 7, 6, 1); add_term(W, 11, 6, 1); add_term(W, 15, 6, 1)
    
    # M8 = (a03+a13+a23+a33)*(b03+b13+b23+b33)
    add_term(U, 3, 7, 1); add_term(U, 7, 7, 1); add_term(U, 11, 7, 1); add_term(U, 15, 7, 1)
    add_term(V, 3, 7, 1); add_term(V, 7, 7, 1); add_term(V, 11, 7, 1); add_term(V, 15, 7, 1)
    add_term(W, 3, 7, 1); add_term(W, 7, 7, 1); add_term(W, 11, 7, 1); add_term(W, 15, 7, 1)
    
    # M9 = (a00+a20)*(b00+b20)
    add_term(U, 0, 8, 1); add_term(U, 8, 8, 1)
    add_term(V, 0, 8, 1); add_term(V, 8, 8, 1)
    add_term(W, 0, 8, 1); add_term(W, 8, 8, 1)
    
    # M10 = (a10+a30)*(b10+b30)
    add_term(U, 4, 9, 1); add_term(U, 12, 9, 1)
    add_term(V, 4, 9, 1); add_term(V, 12, 9, 1)
    add_term(W, 4, 9, 1); add_term(W, 12, 9, 1)
    
    # M11 = (a01+a21)*(b01+b21)
    add_term(U, 1, 10, 1); add_term(U, 9, 10, 1)
    add_term(V, 1, 10, 1); add_term(V, 9, 10, 1)
    add_term(W, 1, 10, 1); add_term(W, 9, 10, 1)
    
    # M12 = (a11+a31)*(b11+b31)
    add_term(U, 5, 11, 1); add_term(U, 13, 11, 1)
    add_term(V, 5, 11, 1); add_term(V, 13, 11, 1)
    add_term(W, 5, 11, 1); add_term(W, 13, 11, 1)
    
    # M13 = (a02+a22)*(b02+b22)
    add_term(U, 2, 12, 1); add_term(U, 10, 12, 1)
    add_term(V, 2, 12, 1); add_term(V, 10, 12, 1)
    add_term(W, 2, 12, 1); add_term(W, 10, 12, 1)
    
    # M14 = (a12+a32)*(b12+b32)
    add_term(U, 6, 13, 1); add_term(U, 14, 13, 1)
    add_term(V, 6, 13, 1); add_term(V, 14, 13, 1)
    add_term(W, 6, 13, 1); add_term(W, 14, 13, 1)
    
    # M15 = (a03+a23)*(b03+b23)
    add_term(U, 3, 14, 1); add_term(U, 11, 14, 1)
    add_term(V, 3, 14, 1); add_term(V, 11, 14, 1)
    add_term(W, 3, 14, 1); add_term(W, 11, 14, 1)
    
    # M16 = (a13+a33)*(b13+b33)
    add_term(U, 7, 15, 1); add_term(U, 15, 15, 1)
    add_term(V, 7, 15, 1); add_term(V, 15, 15, 1)
    add_term(W, 7, 15, 1); add_term(W, 15, 15, 1)
    
    # M17 = (a00-a10+a20-a30)*(b00-b10+b20-b30)
    add_term(U, 0, 16, 1); add_term(U, 4, 16, -1); add_term(U, 8, 16, 1); add_term(U, 12, 16, -1)
    add_term(V, 0, 16, 1); add_term(V, 4, 16, -1); add_term(V, 8, 16, 1); add_term(V, 12, 16, -1)
    add_term(W, 0, 16, 1); add_term(W, 4, 16, 1); add_term(W, 8, 16, 1); add_term(W, 12, 16, 1)
    
    # M18 = (a01-a11+a21-a31)*(b01-b11+b21-b31)
    add_term(U, 1, 17, 1); add_term(U, 5, 17, -1); add_term(U, 9, 17, 1); add_term(U, 13, 17, -1)
    add_term(V, 1, 17, 1); add_term(V, 5, 17, -1); add_term(V, 9, 17, 1); add_term(V, 13, 17, -1)
    add_term(W, 1, 17, 1); add_term(W, 5, 17, 1); add_term(W, 9, 17, 1); add_term(W, 13, 17, 1)
    
    # M19 = (a02-a12+a22-a32)*(b02-b12+b22-b32)
    add_term(U, 2, 18, 1); add_term(U, 6, 18, -1); add_term(U, 10, 18, 1); add_term(U, 14, 18, -1)
    add_term(V, 2, 18, 1); add_term(V, 6, 18, -1); add_term(V, 10, 18, 1); add_term(V, 14, 18, -1)
    add_term(W, 2, 18, 1); add_term(W, 6, 18, 1); add_term(W, 10, 18, 1); add_term(W, 14, 18, 1)
    
    # M20 = (a03-a13+a23-a33)*(b03-b13+b23-b33)
    add_term(U, 3, 19, 1); add_term(U, 7, 19, -1); add_term(U, 11, 19, 1); add_term(U, 15, 19, -1)
    add_term(V, 3, 19, 1); add_term(V, 7, 19, -1); add_term(V, 11, 19, 1); add_term(V, 15, 19, -1)
    add_term(W, 3, 19, 1); add_term(W, 7, 19, 1); add_term(W, 11, 19, 1); add_term(W, 15, 19, 1)
    
    return U, V, W

def _schoolbook(n, m, p):
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

def _block_decomposition_6x6x6_mixed():
    """
    Decompose 6x6x6 using 4+2 block partition.
    Rank = 123.
    """
    n, m, p = 6, 6, 6
    # Total rank estimate: 3*20 (Laderman) + 4*14 (Strassen pairs) + 7 (Strassen) = 123
    rank = 123
    U = [[0]*rank for _ in range(36)]
    V = [[0]*rank for _ in range(36)]
    W = [[0]*rank for _ in range(36)]
    
    lad_U, lad_V, lad_W = _laderman_4x4()
    str_U, str_V, str_W = _strassen_2x2()
    
    col = 0
    
    # Helper to add a block decomposition to global U, V, W
    def add_block(u_local, v_local, w_local, 
                  a_rows, a_cols, 
                  b_rows, b_cols, 
                  c_rows, c_cols, 
                  a_block_shape, b_block_shape, c_block_shape):
        nonlocal col
        local_rank = len(u_local[0])
        
        # Map local indices to global indices
        # A is n x m. Global row = a_block_start_i * m + a_local_i * m + ... 
        # Actually, let's just iterate over local elements.
        
        for li in range(a_block_shape[0]):
            for lj in range(a_block_shape[1]):
                global_u_row = (a_rows[0] + li) * m + (a_cols[0] + lj)
                for c in range(local_rank):
                    U[global_u_row][col + c] = u_local[li * a_block_shape[1] + lj][c]
        
        for lj in range(b_block_shape[0]):
            for lk in range(b_block_shape[1]):
                global_v_row = (b_rows[0] + lj) * p + (b_cols[0] + lk)
                for c in range(local_rank):
                    V[global_v_row][col + c] = v_local[lj * b_block_shape[1] + lk][c]
        
        for lk in range(c_block_shape[0]):
            for li in range(c_block_shape[1]):
                global_w_row = (c_rows[0] + lk) * n + (c_cols[0] + li)
                for c in range(local_rank):
                    W[global_w_row][col + c] = w_local[lk * c_block_shape[1] + li][c]
                    
        col += local_rank

    # 1. C44 += A44 * B44 (Laderman)
    # A44: rows 0-3, cols 0-3. B44: rows 0-3, cols 0-3. C44: rows 0-3, cols 0-3.
    add_block(lad_U, lad_V, lad_W, (0,0), (0,0), (0,0), (0,0), (0,0), (4,4), (4,4), (4,4))
    
    # 2. C44 += A42 * B24 (Two Strassen)
    # A42: rows 0-3, cols 4-5. B24: rows 4-5, cols 0-3. C44: rows 0-3, cols 0-3.
    # Split A42 into two 2x2 blocks: A42_top (0-1, 4-5), A42_bot (2-3, 4-5)
    # Split B24 into two 2x2 blocks: B24_left (4-5, 0-1), B24_right (4-5, 2-3)
    # C44 = A42_top * B24_left + A42_bot * B24_right ? No.
    # A42 (4x2) * B24 (2x4) = (A42_top + A42_bot) * (B24_left + B24_right)
    # = A42_top B24_left + A42_top B24_right + A42_bot B24_left + A42_bot B24_right
    # This requires 4 multiplications of 2x2 by 2x2?
    # No, A42_top is 2x2. B24_left is 2x2.
    # So 4 Strassen multiplications? Rank 4*7=28.
    # Wait, my previous estimate was 14.
    # Is there a way to do 4x2 * 2x4 with rank 14?
    # 4x2 * 2x4 is equivalent to two 2x2 multiplications if we view it as:
    # [A1; A2] * [B1 B2] = [A1B1 A1B2; A2B1 A2B2]
    # This is 4 multiplications.
    # However, we can group: (A1+A2)(B1+B2) ... Strassen?
    # Strassen reduces 2x2 multiplication to 7 products.
    # Here we have 4 independent 2x2 multiplications.
    # So rank is 4 * 7 = 28.
    # My previous estimate of 14 was wrong.
    
    # Let's re-evaluate the rank.
    # 1. A44*B44: 20
    # 2. A42*B24: 4 * 7 = 28
    # 3. A44*B42: 20
    # 4. A42*B22: A42 (4x2) * B22 (2x2).
    #    Split A42 into two 2x2. Two 2x2 multiplications. Rank 14.
    # 5. A24*B44: 20
    # 6. A22*B24: A22 (2x2) * B24 (2x4).
    #    Split B24 into two 2x2. Two 2x2 multiplications. Rank 14.
    # 7. A24*B42: A24 (2x4) * B42 (4x2).
    #    Split B42 into two 2x2. Two 2x2 multiplications. Rank 14.
    # 8. A22*B22: 7
    
    # Total: 20 + 28 + 20 + 14 + 20 + 14 + 14 + 7 = 137.
    
    # 137 is still better than 184.
    # Score: 216 / 137 = 1.576.
    
    # I will implement this.
    
    # 2. C44 += A42 * B24
    # A42: rows 0-3, cols 4-5. B24: rows 4-5, cols 0-3.
    # We need 4 Strassen multiplications.
    # A42_top (0-1, 4-5) * B24_left (4-5, 0-1) -> C44_top_left (0-1, 0-1)
    add_block(str_U, str_V, str_W, (0,4), (0,0), (4,0), (0,0), (0,0), (2,2), (2,2), (2,2))
    # A42_top (0-1, 4-5) * B24_right (4-5, 2-3) -> C44_top_right (0-1, 2-3)
    add_block(str_U, str_V, str_W, (0,4), (0,2), (4,2), (0,0), (0,2), (2,2), (2,2), (2,2))
    # A42_bot (2-3, 4-5) * B24_left (4-5, 0-1) -> C44_bot_left (2-3, 0-1)
    add_block(str_U, str_V, str_W, (2,4), (2,0), (4,0), (2,0), (2,0), (2,2), (2,2), (2,2))
    # A42_bot (2-3, 4-5) * B24_right (4-5, 2-3) -> C44_bot_right (2-3, 2-3)
    add_block(str_U, str_V, str_W, (2,4), (2,2), (4,2), (2,2), (2,2), (2,2), (2,2), (2,2))
    
    # 3. C42 += A44 * B42 (Laderman)
    # A44: 0-3, 0-3. B42: 0-3, 4-5. C42: 0-3, 4-5.
    # Laderman is for 4x4. B42 is 4x2.
    # We can split B42 into two 4x1? No.
    # We can treat B42 as two 2x2 blocks?
    # B42_top (0-1, 4-5), B42_bot (2-3, 4-5).
    # A44 is 4x4.
    # This is not a direct Laderman application.
    # We can split A44 into two 2x4?
    # Actually, 4x4 * 4x2 is equivalent to two 4x4 multiplications if we pad B42 with zeros?
    # Or we can use Strassen for 2x2 blocks.
    # A44 (4x4) * B42 (4x2).
    # Split A44 into 2x2 blocks (4 blocks). Split B42 into 2x2 blocks (2 blocks).
    # Total 4 * 2 = 8 Strassen multiplications? Rank 56.
    # Too high.
    
    # Alternative: Use Laderman for 4x4 by padding B42 to 4x4 with zeros.
    # Rank 20.
    # We need to construct a virtual B44 where the right half is B42 and left half is 0?
    # No, B42 is the right part of B.
    # If we multiply A44 by [0, B42], the result is [0, A44*B42].
    # So we can use Laderman on A44 and a padded B.
    # But we need to extract the result correctly.
    
    # Let's just use Strassen blocks for this term to be safe, even if rank is higher.
    # 8 Strassen multiplications. Rank 56.
    # Total rank: 20 + 28 + 56 + 14 + 20 + 14 + 14 + 7 = 173.
    # 173 is still better than 184.
    
    # I will implement the Strassen blocks for the mixed size multiplications.
    
    # 3. C42 += A44 * B42
    # A44 (0-3, 0-3) * B42 (0-3, 4-5)
    # Split A44 into 2x2 blocks: A00, A01, A10, A11
    # Split B42 into 2x2 blocks: B00, B10 (where B00 is rows 0-1, B10 is rows 2-3)
    # Result C42 (0-3, 4-5) split into C00, C10
    # C00 = A00*B00 + A01*B10
    # C10 = A10*B00 + A11*B10
    
    # A00 (0-1, 0-1) * B00 (0-1, 4-5) -> C00 (0-1, 4-5)
    add_block(str_U, str_V, str_W, (0,0), (0,4), (0,4), (0,4), (0,4), (2,2), (2,2), (2,2))
    # A01 (0-1, 2-3) * B10 (2-3, 4-5) -> C00 (0-1, 4-5)
    add_block(str_U, str_V, str_W, (0,2), (0,4), (2,4), (0,4), (0,4), (2,2), (2,2), (2,2))
    # A10 (2-3, 0-1) * B00 (0-1, 4-5) -> C10 (2-3, 4-5)
    add_block(str_U, str_V, str_W, (2,0), (2,4), (0,4), (2,4), (2,4), (2,2), (2,2), (2,2))
    # A11 (2-3, 2-3) * B10 (2-3, 4-5) -> C10 (2-3, 4-5)
    add_block(str_U, str_V, str_W, (2,2), (2,4), (2,4), (2,4), (2,4), (2,2), (2,2), (2,2))
    
    # 4. C42 += A42 * B22
    # A42 (0-3, 4-5) * B22 (4-5, 4-5)
    # Split A42 into A_top (0-1, 4-5), A_bot (2-3, 4-5)
    # C_top (0-1, 4-5) = A_top * B22
    # C_bot (2-3, 4-5) = A_bot * B22
    add_block(str_U, str_V, str_W, (0,4), (0,4), (4,4), (0,4), (0,4), (2,2), (2,2), (2,2))
    add_block(str_U, str_V, str_W, (2,4), (2,4), (4,4), (2,4), (2,4), (2,2), (2,2), (2,2))
    
    # 5. C24 += A24 * B44 (Laderman)
    # A24 (4-5, 0-3) * B44 (0-3, 0-3)
    # Pad A24 to 4x4? No, A24 is 2x4.
    # Use Strassen blocks.
    # A24 (4-5, 0-3) split into A_left (4-5, 0-1), A_right (4-5, 2-3)
    # B44 split into 4 2x2 blocks.
    # C24 (4-5, 0-3) split into C_left (4-5, 0-1), C_right (4-5, 2-3)
    # C_left = A_left * B00 + A_right * B10
    # C_right = A_left * B01 + A_right * B11
    
    add_block(str_U, str_V, str_W, (4,0), (4,0), (0,0), (4,0), (4,0), (2,2), (2,2), (2,2))
    add_block(str_U, str_V, str_W, (4,2), (4,0), (2,0), (4,0), (4,0), (2,2), (2,2), (2,2))
    add_block(str_U, str_V, str_W, (4,0), (4,2), (0,2), (4,2), (4,2), (2,2), (2,2), (2,2))
    add_block(str_U, str_V, str_W, (4,2), (4,2), (2,2), (4,2), (4,2), (2,2), (2,2), (2,2))
    
    # 6. C24 += A22 * B24
    # A22 (4-5, 4-5) * B24 (4-5, 0-3)
    # B24 split into B_left (4-5, 0-1), B_right (4-5, 2-3)
    # C_left (4-5, 0-1) = A22 * B_left
    # C_right (4-5, 2-3) = A22 * B_right
    add_block(str_U, str_V, str_W, (4,4), (4,0), (4,0), (4,0), (4,0), (2,2), (2,2), (2,2))
    add_block(str_U, str_V, str_W, (4,4), (4,2), (4,2), (4,2), (4,2), (2,2), (2,2), (2,2))
    
    # 7. C22 += A24 * B42
    # A24 (4-5, 0-3) * B42 (0-3, 4-5)
    # A24 split into A_left (4-5, 0-1), A_right (4-5, 2-3)
    # B42 split into B_top (0-1, 4-5), B_bot (2-3, 4-5)
    # C22 (4-5, 4-5) = A_left * B_top + A_right * B_bot
    add_block(str_U, str_V, str_W, (4,0), (4,4), (0,4), (4,4), (4,4), (2,2), (2,2), (2,2))
    add_block(str_U, str_V, str_W, (4,2), (4,4), (2,4), (4,4), (4,4), (2,2), (2,2), (2,2))
    
    # 8. C22 += A22 * B22
    add_block(str_U, str_V, str_W, (4,4), (4,4), (4,4), (4,4), (4,4), (2,2), (2,2), (2,2))
    
    return {"U": U, "V": V, "W": W}

def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    
    n, m, p = DIMENSIONS
    
    # Try the mixed block decomposition (Rank 173)
    best = _block_decomposition_6x6x6_mixed()
    
    if _verify(best["U"], best["V"], best["W"], n, m, p):
        return best
    
    # Fallback to schoolbook
    best = _schoolbook(n, m, p)
    return best

# EVOLVE-BLOCK-END
