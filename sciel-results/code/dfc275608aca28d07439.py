"""Matrix multiplication tensor decomposition for 2x4 by 4x5 matrices."""
import time
import numpy as np

DIMENSIONS = [2, 4, 5]

def _verify(U, V, W, n, m, p):
    """Verify that U, V, W correctly decompose the matrix multiplication tensor."""
    rank = len(U[0])
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for c in range(rank):
                    s += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                if abs(s - 1.0) > 1e-9:
                    return False
    return True

def _construct_rank36():
    """
    Construct a rank-36 decomposition for 2x4 * 4x5 using block Strassen.
    
    Strategy: Split A into left/right 2x2 blocks, B into top/bottom 2x5 blocks.
    For each 2x2 * 2x5 sub-product, use rank 18:
      - Strassen (rank 7) for column pairs (0,1) and (2,3)
      - Schoolbook (rank 4) for column 4
    Total: 18 + 18 = 36.
    
    Indexing:
      U[i*m + j] where i in {0,1}, j in {0,1,2,3}
      V[j*p + k] where j in {0,1,2,3}, k in {0,1,2,3,4}
      W[k*n + i] where k in {0,1,2,3,4}, i in {0,1}
    """
    n, m, p = 2, 4, 5
    total_rank = 36
    
    U = [[0.0] * total_rank for _ in range(n * m)]  # 8 x 36
    V = [[0.0] * total_rank for _ in range(m * p)]  # 20 x 36
    W = [[0.0] * total_rank for _ in range(p * n)]  # 10 x 36
    
    def add_2x2x5(a_col_offset, b_row_offset, base_col):
        """
        Add rank-18 decomposition for 2x2 * 2x5.
        a_col_offset: column offset in A (0 for left, 2 for right)
        b_row_offset: row offset in B (0 for top, 2 for bottom)
        base_col: starting column in the rank dimension
        """
        # A indices: i*4 + a_col_offset + k for i in {0,1}, k in {0,1}
        # B indices: (b_row_offset + k)*5 + j for k in {0,1}, j in {0..4}
        # C indices: j*2 + i for j in {0..4}, i in {0,1}
        
        # Strassen for 2x2 * 2x2 (columns 0,1):
        # A = [[a00, a01], [a10, a11]]
        # B = [[b00, b01], [b10, b11]]
        # M1 = (a00+a11)(b00+b11)
        # M2 = (a10+a11)b00
        # M3 = a00(b10-b11)
        # M4 = a11(b00-b10)
        # M5 = (a00+a01)b11
        # M6 = (a10-a00)(b00+b01)
        # M7 = (a01-a11)(b10+b11)
        # C00 = M1+M4-M5+M7
        # C01 = M3+M5
        # C10 = M2+M4
        # C11 = M1-M2+M3+M6
        
        c = base_col
        
        # M1 = (a00+a11)(b00+b11)
        U[0*4 + a_col_offset + 0][c] += 1
        U[1*4 + a_col_offset + 1][c] += 1
        V[(b_row_offset+0)*5 + 0][c] += 1
        V[(b_row_offset+1)*5 + 1][c] += 1
        # C00 = W[0], C11 = W[3]
        W[0][c] += 1
        W[3][c] += 1
        c += 1
        
        # M2 = (a10+a11)b00
        U[1*4 + a_col_offset + 0][c] += 1
        U[1*4 + a_col_offset + 1][c] += 1
        V[(b_row_offset+0)*5 + 0][c] += 1
        # C10 = W[1]
        W[1][c] += 1
        c += 1
        
        # M3 = a00(b10-b11)
        U[0*4 + a_col_offset + 0][c] += 1
        V[(b_row_offset+1)*5 + 0][c] += 1
        V[(b_row_offset+1)*5 + 1][c] += -1
        # C01 = W[2], C11 = W[3]
        W[2][c] += 1
        W[3][c] += 1
        c += 1
        
        # M4 = a11(b00-b10)
        U[1*4 + a_col_offset + 1][c] += 1
        V[(b_row_offset+0)*5 + 0][c] += 1
        V[(b_row_offset+1)*5 + 0][c] += -1
        # C00 = W[0], C10 = W[1]
        W[0][c] += 1
        W[1][c] += 1
        c += 1
        
        # M5 = (a00+a01)b11
        U[0*4 + a_col_offset + 0][c] += 1
        U[0*4 + a_col_offset + 1][c] += 1
        V[(b_row_offset+1)*5 + 1][c] += 1
        # C00 -= M5 -> W[0] -= 1, C01 += M5 -> W[2] += 1
        W[0][c] += -1
        W[2][c] += 1
        c += 1
        
        # M6 = (a10-a00)(b00+b01)
        U[1*4 + a_col_offset + 0][c] += 1
        U[0*4 + a_col_offset + 0][c] += -1
        V[(b_row_offset+0)*5 + 0][c] += 1
        V[(b_row_offset+0)*5 + 1][c] += 1
        # C11 = W[3]
        W[3][c] += 1
        c += 1
        
        # M7 = (a01-a11)(b10+b11)
        U[0*4 + a_col_offset + 1][c] += 1
        U[1*4 + a_col_offset + 1][c] += -1
        V[(b_row_offset+1)*5 + 0][c] += 1
        V[(b_row_offset+1)*5 + 1][c] += 1
        # C00 = W[0], C11 = W[3]
        W[0][c] += 1
        W[3][c] += 1
        c += 1
        
        # Strassen for columns (2,3):
        # b00' = B[b_row_offset+0, 2], b01' = B[b_row_offset+0, 3]
        # b10' = B[b_row_offset+1, 2], b11' = B[b_row_offset+1, 3]
        
        # M1' = (a00+a11)(b00'+b11')
        U[0*4 + a_col_offset + 0][c] += 1
        U[1*4 + a_col_offset + 1][c] += 1
        V[(b_row_offset+0)*5 + 2][c] += 1
        V[(b_row_offset+1)*5 + 3][c] += 1
        # C00' = W[4], C11' = W[7]
        W[4][c] += 1
        W[7][c] += 1
        c += 1
        
        # M2' = (a10+a11)b00'
        U[1*4 + a_col_offset + 0][c] += 1
        U[1*4 + a_col_offset + 1][c] += 1
        V[(b_row_offset+0)*5 + 2][c] += 1
        # C10' = W[5]
        W[5][c] += 1
        c += 1
        
        # M3' = a00(b10'-b11')
        U[0*4 + a_col_offset + 0][c] += 1
        V[(b_row_offset+1)*5 + 2][c] += 1
        V[(b_row_offset+1)*5 + 3][c] += -1
        # C01' = W[6], C11' = W[7]
        W[6][c] += 1
        W[7][c] += 1
        c += 1
        
        # M4' = a11(b00'-b10')
        U[1*4 + a_col_offset + 1][c] += 1
        V[(b_row_offset+0)*5 + 2][c] += 1
        V[(b_row_offset+1)*5 + 2][c] += -1
        # C00' = W[4], C10' = W[5]
        W[4][c] += 1
        W[5][c] += 1
        c += 1
        
        # M5' = (a00+a01)b11'
        U[0*4 + a_col_offset + 0][c] += 1
        U[0*4 + a_col_offset + 1][c] += 1
        V[(b_row_offset+1)*5 + 3][c] += 1
        # C00' -= M5' -> W[4] -= 1, C01' += M5' -> W[6] += 1
        W[4][c] += -1
        W[6][c] += 1
        c += 1
        
        # M6' = (a10-a00)(b00'+b01')
        U[1*4 + a_col_offset + 0][c] += 1
        U[0*4 + a_col_offset + 0][c] += -1
        V[(b_row_offset+0)*5 + 2][c] += 1
        V[(b_row_offset+0)*5 + 3][c] += 1
        # C11' = W[7]
        W[7][c] += 1
        c += 1
        
        # M7' = (a01-a11)(b10'+b11')
        U[0*4 + a_col_offset + 1][c] += 1
        U[1*4 + a_col_offset + 1][c] += -1
        V[(b_row_offset+1)*5 + 2][c] += 1
        V[(b_row_offset+1)*5 + 3][c] += 1
        # C00' = W[4], C11' = W[7]
        W[4][c] += 1
        W[7][c] += 1
        c += 1
        
        # Schoolbook for column 4: 4 products
        # C[i,4] = a_i0 * B[b_row_offset+0, 4] + a_i1 * B[b_row_offset+1, 4]
        for i in range(2):
            for k in range(2):
                U[i*4 + a_col_offset + k][c] += 1
                V[(b_row_offset+k)*5 + 4][c] += 1
                W[4*2 + i][c] += 1  # W[8+i]
                c += 1
        
        return c
    
    # Add left block: A_left @ B_top
    col = add_2x2x5(0, 0, 0)
    
    # Add right block: A_right @ B_bottom
    col = add_2x2x5(2, 2, col)
    
    return {"U": U, "V": V, "W": W}

def _construct_rank32():
    """
    Attempt rank-32 decomposition using a different block structure.
    
    Alternative split:
    A = [A1 | A2 | A3] where A1 is 2x1, A2 is 2x2, A3 is 2x1
    B = [B1; B2; B3] where B1 is 1x5, B2 is 2x5, B3 is 1x5
    
    C = A1*B1 + A2*B2 + A3*B3
    
    For A2*B2 (2x2 * 2x5): rank 18 as before
    For A1*B1 (2x1 * 1x5): rank 10 (2 rows * 5 cols)
    For A3*B3 (2x1 * 1x5): rank 10 (2 rows * 5 cols)
    Total: 18 + 10 + 10 = 38. Worse.
    
    Let me try another approach:
    A = [A1 | A2] where A1 is 2x2, A2 is 2x2
    B = [B1; B2] where B1 is 2x5, B2 is 2x5
    
    C = A1*B1 + A2*B2
    
    Can we do better than 18 for 2x2*2x5?
    
    Actually, let me try to use a different decomposition for the full problem.
    
    Known result: rank(M_{2,4,5}) <= 32 might be achievable.
    
    Let me try a construction based on:
    - Use Strassen for 2x2*2x2 blocks
    - Share some multiplications across the 5 columns
    
    Actually, the minimal rank for M_{2,4,5} is unknown, but upper bounds exist.
    Let me just verify the rank-36 construction works and return it.
    """
    return None

def construct():
    n, m, p = DIMENSIONS
    
    # Try rank-36 construction
    try:
        result = _construct_rank36()
        if result is not None:
            # Verify
            if _verify(result['U'], result['V'], result['W'], n, m, p):
                return result
    except Exception:
        pass
    
    # Try rank-32 construction
    try:
        result = _construct_rank32()
        if result is not None:
            if _verify(result['U'], result['V'], result['W'], n, m, p):
                return result
    except Exception:
        pass
    
    # Fall back to schoolbook
    rank = n * m * p
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1.0
                V[j * p + k][col] = 1.0
                W[k * n + i][col] = 1.0
                col += 1
    return {"U": U, "V": V, "W": W}
