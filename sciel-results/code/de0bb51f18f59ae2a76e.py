"""Hierarchical Strassen decomposition for 2x4 * 4x8 matrix multiplication."""
import time

DIMENSIONS = [2, 4, 8]

_DEADLINE = time.time() + 120

def _strassen_2x2():
    """
    Return the 7 rank-1 tensors for 2x2 * 2x2 matrix multiplication.
    Each product is (u_dict, v_dict, w_dict) where each dict maps (row,col) -> coeff.
    """
    products = [
        # M1 = (A00+A11)*(B00+B11) -> C00+=1, C11+=1
        ({(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}),
        # M2 = (A10+A11)*B00 -> C10+=1
        ({(1,0):1, (1,1):1}, {(0,0):1}, {(1,0):1}),
        # M3 = A00*(B10-B00) -> C01+=1
        ({(0,0):1}, {(1,0):1, (0,0):-1}, {(0,1):1}),
        # M4 = A11*(B01-B11) -> C00+=1, C10+=1
        ({(1,1):1}, {(0,1):1, (1,1):-1}, {(0,0):1, (1,0):1}),
        # M5 = (A00+A01)*B11 -> C00+=-1, C01+=1
        ({(0,0):1, (0,1):1}, {(1,1):1}, {(0,0):-1, (0,1):1}),
        # M6 = (A10-A00)*(B00+B01) -> C11+=1
        ({(1,0):1, (0,0):-1}, {(0,0):1, (0,1):1}, {(1,1):1}),
        # M7 = (A01-A11)*(B10+B11) -> C00+=1, C11+=1
        ({(0,1):1, (1,1):-1}, {(1,0):1, (1,1):1}, {(0,0):1, (1,1):1}),
    ]
    return products

def _build_2x4_4x8_decomposition():
    """
    Build the full 2x4 * 4x8 tensor decomposition using Strassen.
    Rank = 56.
    
    A (2x4) split into A1 (2x2, cols 0-1) and A2 (2x2, cols 2-3).
    B (4x8) split into B_top (2x8, rows 0-1) and B_bot (2x8, rows 2-3).
    C = A1*B_top + A2*B_bot.
    Each 2x2*2x8 uses 4 Strassen 2x2*2x2 decompositions (rank 7 each) = rank 28.
    Total rank = 56.
    """
    n, m, p = 2, 4, 8
    total_rank = 56
    
    U = [[0]*total_rank for _ in range(n*m)]   # (4, 56)
    V = [[0]*total_rank for _ in range(m*p)]   # (16, 56)
    W = [[0]*total_rank for _ in range(p*n)]   # (16, 56)
    
    sub_products = _strassen_2x2()
    
    col = 0
    for half in range(2):
        a_col_offset = half * 2
        b_row_offset = half * 2
        
        for block in range(4):
            b_col_offset = block * 2
            
            for u_sub, v_sub, w_sub in sub_products:
                for (sr, sc), coeff in u_sub.items():
                    a_idx = sr * m + (a_col_offset + sc)
                    U[a_idx][col] += coeff
                
                for (sr, sc), coeff in v_sub.items():
                    b_idx = (b_row_offset + sr) * p + (b_col_offset + sc)
                    V[b_idx][col] += coeff
                
                for (sr, sc), coeff in w_sub.items():
                    c_idx = (block * 2 + sc) * n + sr
                    W[c_idx][col] += coeff
                
                col += 1
    
    return U, V, W

def construct():
    global _DEADLINE
    if time.time() > _DEADLINE - 5:
        n, m, p = DIMENSIONS
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
    
    U, V, W = _build_2x4_4x8_decomposition()
    return {"U": U, "V": V, "W": W}
