import time

def construct():
    """
    Rank-42 exact decomposition for 2x4 @ 4x6 matrix multiplication.
    
    Strategy: Split the 4x4 middle dimension into two 2x2 blocks (left and right halves).
    This gives C = A_left @ B_top + A_right @ B_bottom, where each term is a 2x2 @ 2x6 multiplication.
    Each 2x2 @ 2x6 is decomposed into three 2x2 @ 2x2 blocks using Strassen's algorithm (rank 7 each).
    Total rank per 2x2 @ 2x6 block = 3 * 7 = 21.
    Total rank = 21 + 21 = 42.
    
    U: (n*m, rank) = (8, 42)
    V: (m*p, rank) = (24, 42)
    W: (p*n, rank) = (12, 42)
    
    W index convention: W[k*n+i, r] where k is column index (0..5), i is row index (0..1)
    So W index = k*2 + i for output C[i][k]
    """
    n, m, p = 2, 4, 6
    rank = 42
    
    U = [[0] * rank for _ in range(n * m)]  # 8 x 42
    V = [[0] * rank for _ in range(m * p)]  # 24 x 42
    W = [[0] * rank for _ in range(p * n)]  # 12 x 42
    
    col = 0
    
    # Strassen 2x2x2 core: 7 multiplications
    # For X @ Y where X,Y are 2x2:
    # M1 = (X00+X11)*(Y00+Y11)
    # M2 = (X10+X11)*Y00
    # M3 = X00*(Y01-Y11)
    # M4 = X11*(Y10-Y00)
    # M5 = (X00+X01)*Y11
    # M6 = (X10-X00)*(Y00+Y01)
    # M7 = (X01-X11)*(Y10+Y11)
    # Result:
    # Z00 = M1+M4-M5+M7
    # Z01 = M3+M5
    # Z10 = M2+M4
    # Z11 = M1-M2+M3+M6
    
    strassen_terms = [
        # M1: (A00+A11)*(B00+B11)
        {
            'A': [(0,0,1), (1,1,1)],
            'B': [(0,0,1), (1,1,1)],
            'W': [(0,0,1), (1,1,1)]
        },
        # M2: (A10+A11)*B00
        {
            'A': [(1,0,1), (1,1,1)],
            'B': [(0,0,1)],
            'W': [(1,0,1), (1,1,-1)]
        },
        # M3: A00*(B01-B11)
        {
            'A': [(0,0,1)],
            'B': [(0,1,1), (1,1,-1)],
            'W': [(0,1,1), (1,1,1)]
        },
        # M4: A11*(B10-B00)
        {
            'A': [(1,1,1)],
            'B': [(1,0,1), (0,0,-1)],
            'W': [(0,0,1), (1,0,1)]
        },
        # M5: (A00+A01)*B11
        {
            'A': [(0,0,1), (0,1,1)],
            'B': [(1,1,1)],
            'W': [(0,0,-1), (0,1,1)]
        },
        # M6: (A10-A00)*(B00+B01)
        {
            'A': [(1,0,1), (0,0,-1)],
            'B': [(0,0,1), (0,1,1)],
            'W': [(1,1,1)]
        },
        # M7: (A01-A11)*(B10+B11)
        {
            'A': [(0,1,1), (1,1,-1)],
            'B': [(1,0,1), (1,1,1)],
            'W': [(0,0,1)]
        }
    ]
    
    def apply_strassen(a_base, b_base, out_base):
        nonlocal col
        for term in strassen_terms:
            for (ar, ac, sign) in term['A']:
                u_idx = (a_base[0] + ar) * m + (a_base[1] + ac)
                U[u_idx][col] += sign
            
            for (br, bc, sign) in term['B']:
                v_idx = (b_base[0] + br) * p + (b_base[1] + bc)
                V[v_idx][col] += sign
            
            for (or_, oc, sign) in term['W']:
                w_idx = (out_base[1] + oc) * n + (out_base[0] + or_)
                W[w_idx][col] += sign
            
            col += 1
    
    # Left half: A[:,0:2] @ B[0:2,:]
    for sub in range(3):
        b_col_offset = 2 * sub
        out_col_offset = 2 * sub
        apply_strassen((0, 0), (0, b_col_offset), (0, out_col_offset))
    
    # Right half: A[:,2:4] @ B[2:4,:]
    for sub in range(3):
        b_col_offset = 2 * sub
        out_col_offset = 2 * sub
        apply_strassen((0, 2), (2, b_col_offset), (0, out_col_offset))
    
    assert col == rank, f"Expected rank {rank}, got {col}"
    
    return {"U": U, "V": V, "W": W}
