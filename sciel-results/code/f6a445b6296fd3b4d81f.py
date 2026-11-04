"""Matrix multiplication 4,4,8 - Strassen-based decomposition."""
DIMENSIONS = [4, 4, 8]

def construct():
    n, m, p = DIMENSIONS  # 4, 4, 8
    
    # Use Strassen for each 4x4 * 4x4 block.
    # 4x4*4x4 via Strassen (one level) has rank 28.
    # Two independent blocks: total rank 56.
    
    rank = 56
    U = [[0] * rank for _ in range(n * m)]  # 16 x 56
    V = [[0] * rank for _ in range(m * p)]  # 32 x 56
    W = [[0] * rank for _ in range(p * n)]  # 32 x 56
    
    # Strassen 4x4*4x4: partition into 2x2 blocks.
    # A = [[A11, A12], [A21, A22]], B = [[B11, B12], [B21, B22]]
    # M1 = (A11+A22)(B11+B22), M2 = (A21+A22)B11, M3 = A11(B12-B22)
    # M4 = A22(B21-B11), M5 = (A11+A12)B22, M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    # C11 = M1+M4-M5+M7, C12 = M3+M5, C21 = M2+M4, C22 = M1-M2+M3+M6
    
    def add_strassen_block(col_offset, b_col_base):
        """Add 28 rank-1 terms for one 4x4*4x4 Strassen block.
        
        col_offset: starting column in U,V,W
        b_col_base: 0 or 4, the base column offset for B
        """
        col = col_offset
        block_off = {11: (0,0), 12: (0,2), 21: (2,0), 22: (2,2)}
        
        # Each M term is a 2x2*2x2 product expanded into 4 scalar products.
        # For M = X*Y where X,Y are 2x2: M[a,c] = sum_b X[a,b]*Y[b,c]
        
        def process(x_coefs, y_coefs, c_block, sign):
            """x_coefs: list of (coef, block), y_coefs: list of (coef, block)"""
            nonlocal col
            for (cx, bx) in x_coefs:
                for (cy, by) in y_coefs:
                    for a in range(2):
                        for b in range(2):
                            for c in range(2):
                                ar = block_off[bx][0] + a
                                ac = block_off[bx][1] + b
                                br = block_off[by][0] + b
                                bc = block_off[by][1] + c + b_col_base
                                cr = block_off[c_block][0] + c
                                cc = block_off[c_block][1] + a
                                u_idx = ar * 4 + ac
                                v_idx = br * 8 + bc
                                w_idx = cr * 4 + cc
                                val = cx * cy * sign
                                U[u_idx][col] += val
                                V[v_idx][col] += val
                                W[w_idx][col] += val
                                col += 1
        
        # M1 = (A11+A22)(B11+B22) -> C11(+), C22(+)
        process([(1,11),(1,22)], [(1,11),(1,22)], 11, 1)
        process([(1,11),(1,22)], [(1,11),(1,22)], 22, 1)
        # M2 = (A21+A22)B11 -> C21(+), C22(-)
        process([(1,21),(1,22)], [(1,11)], 21, 1)
        process([(1,21),(1,22)], [(1,11)], 22, -1)
        # M3 = A11(B12-B22) -> C12(+), C22(+)
        process([(1,11)], [(1,12),(-1,22)], 12, 1)
        process([(1,11)], [(1,12),(-1,22)], 22, 1)
        # M4 = A22(B21-B11) -> C11(+), C21(+)
        process([(1,22)], [(1,21),(-1,11)], 11, 1)
        process([(1,22)], [(1,21),(-1,11)], 21, 1)
        # M5 = (A11+A12)B22 -> C11(-), C12(+)
        process([(1,11),(1,12)], [(1,22)], 11, -1)
        process([(1,11),(1,12)], [(1,22)], 12, 1)
        # M6 = (A21-A11)(B11+B12) -> C22(+)
        process([(1,21),(-1,11)], [(1,11),(1,12)], 22, 1)
        # M7 = (A12-A22)(B21+B22) -> C11(+)
        process([(1,12),(-1,22)], [(1,21),(1,22)], 11, 1)
        
        return col
    
    col = add_strassen_block(0, 0)
    add_strassen_block(col, 4)
    
    return {"U": U, "V": V, "W": W}
