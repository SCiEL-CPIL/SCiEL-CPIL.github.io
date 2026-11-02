DIMENSIONS = [2, 4, 8]

def construct():
    n, m, p = DIMENSIONS
    # n=2, m=4, p=8
    # Tile into 2x2 blocks
    # A is 2x4: [A11 A12] where each is 2x2
    # B is 4x8: 2 rows of 4 blocks, each 2x2
    # C is 2x8: 1 row of 4 blocks, each 2x2
    
    # Strassen for 2x2 * 2x2:
    # M = X * Y, X,Y are 2x2
    # 7 mults:
    # t1 = (x00+x11) * (y00+y11)
    # t2 = (x10+x11) * y00
    # t3 = x00 * (y01-y11)
    # t4 = x11 * (y10-y00)
    # t5 = (x00+x01) * y11
    # t6 = (x10-x00) * (y00+y01)
    # t7 = (x01-x11) * (y10+y11)
    #
    # Result:
    # m00 = t1 + t4 - t5 + t7
    # m01 = t3 + t5
    # m10 = t2 + t4
    # m11 = t1 - t2 + t3 + t6
    
    # In tensor decomposition, each mult corresponds to a rank-1 term.
    # For A-block ab (0 or 1) and B-block bb (0..3):
    # X = A[ab], Y = B[bb]
    # The 7 terms have specific U, V, W patterns.
    
    rank = 2 * 4 * 7  # 2 A-blocks, 4 B-blocks, 7 Strassen mults each
    U = [[0] * rank for _ in range(n * m)]  # 8 x rank
    V = [[0] * rank for _ in range(m * p)]  # 32 x rank
    W = [[0] * rank for _ in range(p * n)]  # 16 x rank
    
    col = 0
    for ab in range(2):  # A block: 0 or 1
        for bb in range(4):  # B block: 0..3
            # Strassen mult 1: (x00+x11) * (y00+y11)
            # U entries: for (i,j) in A-block ab
            # A-block ab covers columns 2*ab to 2*ab+1
            # x00 = A[0, 2*ab], x11 = A[1, 2*ab+1]
            # So U[(0, 2*ab), col] += 1, U[(1, 2*ab+1), col] += 1
            # V entries: B-block bb covers rows 2*0 to 2*0+1? No.
            # B is 4x8. B-block bb: row range? 
            # B has 4 rows, 8 cols. Tiling: 2 row-blocks, 4 col-blocks.
            # B-block (rb, cb) where rb in 0..1, cb in 0..3
            # But we're iterating bb from 0..3, which should be the col-block index.
            # Wait, I need to re-examine.
            
            # Let me re-index. B is 4x8. We tile into 2x2 blocks.
            # Row blocks: 2 (rows 0-1, rows 2-3)
            # Col blocks: 4 (cols 0-1, 2-3, 4-5, 6-7)
            # So B has 2*4 = 8 blocks.
            # But A is 2x4, so 1 row-block, 2 col-blocks.
            # C = A*B: for each col-block of B, we do A * B_colblock.
            # A_colblock0 * B_colblock0_col0 + A_colblock1 * B_colblock0_col1... 
            # No, this is getting confused.
            
            pass
    
    # Let me restart with clearer indexing.
    # A: 2 rows, 4 cols. Block into 2x2: A[0:2, 0:2] and A[0:2, 2:4]
    # B: 4 rows, 8 cols. Block into 2x2: B[0:2, 0:2], B[0:2, 2:4], B[0:2, 4:6], B[0:2, 6:8],
    #                                  B[2:4, 0:2], B[2:4, 2:4], B[2:4, 4:6], B[2:4, 6:8]
    # C: 2 rows, 8 cols. C[:, 0:2] = A[:, 0:2] * B[0:2, 0:2] + A[:, 2:4] * B[2:4, 0:2]
    #    C[:, 2:4] = A[:, 0:2] * B[0:2, 2:4] + A[:, 2:4] * B[2:4, 2:4]
    #    etc.
    
    # So for each output col-block cb (0..3):
    #   C[:, 2cb:2cb+2] = A[:, 0:2] * B[0:2, 2cb:2cb+2] + A[:, 2:4] * B[2:4, 2cb:2cb+2]
    
    # Each 2x2*2x2 product uses 7 mults.
    # Total: 4 col-blocks * 2 (A-blocks) * 7 = 56 mults.
    
    rank = 56
    U = [[0] * rank for _ in range(n * m)]  # 8 x 56
    V = [[0] * rank for _ in range(m * p)]  # 32 x 56
    W = [[0] * rank for _ in range(p * n)]  # 16 x 56
    
    col = 0
    for cb in range(4):  # output col-block
        for ab in range(2):  # A col-block (0 or 1)
            rb = ab  # B row-block: if ab=0, use B rows 0-1; if ab=1, use B rows 2-3
            # A-block: rows 0-1, cols 2*ab to 2*ab+1
            # B-block: rows 2*rb to 2*rb+1, cols 2*cb to 2*cb+1
            # C-block: rows 0-1, cols 2*cb to 2*cb+1
            
            # Strassen 7 mults
            # t1: (x00+x11)*(y00+y11)
            U[0 * m + 2*ab][col] = 1    # x00 = A[0, 2ab]
            U[1 * m + 2*ab+1][col] = 1  # x11 = A[1, 2ab+1]
            V[2*rb * p + 2*cb][col] = 1    # y00 = B[2rb, 2cb]
            V[(2*rb+1) * p + 2*cb+1][col] = 1  # y11 = B[2rb+1, 2cb+1]
            # W: for C[0,2cb] and C[1,2cb+1]... 
            # m00 = t1 + ..., m11 = t1 + ...
            # t1 contributes to m00 and m11
            # m00 = C[0, 2cb], m11 = C[1, 2cb+1]
            W[2*cb * n + 0][col] = 1  # C[0, 2cb], index = 2cb*2 + 0
            W[(2*cb+1) * n + 1][col] = 1  # C[1, 2cb+1], index = (2cb+1)*2 + 1
            col += 1
            
            # t2: (x10+x11)*y00
            U[1 * m + 2*ab][col] = 1    # x10 = A[1, 2ab]
            U[1 * m + 2*ab+1][col] = 1  # x11
            V[2*rb * p + 2*cb][col] = 1  # y00
            # m10 = t2 + ..., m11 = -t2 + ...
            # m10 = C[1, 2cb], m11 = C[1, 2cb+1]
            W[2*cb * n + 1][col] = 1    # C[1, 2cb]
            W[(2*cb+1) * n + 1][col] = -1  # C[1, 2cb+1]
            col += 1
            
            # t3: x00*(y01-y11)
            U[0 * m + 2*ab][col] = 1    # x00
            V[2*rb * p + 2*cb+1][col] = 1    # y01
            V[(2*rb+1) * p + 2*cb+1][col] = -1  # -y11
            # m01 = t3 + ..., m11 = t3 + ...
            # m01 = C[0, 2cb+1], m11 = C[1, 2cb+1]
            W[2*cb * n + 1][col] = 1    # C[0, 2cb+1]
            W[(2*cb+1) * n + 1][col] = 1  # C[1, 2cb+1]
            col += 1
            
            # t4: x11*(y10-y00)
            U[1 * m + 2*ab+1][col] = 1  # x11
            V[(2*rb+1) * p + 2*cb][col] = 1  # y10
            V[2*rb * p + 2*cb][col] = -1    # -y00
            # m00 = t4 + ..., m10 = t4 + ...
            # m00 = C[0, 2cb], m10 = C[1, 2cb]
            W[2*cb * n + 0][col] = 1    # C[0, 2cb]
            W[2*cb * n + 1][col] = 1    # C[1, 2cb]
            col += 1
            
            # t5: (x00+x01)*y11
            U[0 * m + 2*ab][col] = 1    # x00
            U[0 * m + 2*ab+1][col] = 1  # x01
            V[(2*rb+1) * p + 2*cb+1][col] = 1  # y11
            # m00 = -t5 + ..., m01 = t5 + ...
            # m00 = C[0, 2cb], m01 = C[0, 2cb+1]
            W[2*cb * n + 0][col] = -1   # C[0, 2cb]
            W[2*cb * n + 1][col] = 1    # C[0, 2cb+1]
            col += 1
            
            # t6: (x10-x00)*(y00+y01)
            U[1 * m + 2*ab][col] = 1    # x10
            U[0 * m + 2*ab][col] = -1   # -x00
            V[2*rb * p + 2*cb][col] = 1    # y00
            V[2*rb * p + 2*cb+1][col] = 1  # y01
            # m11 = t6 + ...
            # m11 = C[1, 2cb+1]
            W[(2*cb+1) * n + 1][col] = 1  # C[1, 2cb+1]
            col += 1
            
            # t7: (x01-x11)*(y10+y11)
            U[0 * m + 2*ab+1][col] = 1  # x01
            U[1 * m + 2*ab+1][col] = -1 # -x11
            V[(2*rb+1) * p + 2*cb][col] = 1  # y10
            V[(2*rb+1) * p + 2*cb+1][col] = 1  # y11
            # m00 = t7 + ..., m10 = ... 
            # m00 = C[0, 2cb], m10? No: m10 = t2 + t4, no t7.
            # m00 = t1 + t4 - t5 + t7
            # m01 = t3 + t5
            # m10 = t2 + t4
            # m11 = t1 - t2 + t3 + t6
            # So t7 only in m00.
            W[2*cb * n + 0][col] = 1    # C[0, 2cb]
            col += 1
    
    return {"U": U, "V": V, "W": W}
