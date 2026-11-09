import numpy as np

def construct():
    n, m, p = 4, 4, 4
    rank = 49
    
    # 2x2 Strassen coefficients
    # U_elem[t] corresponds to A[0,0], A[0,1], A[1,0], A[1,1]
    U_2x2 = np.array([
        [1, 0, 0, 1],
        [0, 0, 1, 1],
        [1, 0, 0, 0],
        [0, 0, 0, 1],
        [1, 1, 0, 0],
        [-1, 0, 1, 0],
        [0, 1, 0, -1]
    ])
    
    V_2x2 = np.array([
        [1, 0, 0, 1],
        [1, 0, 0, 0],
        [0, 1, 0, -1],
        [-1, 0, 1, 0],
        [0, 0, 0, 1],
        [1, 1, 0, 0],
        [0, 0, 1, 1]
    ])
    
    # W_2x2[t] is 2x2 matrix for C[0,0], C[0,1], C[1,0], C[1,1]
    # Flattened to 4 elements in row-major order: C00, C01, C10, C11
    W_2x2 = np.array([
        [1, 0, 0, 1],    # C00+=M, C11+=M
        [0, 0, 1, -1],   # C10+=M, C11-=M
        [0, 1, 0, 1],    # C01+=M, C11+=M
        [1, 0, 1, 0],    # C00+=M, C10+=M
        [-1, 1, 0, 0],   # C00-=M, C01+=M
        [0, 0, 0, 1],    # C11+=M
        [1, 0, 0, 0]     # C00+=M
    ])
    
    U = np.zeros((16, 49), dtype=int)
    V = np.zeros((16, 49), dtype=int)
    W = np.zeros((16, 49), dtype=int)
    
    for s in range(7):
        for t in range(7):
            col = s * 7 + t
            
            # U construction
            # A is 4x4. Blocks are 2x2.
            # Block (bi, bj) contains A[bi*2:bi*2+2, bj*2:bj*2+2]
            # U_2x2[t] gives coefficients for the 4 elements of the 2x2 block of A.
            # We need to map the 2x2 Strassen block indices to 4x4 global indices.
            
            # The 2x2 Strassen decomposition is applied to the "block" matrix A_blocks.
            # A_blocks is a 2x2 matrix of 2x2 blocks.
            # Let A_blocks[i, j] be the block.
            # The Strassen products are M_s = (sum c_ai A_blocks[i,j]) * (sum c_bj B_blocks[j,k])
            
            # We iterate over the Strassen terms s (block level) and t (element level).
            # Actually, the Kronecker product structure means:
            # U_global = U_block \otimes U_elem
            # The indices are (s, t) -> col = s*7 + t.
            
            # For a fixed s (block term) and t (element term):
            # The contribution to U comes from the block coefficient c_as and element coefficient c_at.
            
            # Let's define the block-level coefficients for A, B, C.
            # These are the same as the 2x2 Strassen coefficients, but they apply to blocks.
            
            # A_blocks coefficients for term s:
            # s=0: A00 + A11
            # s=1: A10 + A11
            # s=2: A00
            # s=3: A11
            # s=4: A00 + A01
            # s=5: A10 - A00
            # s=6: A01 - A11
            
            # We can reuse the U_2x2 array structure if we interpret the indices correctly.
            # U_2x2[s] gives coefficients for A00, A01, A10, A11.
            # Let's call these block coefficients c_bs.
            c_bs = U_2x2[s]
            
            # Element coefficients for term t:
            c_et = U_2x2[t] # Note: U_2x2 is used for both block and element levels in the Kronecker product.
            
            # So for each block (bi, bj) with coefficient c_bs[bi*2+bj]:
            # And each element (ei, ej) within that block with coefficient c_et[ei*2+ej]:
            # The global row index is (bi*2+ei)*4 + (bj*2+ej).
            
            for bi in range(2):
                for bj in range(2):
                    c_block = c_bs[bi*2 + bj]
                    if c_block == 0:
                        continue
                    for ei in range(2):
                        for ej in range(2):
                            c_elem = c_et[ei*2 + ej]
                            if c_elem == 0:
                                continue
                            row_a = (bi*2 + ei)*4 + (bj*2 + ej)
                            U[row_a, col] += c_block * c_elem
            
            # V construction
            # B_blocks coefficients for term s:
            # s=0: B00 + B11
            # s=1: B00
            # s=2: B01 - B11
            # s=3: B10 - B00
            # s=4: B11
            # s=5: B00 + B01
            # s=6: B10 + B11
            
            # This is NOT the same as U_2x2.
            # U_2x2 was for A.
            # Let's define V_2x2_block explicitly.
            V_block = np.array([
                [1, 0, 0, 1], # B00+B11
                [1, 0, 0, 0], # B00
                [0, 1, 0, -1], # B01-B11
                [-1, 0, 1, 0], # B10-B00
                [0, 0, 0, 1], # B11
                [1, 1, 0, 0], # B00+B01
                [0, 0, 1, 1]  # B10+B11
            ])
            c_bs_v = V_block[s]
            c_et_v = V_2x2[t] # Element level V coefficients
            
            for bj in range(2):
                for bk in range(2):
                    c_block = c_bs_v[bj*2 + bk]
                    if c_block == 0:
                        continue
                    for ej in range(2):
                        for ek in range(2):
                            c_elem = c_et_v[ej*2 + ek]
                            if c_elem == 0:
                                continue
                            row_b = (bj*2 + ej)*4 + (bk*2 + ek)
                            V[row_b, col] += c_block * c_elem
            
            # W construction
            # C_blocks coefficients for term s:
            # s=0: C00 + C11
            # s=1: C10 - C11
            # s=2: C01 + C11
            # s=3: C00 + C10
            # s=4: -C00 + C01
            # s=5: C11
            # s=6: C00
            
            W_block = np.array([
                [1, 0, 0, 1], # C00+C11
                [0, 0, 1, -1], # C10-C11
                [0, 1, 0, 1], # C01+C11
                [1, 0, 1, 0], # C00+C10
                [-1, 1, 0, 0], # -C00+C01
                [0, 0, 0, 1], # C11
                [1, 0, 0, 0]  # C00
            ])
            c_bs_w = W_block[s]
            c_et_w = W_2x2[t] # Element level W coefficients
            
            # W row index is k*n + i = k*4 + i.
            # i is row in C, k is col in C.
            # Block (bi, bk) -> i in [2*bi, 2*bi+1], k in [2*bk, 2*bk+1].
            
            for bi in range(2):
                for bk in range(2):
                    c_block = c_bs_w[bi*2 + bk]
                    if c_block == 0:
                        continue
                    for ei in range(2):
                        for ek in range(2):
                            c_elem = c_et_w[ei*2 + ek]
                            if c_elem == 0:
                                continue
                            i = bi*2 + ei
                            k = bk*2 + ek
                            row_w = k*4 + i
                            W[row_w, col] += c_block * c_elem

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
