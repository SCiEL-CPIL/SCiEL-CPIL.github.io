import time
import numpy as np

DIMENSIONS = [3, 4, 7]

def construct():
    n, m, p = DIMENSIONS
    rank = 0
    U = {}
    V = {}
    W = {}
    
    # Helper to add a block decomposition
    # i0, j0, k0: start indices in A, B, C
    # ni, nj, nk: dimensions of the block
    # decomp: list of (u_coeffs, v_coeffs, w_coeffs)
    # u_coeffs: dict (i, j) -> val
    # v_coeffs: dict (j, k) -> val
    # w_coeffs: dict (k, i) -> val
    def add_block(i0, j0, k0, ni, nj, nk, decomp):
        nonlocal rank
        for u_c, v_c, w_c in decomp:
            col = rank
            for (i, j), val in u_c.items():
                key = (i0 + i, j0 + j)
                if key not in U:
                    U[key] = {}
                U[key][col] = val
            for (j, k), val in v_c.items():
                key = (j0 + j, k0 + k)
                if key not in V:
                    V[key] = {}
                V[key][col] = val
            for (k, i), val in w_c.items():
                key = (k0 + k, i0 + i)
                if key not in W:
                    W[key] = {}
                W[key][col] = val
            rank += 1

    # Strassen 2x2x2
    def strassen_222():
        # M1 = (A11+A22)(B11+B22)
        # M2 = (A21+A22)B11
        # M3 = A11(B12-B22)
        # M4 = A22(B21-B11)
        # M5 = (A11+A12)B22
        # M6 = (A21-A11)(B11+B12)
        # M7 = (A12-A22)(B21+B22)
        
        decomp = []
        
        # M1
        u1 = {(0,0): 1, (1,1): 1}
        v1 = {(0,0): 1, (0,1): 1, (1,0): 1, (1,1): 1}
        w1 = {(0,0): 1, (1,1): 1}
        decomp.append((u1, v1, w1))
        
        # M2
        u2 = {(1,0): 1, (1,1): 1}
        v2 = {(0,0): 1}
        w2 = {(0,0): 1, (1,0): 1}
        decomp.append((u2, v2, w2))
        
        # M3
        u3 = {(0,0): 1}
        v3 = {(0,1): 1, (1,1): -1}
        w3 = {(0,1): 1}
        decomp.append((u3, v3, w3))
        
        # M4
        u4 = {(1,1): 1}
        v4 = {(1,0): 1, (0,0): -1}
        w4 = {(0,0): 1, (1,0): 1}
        decomp.append((u4, v4, w4))
        
        # M5
        u5 = {(0,0): 1, (0,1): 1}
        v5 = {(1,1): 1}
        w5 = {(0,0): -1, (1,1): 1}
        decomp.append((u5, v5, w5))
        
        # M6
        u6 = {(1,0): 1, (0,0): -1}
        v6 = {(0,0): 1, (0,1): 1}
        w6 = {(1,1): 1}
        decomp.append((u6, v6, w6))
        
        # M7
        u7 = {(0,1): 1, (1,1): -1}
        v7 = {(1,0): 1, (1,1): 1}
        w7 = {(0,0): 1, (1,1): 1}
        decomp.append((u7, v7, w7))
        
        return decomp

    # Schoolbook for 2x2x1
    def schoolbook_221():
        decomp = []
        # A11 B11
        u1 = {(0,0): 1}
        v1 = {(0,0): 1}
        w1 = {(0,0): 1}
        decomp.append((u1, v1, w1))
        
        # A12 B21
        u2 = {(0,1): 1}
        v2 = {(1,0): 1}
        w2 = {(0,0): 1}
        decomp.append((u2, v2, w2))
        
        # A21 B11
        u3 = {(1,0): 1}
        v3 = {(0,0): 1}
        w3 = {(1,0): 1}
        decomp.append((u3, v3, w3))
        
        # A22 B21
        u4 = {(1,1): 1}
        v4 = {(1,0): 1}
        w4 = {(1,0): 1}
        decomp.append((u4, v4, w4))
        
        return decomp

    # Schoolbook for 1x2x2
    def schoolbook_122():
        decomp = []
        # A11 B11
        u1 = {(0,0): 1}
        v1 = {(0,0): 1}
        w1 = {(0,0): 1}
        decomp.append((u1, v1, w1))
        
        # A12 B21
        u2 = {(0,1): 1}
        v2 = {(1,0): 1}
        w2 = {(0,0): 1}
        decomp.append((u2, v2, w2))
        
        # A11 B12
        u3 = {(0,0): 1}
        v3 = {(0,1): 1}
        w3 = {(1,0): 1}
        decomp.append((u3, v3, w3))
        
        # A12 B22
        u4 = {(0,1): 1}
        v4 = {(1,1): 1}
        w4 = {(1,0): 1}
        decomp.append((u4, v4, w4))
        
        return decomp

    # Schoolbook for 1x4x2
    def schoolbook_142():
        decomp = []
        for j in range(4):
            for k in range(2):
                u = {(0,j): 1}
                v = {(j,k): 1}
                w = {(k,0): 1}
                decomp.append((u, v, w))
        return decomp

    # Schoolbook for 1x4x1
    def schoolbook_141():
        decomp = []
        for j in range(4):
            u = {(0,j): 1}
            v = {(j,0): 1}
            w = {(0,0): 1}
            decomp.append((u, v, w))
        return decomp

    # Combine blocks
    # A1 (2x4), B1 (4x4) -> C1 (2x4)
    # A1 = A11 (2x2) A12 (2x2)
    # B1 = B11 (2x2) B12 (2x2) on top? No, B1 is 4x4.
    # B1 = [B11; B12] where B11 is 2x4? No.
    # Let's stick to the 2x2x2 blocks.
    
    # Block 1: A1 (2x4) * B1 (4x4)
    # Split B1 into B1a (4x2) and B1b (4x2)
    # A1 * B1a (2x4x2)
    # Split A1 into A11 (2x2) and A12 (2x2)
    # Split B1a into B1a1 (2x2) and B1a2 (2x2)
    # A11 * B1a1 (2x2x2) at (0,0,0)
    add_block(0, 0, 0, 2, 2, 2, strassen_222())
    # A12 * B1a2 (2x2x2) at (0,2,0)
    add_block(0, 2, 0, 2, 2, 2, strassen_222())
    
    # A1 * B1b (2x4x2)
    # A11 * B1b1 (2x2x2) at (0,0,2)
    add_block(0, 0, 2, 2, 2, 2, strassen_222())
    # A12 * B1b2 (2x2x2) at (0,2,2)
    add_block(0, 2, 2, 2, 2, 2, strassen_222())
    
    # Block 2: A1 (2x4) * B2 (4x3)
    # Split B2 into B2a (4x2) and B2b (4x1)
    # A1 * B2a (2x4x2)
    add_block(0, 0, 4, 2, 2, 2, strassen_222())
    add_block(0, 2, 4, 2, 2, 2, strassen_222())
    
    # A1 * B2b (2x4x1)
    # A11 * B2b1 (2x2x1) at (0,0,6)
    add_block(0, 0, 6, 2, 2, 1, schoolbook_221())
    # A12 * B2b2 (2x2x1) at (0,2,6)
    add_block(0, 2, 6, 2, 2, 1, schoolbook_221())
    
    # Block 3: A2 (1x4) * B1 (4x4)
    # A2 * B1a (1x4x2)
    add_block(2, 0, 0, 1, 4, 2, schoolbook_142())
    # A2 * B1b (1x4x2)
    add_block(2, 0, 2, 1, 4, 2, schoolbook_142())
    
    # Block 4: A2 (1x4) * B2 (4x3)
    # A2 * B2a (1x4x2)
    add_block(2, 0, 4, 1, 4, 2, schoolbook_142())
    # A2 * B2b (1x4x1)
    add_block(2, 0, 6, 1, 4, 1, schoolbook_141())

    # Convert dicts to lists
    U_list = [[0] * rank for _ in range(n * m)]
    V_list = [[0] * rank for _ in range(m * p)]
    W_list = [[0] * rank for _ in range(p * n)]
    
    for (i, j), cols in U.items():
        for c, val in cols.items():
            U_list[i * m + j][c] = val
            
    for (j, k), cols in V.items():
        for c, val in cols.items():
            V_list[j * p + k][c] = val
            
    for (k, i), cols in W.items():
        for c, val in cols.items():
            W_list[k * n + i][c] = val
            
    return {"U": U_list, "V": V_list, "W": W_list}
