import numpy as np

DIMENSIONS = [4, 4, 6]

def construct():
    n, m, p = DIMENSIONS
    rank = 84
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    # Strassen coefficients for 2x2x2
    # M1 = (X11 + X22)(Y11 + Y22)
    # M2 = (X21 + X22)Y11
    # M3 = X11(Y12 - Y22)
    # M4 = X22(Y21 - Y11)
    # M5 = (X11 + X12)Y22
    # M6 = (X21 - X11)(Y11 + Y12)
    # M7 = (X12 - X22)(Y21 + Y22)
    
    # Z11 = M1 - M4 + M5 - M7
    # Z12 = M3 + M7
    # Z21 = M2 + M4
    # Z22 = M1 - M2 + M3 + M6
    
    # We need to map the 2x2 blocks to the global indices.
    # A is 4x4.
    # A11: rows 0-1, cols 0-1
    # A12: rows 0-1, cols 2-3
    # A21: rows 2-3, cols 0-1
    # A22: rows 2-3, cols 2-3
    
    # B is 4x6.
    # B1j: rows 0-1, cols 2j, 2j+1
    # B2j: rows 2-3, cols 2j, 2j+1
    
    # C is 4x6.
    # C1j: rows 0-1, cols 2j, 2j+1
    # C2j: rows 2-3, cols 2j, 2j+1
    
    # Products for C1j:
    # P1 = A11 * B1j
    # P2 = A12 * B2j
    
    # Products for C2j:
    # P3 = A21 * B1j
    # P4 = A22 * B2j
    
    # Define the 4 products for each j
    # Each product is defined by (A_block, B_block, C_block)
    # A_block is (r_start, c_start)
    # B_block is (r_start, c_start)
    # C_block is (r_start, c_start)
    
    products = []
    for j in range(3):
        # C1j
        # A11 (0,0) * B1j (0, 2j) -> C1j (0, 2j)
        products.append(((0, 0), (0, 2*j), (0, 2*j)))
        # A12 (0,2) * B2j (2, 2j) -> C1j (0, 2j)
        products.append(((0, 2), (2, 2*j), (0, 2*j)))
        # C2j
        # A21 (2,0) * B1j (0, 2j) -> C2j (2, 2j)
        products.append(((2, 0), (0, 2*j), (2, 2*j)))
        # A22 (2,2) * B2j (2, 2j) -> C2j (2, 2j)
        products.append(((2, 2), (2, 2*j), (2, 2*j)))
        
    # Strassen terms definition
    # (coeff_U, coeff_V, coeff_W) for each of the 7 terms
    # U corresponds to A entries, V to B entries, W to C entries.
    # Indices in 2x2 block: 00, 01, 10, 11
    
    # M1: (X00 + X11)(Y00 + Y11) -> Z00, Z11
    # M2: (X10 + X11)Y00 -> Z10
    # M3: X00(Y01 - Y11) -> Z01
    # M4: X11(Y10 - Y00) -> Z00, Z10
    # M5: (X00 + X01)Y11 -> Z00, Z11
    # M6: (X10 - X00)(Y00 + Y01) -> Z11
    # M7: (X01 - X11)(Y10 + Y11) -> Z01, Z00, Z11
    
    # We will generate 7 rank-1 tensors for each product.
    
    for (ar, ac), (br, bc), (cr, cc) in products:
        # ar, ac: start row, col of A block
        # br, bc: start row, col of B block
        # cr, cc: start row, col of C block
        
        # Helper to get global index
        def idx_a(r, c): return (ar + r) * m + (ac + c)
        def idx_b(r, c): return (br + r) * p + (bc + c)
        def idx_c(r, c): return (cr + r) * n + (cc + c)
        
        # Term 1: M1 = (X00 + X11)(Y00 + Y11)
        # U: X00 + X11
        # V: Y00 + Y11
        # W: Z00 + Z11 (with coeff 1)
        U[idx_a(0,0)][col] += 1
        U[idx_a(1,1)][col] += 1
        V[idx_b(0,0)][col] += 1
        V[idx_b(1,1)][col] += 1
        W[idx_c(0,0)][col] += 1 # Z00
        W[idx_c(1,1)][col] += 1 # Z11
        col += 1
        
        # Term 2: M2 = (X10 + X11)Y00
        U[idx_a(1,0)][col] += 1
        U[idx_a(1,1)][col] += 1
        V[idx_b(0,0)][col] += 1
        W[idx_c(1,0)][col] += 1 # Z10
        col += 1
        
        # Term 3: M3 = X00(Y01 - Y11)
        U[idx_a(0,0)][col] += 1
        V[idx_b(0,1)][col] += 1
        V[idx_b(1,1)][col] += -1
        W[idx_c(0,1)][col] += 1 # Z01
        col += 1
        
        # Term 4: M4 = X11(Y10 - Y00)
        U[idx_a(1,1)][col] += 1
        V[idx_b(1,0)][col] += 1
        V[idx_b(0,0)][col] += -1
        W[idx_c(0,0)][col] += -1 # Z00
        W[idx_c(1,0)][col] += 1 # Z10
        col += 1
        
        # Term 5: M5 = (X00 + X01)Y11
        U[idx_a(0,0)][col] += 1
        U[idx_a(0,1)][col] += 1
        V[idx_b(1,1)][col] += 1
        W[idx_c(0,0)][col] += 1 # Z00
        W[idx_c(1,1)][col] += 1 # Z11
        col += 1
        
        # Term 6: M6 = (X10 - X00)(Y00 + Y01)
        U[idx_a(1,0)][col] += 1
        U[idx_a(0,0)][col] += -1
        V[idx_b(0,0)][col] += 1
        V[idx_b(0,1)][col] += 1
        W[idx_c(1,1)][col] += 1 # Z11
        col += 1
        
        # Term 7: M7 = (X01 - X11)(Y10 + Y11)
        U[idx_a(0,1)][col] += 1
        U[idx_a(1,1)][col] += -1
        V[idx_b(1,0)][col] += 1
        V[idx_b(1,1)][col] += 1
        W[idx_c(0,1)][col] += 1 # Z01
        W[idx_c(0,0)][col] += -1 # Z00
        W[idx_c(1,1)][col] += 1 # Z11
        col += 1
        
    return {"U": U, "V": V, "W": W}
